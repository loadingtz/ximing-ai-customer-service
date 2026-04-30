"""主 Agent：意图分类 → 检索 → Tool Use 循环 → 合规过滤 → 输出。

设计要点：
- System Prompt 用 cache_control: ephemeral（system_base + 三个场景 prompt 都是稳定文本）。
- 工具循环手动管理，便于在 stop_reason='tool_use' 时插入审计日志和压测点。
- 每轮回复都过 safety/compliance_filter；命中违规 → 重写为安全话术 + need_human=true。
- 检索置信度 < 阈值 → 直接转人工（不让模型乱编）。
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import llm as _llm
from retrieval.hybrid_search import HybridIndex, Hit, Node
from safety import compliance_filter
from tools import business_tools

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
MODEL = os.getenv("XIMING_MODEL", "claude-opus-4-7")
CONFIDENCE_THRESHOLD = 0.18

# 用关键词做轻量意图分类——足够覆盖三类场景；生产可换 Haiku 4.5 做分类
INTENT_HINTS = {
    "recommend": ["推荐", "送礼", "送领导", "买什么", "选哪", "适合", "礼盒", "预算", "口粮", "新手"],
    "brewing":   ["怎么泡", "冲泡", "苦", "涩", "味淡", "出汤", "投茶", "水温", "盖碗", "煮饮"],
    "aftersale": ["发霉", "受潮", "破损", "退货", "退款", "退换", "漏发", "没收到", "投诉", "315", "起诉", "曝光"],
}

# 意图 → 优先检索的 node_type 集合（必须与 schema.py 实际类型对齐）
# 之前用了 ["product", "brand"] 等不存在的根名导致全过滤掉、走 fallback；
# 现在直接列实际 17 类，保证 product_sku（最具体的 SKU 信息）能命中。
INTENT_NODE_TYPES = {
    "recommend": [
        "product_sku", "product_category", "product_scene",
        "product_origin", "product_grade",
        "process_aroma", "process_flavor", "process_craft", "process_compare",
        "brand_story", "brand_collab",
    ],
    "brewing": [
        "brewing_params", "brewing_vessel",
        "brewing_category_tip", "brewing_troubleshoot",
        "process_aroma", "process_flavor",
    ],
    "aftersale": [
        "commerce_order", "commerce_logistics", "commerce_return_policy",
        "commerce_aftersale_sop", "commerce_membership", "advice_storage",
    ],
}

ESCALATION_KEYWORDS = ["315", "曝光", "起诉", "媒体", "投诉", "赔我", "没人管", "法院", "公安", "黑猫"]


@dataclass
class AgentResponse:
    reply: str
    cite: list[str] = field(default_factory=list)
    need_human: bool = False
    next_action: str | None = None
    intent: str | None = None
    confidence: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    raw_violations: list[str] = field(default_factory=list)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def classify_intent(user_msg: str) -> str:
    msg = user_msg.strip()
    scores = {k: sum(1 for w in ws if w in msg) for k, ws in INTENT_HINTS.items()}
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else "recommend"   # 默认按推荐入口


_BUDGET_PAT = re.compile(r"(?:预算|大概|约)?\s*([1-9]\d{2,4})\s*(?:元|块|左右|以内|以下|上下)?")
_PRICE_PAT = re.compile(r"([1-9]\d{1,5})\s*元")


def extract_budget(user_msg: str) -> int | None:
    """从用户消息抽出预算（元）。仅在 recommend 场景使用，兜底 None。"""
    # 抓所有候选数字，挑"预算/大概/左右/以内"附近的；否则取最大
    candidates = [int(m.group(1)) for m in _BUDGET_PAT.finditer(user_msg) if 100 <= int(m.group(1)) <= 100000]
    if not candidates:
        return None
    # 优先附近带"预算/左右/以内"上下文的
    for kw in ("预算", "左右", "以内", "以下"):
        if kw in user_msg:
            return candidates[0]
    return None


def _node_max_price(text: str) -> int | None:
    """节点正文里出现的最大价格（元）。礼盒/罐装混合时取最大值代表上限。"""
    nums = _PRICE_PAT.findall(text)
    return max(int(n) for n in nums) if nums else None


def budget_rerank(hits: list[Hit], budget: int | None) -> list[Hit]:
    """预算敏感：节点最大价 > budget×1.5 时分数 ×0.4 demote。
    避免"送领导预算 800"被 25000 元/斤的高端款占据 top-3。"""
    if not budget or not hits:
        return hits
    rescored: list[Hit] = []
    for h in hits:
        price = _node_max_price(h.node.text)
        score = h.score
        if price and price > budget * 1.5:
            score = score * 0.4
        rescored.append(Hit(node=h.node, score=score,
                            score_vector=h.score_vector, score_bm25=h.score_bm25))
    rescored.sort(key=lambda h: -h.score)
    return rescored


def detect_category(user_msg: str) -> str | None:
    for c in ("岩茶", "白茶", "红茶", "肉桂", "大红袍", "水仙", "白毫银针", "白牡丹", "金骏眉", "正山小种"):
        if c in user_msg:
            if c in ("肉桂", "大红袍", "水仙"):
                return "岩茶"
            if c in ("白毫银针", "白牡丹"):
                return "白茶"
            if c in ("金骏眉", "正山小种"):
                return "红茶"
            return c
    return None


def build_context_block(hits: list[Hit]) -> str:
    if not hits:
        return "<context>（检索结果为空）</context>"
    lines = ["<context>"]
    for i, h in enumerate(hits, 1):
        meta = h.node.meta
        lines.append(
            f"[doc_id={meta.get('doc_id', h.node.doc_id)} | "
            f"node_type={meta.get('node_type','?')} | "
            f"category={meta.get('category','?')} | "
            f"score={h.score:.2f} (vec={h.score_vector:.2f} bm25={h.score_bm25:.2f}) | "
            f"source={meta.get('source_url','')}]"
        )
        lines.append(h.node.text)
        lines.append("")
    lines.append("</context>")
    return "\n".join(lines)


class Agent:
    # 多轮对话历史（LRU + TTL，避免长跑内存泄漏）
    HISTORY_TURNS = 4              # 单 session 最多 4 轮 = 8 条消息
    HISTORY_MAX_SESSIONS = 500     # 最多缓存 500 个 session（约 < 5MB）
    HISTORY_TTL_SECONDS = 60 * 60  # 1 小时不活跃即清

    def __init__(self, index: HybridIndex | None = None):
        from collections import OrderedDict
        import threading
        self.index = index or HybridIndex.load()
        self._system_blocks = self._load_system_blocks()
        self._history: "OrderedDict[str, list[dict]]" = OrderedDict()
        self._history_ts: dict[str, float] = {}
        self._history_lock = threading.Lock()

    def _gc_history(self) -> None:
        """LRU 大小限制 + TTL 过期清理（轻量，每次写入时扫一下）。"""
        import time as _t
        now = _t.time()
        with self._history_lock:
            # TTL 过期
            stale = [sid for sid, ts in self._history_ts.items() if now - ts > self.HISTORY_TTL_SECONDS]
            for sid in stale:
                self._history.pop(sid, None)
                self._history_ts.pop(sid, None)
            # LRU：超过 max sessions 时弹最旧的
            while len(self._history) > self.HISTORY_MAX_SESSIONS:
                sid, _ = self._history.popitem(last=False)
                self._history_ts.pop(sid, None)

    def _hist_get(self, session_id: str) -> list[dict]:
        if not session_id:
            return []
        import time as _t
        with self._history_lock:
            if session_id in self._history:
                self._history.move_to_end(session_id)        # LRU touch
                self._history_ts[session_id] = _t.time()
                return list(self._history[session_id])
        return []

    def _hist_append(self, session_id: str, user_msg: str, reply: str) -> None:
        if not session_id:
            return
        import time as _t
        with self._history_lock:
            hist = self._history.get(session_id) or []
            hist.append({"role": "user", "content": user_msg})
            hist.append({"role": "assistant", "content": reply})
            self._history[session_id] = hist[-self.HISTORY_TURNS * 2:]
            self._history.move_to_end(session_id)
            self._history_ts[session_id] = _t.time()
        self._gc_history()

    def _load_system_blocks(self) -> list[dict]:
        # 把基座 + 三个场景 prompt 拼成一份稳定 system，并打 cache_control
        base = _read(PROMPTS / "system_base.md")
        recommend = _read(PROMPTS / "recommend.md")
        brewing = _read(PROMPTS / "brewing.md")
        aftersale = _read(PROMPTS / "aftersale.md")
        joined = "\n\n---\n\n".join([base, recommend, brewing, aftersale])
        return [{"type": "text", "text": joined, "cache_control": {"type": "ephemeral"}}]

    # ------------------------------------------------------------------
    def chat(self, user_msg: str, session_id: str | None = None) -> AgentResponse:
        # 1) 早期硬规则：升级关键词 → 直接转人工
        if any(k in user_msg for k in ESCALATION_KEYWORDS):
            business_tools.handoff_to_human(reason="用户出现升级关键词", transcript_excerpt=user_msg)
            return AgentResponse(
                reply="非常抱歉给您带来困扰。已为您接通专属顾问，请稍候，1 分钟内有人对接您。",
                need_human=True,
                intent="aftersale",
            )

        # 2) 意图 + 类目 → 检索（含上下文感知，与 chat_stream 行为一致）
        intent = classify_intent(user_msg)
        category = detect_category(user_msg)
        history_for_session = self._hist_get(session_id or "")
        if history_for_session:
            recent_text = " ".join(m.get("content", "") for m in history_for_session[-4:]
                                    if isinstance(m.get("content"), str))
            aftersale_signals = ["退货", "退款", "订单号", "发霉", "受潮", "不好喝", "破损", "漏发", "投诉", "赔"]
            if any(kw in recent_text for kw in aftersale_signals) and intent != "aftersale":
                intent = "aftersale"
                category = None
        if self.index.empty:
            hits: list[Hit] = []
        else:
            preferred = INTENT_NODE_TYPES.get(intent)
            # 先按意图过滤；不够 5 条再回退到全类型补足
            hits = self.index.search(user_msg, k=5, category=category, node_types=preferred)
            if len(hits) < 3:
                fallback = self.index.search(user_msg, k=5, category=category)
                seen = {h.node.doc_id for h in hits}
                for h in fallback:
                    if h.node.doc_id not in seen:
                        hits.append(h)
                        if len(hits) >= 5:
                            break

        # 预算敏感重排：query 含"预算 X" 时把远超预算的节点 demote
        if intent == "recommend":
            hits = budget_rerank(hits, extract_budget(user_msg))

        confidence = self.index.confidence(hits)
        context_block = build_context_block(hits)

        # 3) 置信度过低 + 知识库为空 → 转人工兜底
        if not hits and self.index.empty:
            return AgentResponse(
                reply="知识库还没准备好，我先帮您接通专属顾问处理。",
                need_human=True,
                intent=intent,
                confidence=0.0,
                next_action="run `python -m ingestion.pipeline` 先构建知识库",
            )

        # 4) Claude 工具循环（拼装 messages：历史轮次 + 本轮带 context）
        history = self._hist_get(session_id or "")[-self.HISTORY_TURNS * 2:]
        messages: list[dict] = list(history) + [
            {
                "role": "user",
                "content": (
                    f"当前场景意图：{intent}\n"
                    f"当前会话置信度：{confidence:.2f}\n\n"
                    f"{context_block}\n\n"
                    f"用户消息：{user_msg}"
                ),
            }
        ]

        tool_calls_log: list[dict] = []
        last = None
        for _ in range(5):   # 最多 5 轮工具调用
            last = _llm.chat(
                system=self._system_blocks,
                messages=messages,
                max_tokens=1024,
                model=_llm.default_model(fast=True),    # 客服用快模型 Haiku 4.5
                tools=business_tools.TOOL_SCHEMAS,
            )
            # assistant turn append（保持 anthropic-style content blocks 以兼容多轮）
            assistant_content: list[dict] = []
            if last["text"]:
                assistant_content.append({"type": "text", "text": last["text"]})
            for tc in last["tool_calls"]:
                assistant_content.append({
                    "type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"],
                })
            messages.append({"role": "assistant", "content": assistant_content})

            if not last["tool_calls"]:
                break

            tool_results = []
            for tc in last["tool_calls"]:
                result = business_tools.run_tool(tc["name"], tc["input"])
                tool_calls_log.append({"name": tc["name"], "input": tc["input"], "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_results})

        # 5) LLM 直接返回纯中文，不再走 JSON 包装
        reply = (last["text"] if last else "").strip()
        # 兜底：如果 LLM 仍输出了 JSON / 代码块，剥离取出 reply 字段
        if reply.startswith("{") or "```json" in reply[:50] or '"reply"' in reply[:80]:
            parsed = _parse_json_or_text(reply)
            if parsed.get("reply"):
                reply = parsed["reply"].strip()
        # cite 从 retrieval 命中直接取——客服层不需要让 LLM 重述
        cite = [h.node.meta.get("doc_id") or h.node.doc_id for h in hits[:3]]
        # need_human 由 tool_calls 或合规过滤器决定
        need_human = any(t.get("name") == "handoff_to_human" for t in tool_calls_log)
        next_action = None

        # 6) 出站合规过滤——与 chat_stream 一致：仅记录 warning，不打断 reply
        # 严重违规（疾病/药品宣称）才覆盖；轻量违规（绝对化）只 warning
        check = compliance_filter.check(reply)
        violations = []
        if not check.safe:
            violations = [f"{v.category}:{v.matched}" for v in check.violations]
            # 仅当命中疾病类（最严重）才覆盖话术
            has_disease = any(v.category == "disease" for v in check.violations)
            if has_disease:
                reply = (
                    "茶饮属于食品，无法提供健康疗效相关建议。"
                    "如有健康方面的问题，建议咨询医生。如需咨询茶品本身，我可以继续为您介绍。"
                )

        # 写入会话历史——下一轮自然能看到本轮上下文（与 chat_stream 一致）
        if session_id:
            self._hist_append(session_id, user_msg, reply)

        return AgentResponse(
            reply=reply,
            cite=cite,
            need_human=need_human,
            next_action=next_action,
            intent=intent,
            confidence=confidence,
            tool_calls=tool_calls_log,
            raw_violations=violations,
        )


    def chat_stream(self, user_msg: str, session_id: str | None = None):
        """流式生成器，含会话历史记忆（按 session_id）。

        - session_id 不为空时，把该 session 的最近 N 轮历史拼进 messages，让 LLM 看到上下文
        - 流结束时把这一轮的 user_msg + assistant_reply 追加到历史
        """
        # 1. 升级关键词硬规则
        if any(k in user_msg for k in ESCALATION_KEYWORDS):
            business_tools.handoff_to_human(reason="用户出现升级关键词", transcript_excerpt=user_msg)
            full = "非常抱歉给您带来困扰。已为您接通专属顾问，请稍候，1 分钟内有人对接您。"
            yield full
            yield {"meta": {"need_human": True, "intent": "aftersale", "cite": [], "confidence": 0.0}}
            return

        # 2. 意图 + 检索（含上下文感知）
        intent = classify_intent(user_msg)
        category = detect_category(user_msg)

        # 上下文感知：如果上一轮用户/AI 内容含售后关键词，本轮短回复应锁定 aftersale
        history_for_session = self._hist_get(session_id or "")
        if history_for_session:
            recent_text = " ".join(m.get("content", "") for m in history_for_session[-4:])
            aftersale_signals = ["退货", "退款", "订单号", "发霉", "受潮", "不好喝", "破损", "漏发", "投诉", "赔"]
            in_aftersale = any(kw in recent_text for kw in aftersale_signals)
            if in_aftersale and intent != "aftersale":
                intent = "aftersale"   # 强制锁定售后场景
                category = None        # 售后不限品类
        if self.index.empty:
            yield "知识库还没准备好，已为您接通专属顾问。"
            yield {"meta": {"need_human": True, "intent": intent, "cite": [], "confidence": 0.0}}
            return

        preferred = INTENT_NODE_TYPES.get(intent)
        hits = self.index.search(user_msg, k=5, category=category, node_types=preferred)
        if len(hits) < 3:
            fallback = self.index.search(user_msg, k=5, category=category)
            seen = {h.node.doc_id for h in hits}
            for h in fallback:
                if h.node.doc_id not in seen:
                    hits.append(h)
                    if len(hits) >= 5:
                        break
        if intent == "recommend":
            hits = budget_rerank(hits, extract_budget(user_msg))
        confidence = self.index.confidence(hits)
        context_block = build_context_block(hits)

        # 拼装 messages：历史轮次（轻量，不带 context_block）+ 本轮（带 context）
        history = self._hist_get(session_id or "")[-self.HISTORY_TURNS * 2:]
        messages = list(history) + [{
            "role": "user",
            "content": (
                f"当前场景意图：{intent}\n"
                f"会话置信度：{confidence:.2f}\n\n"
                f"{context_block}\n\n"
                f"用户消息：{user_msg}"
            ),
        }]

        # 3. 流式 LLM
        text_buf = []
        for delta in _llm.chat_stream(
            system=self._system_blocks,
            messages=messages,
            max_tokens=1024,
            model=_llm.default_model(fast=True),
        ):
            if isinstance(delta, dict) and delta.get("_final"):
                final_text = delta["text"]
                break
            text_buf.append(delta)
            yield delta

        full_reply = "".join(text_buf)
        # 出站合规检查——流式已经发完，不再覆盖 reply（撤回会让用户困惑）
        # 命中违规：仅在 meta 里标 warning + 后续可由 ops 流转人工，不打断当前对话
        check = compliance_filter.check(full_reply)
        violations_meta = [v.matched for v in check.violations] if not check.safe else []

        cite = [h.node.meta.get("doc_id") or h.node.doc_id for h in hits[:3]]

        # 写入会话历史——下一轮自然能看到本轮上下文（LRU + TTL 内置防泄漏）
        self._hist_append(session_id, user_msg, full_reply)

        yield {"meta": {
            "need_human": False, "intent": intent,
            "cite": cite, "confidence": round(confidence, 3),
            "compliance_warnings": violations_meta,    # 仅记录，不打断
        }}


def _parse_json_or_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {"reply": text}
