"""LLM 抽取层——严格按 schema.py 的 6 大类 24 个节点契约抽取。

输入：FetchedPage（Jina Reader 返回的 markdown）
输出：list[node_dict]，每个节点必有 node_type ∈ schema.NODE_TYPES，且必填字段齐全
       校验失败的节点丢入 pending_review 由人审

为什么 LLM 而非规则：
- 规则解析必须为每个网页模板写正则，脆弱、难维护
- LLM 可以"读"出隐含信息（"朱熹长房第 27 世后人朱陈松 2009 创立"→ brand_story 节点 + 创立年份字段）
- 同一段文字可能产出多个 node_type（一段产品介绍含 SKU + 工艺 + 香型）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# 让 ingestion 包能 import agent.llm
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent import llm as _llm

from .fetch import FetchedPage
from .schema import NODE_TYPES, schema_summary_for_prompt, validate_node


def _build_schema_doc() -> str:
    return f"""你是熹茗茶叶知识库的"抽取员"。从输入页面（已转为干净 markdown）中识别 0..N 个**结构化知识节点**并输出 JSON。

【可识别的 node_type，共 {len(NODE_TYPES)} 个，分 6 大类】
{schema_summary_for_prompt()}

【输出契约】
- 仅输出 JSON：{{"nodes": [...]}}
- 每个 node 必有 `node_type`（必须是上面 24 个之一）
- 必填字段缺一即视为无效；可选字段尽量填，但不要编造
- text 字段是"给客服读者看的≤200字自包含中文段落"，要把该节点的关键事实写清楚（不要复述原文一长串）
- 同一段原文可以产出多个 node（例如产品页可能同时有 product_sku + process_aroma + brewing_params）
- 与品牌"熹茗"无关的内容直接丢弃，返回空 nodes

【合规红线（命中即在该 node 里加 "sensitive": true，且 text 里删除违规说法）】
- 疾病治疗 / 预防 / 治愈 / 抗癌 / 降三高 / 包治
- 绝对化用语：最、第一、国家级、独家、唯一
- 医疗 / 药品类宣称

【JSON 输出规范】
- 仅输出 JSON：{{"nodes": [...]}}，不要 markdown 围栏、不要注释。
- text/字符串字段中**禁止使用直角双引号 " 包裹品牌名/系列名/概念名**。
  这会破坏 JSON 解析（最常见的事故源）。
  改用：
    全角引号「」（推荐）/ 书名号《》/ 不加引号
  错误：  "text": "品牌定位为"朱子后人""
  正确：  "text": "品牌定位为「朱子后人」"
  正确：  "text": "品牌定位为 朱子后人"
- 节点数 ≤ 8 条；每条 text ≤ 200 字（避免输出超长被截断）。
- 数组里如必须含双引号，用 \\" 转义。
"""


def extract_nodes(page: FetchedPage, *, source_hint: str | None = None) -> list[dict[str, Any]]:
    if not page.text.strip():
        return []
    body = page.text[:24000]
    user = (
        f"页面 URL: {page.url}\n"
        f"页面标题: {page.title}\n"
        f"来源类型提示: {source_hint or '未知'}\n"
        f"------ 页面 markdown ------\n{body}"
    )
    resp = _llm.chat(
        system=[{"type": "text", "text": _build_schema_doc(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        max_tokens=8192,                      # 之前 4096 不够，抽出多节点会被截断
        json_object=True,                     # 让 LLM 强制返回严格 JSON
    )
    text = (resp.get("text") or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # 容错修复：LLM 经常在中文 text 里嵌入未转义的 "（如 "品牌"朱子后人""）
        repaired = _repair_unescaped_inner_quotes(text)
        try:
            data = json.loads(repaired)
            print(f"[extract] JSON 修复成功 {page.url}（替换内嵌引号为「」）")
        except json.JSONDecodeError as e2:
            print(f"[extract] JSON 解析失败 {page.url}: {e}\n----\n{text[:500]}")
            return []

    nodes_in = data.get("nodes") or []
    valid: list[dict[str, Any]] = []
    invalid_count = 0
    for n in nodes_in:
        ok, missing = validate_node(n)
        if not ok:
            invalid_count += 1
            n["needs_review"] = True
            n["validation_missing"] = missing
        n.setdefault("source_url", page.url)
        n.setdefault("source_title", page.title)
        n.setdefault("captured_at", _today())
        n.setdefault("trust_level", 4)
        valid.append(n)
    if invalid_count:
        print(f"[extract] {page.url}: {invalid_count}/{len(nodes_in)} 节点契约校验失败 → pending_review")
    return valid


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _repair_unescaped_inner_quotes(s: str) -> str:
    """LLM 常在 "text": "...含双引号..." 里嵌入未转义的 "。
    判定 " 是否为 JSON 结构性引号：紧跟 :, } 或 ]、, 或换行 → 结构性；否则属于 inner，转成「或」。
    """
    import re
    out = []
    in_string = False
    escape = False
    n = len(s)
    for i, ch in enumerate(s):
        if escape:
            out.append(ch); escape = False; continue
        if ch == "\\":
            out.append(ch); escape = True; continue
        if ch == '"':
            if not in_string:
                # 进入字符串
                in_string = True
                out.append(ch); continue
            # 当前在字符串内——判断这个 " 是结构性闭合还是 inner 引号
            # 看后续非空白字符：若是 , : } ] 则视为结构性；否则视为 inner
            j = i + 1
            while j < n and s[j] in " \t":
                j += 1
            nxt = s[j] if j < n else ""
            if nxt in (",", ":", "}", "]", "\n", "\r", ""):
                in_string = False
                out.append(ch); continue
            # inner 引号 → 替换成「或」
            # 启发式：如果出现位置奇数次（开），用「；偶数次（闭），用」
            # 简化：交替用 「 和 」
            if not hasattr(_repair_unescaped_inner_quotes, "_state"):
                _repair_unescaped_inner_quotes._state = {}
            st = _repair_unescaped_inner_quotes._state.setdefault(id(out), {"open": True})
            out.append("「" if st["open"] else "」")
            st["open"] = not st["open"]
            continue
        out.append(ch)
    return "".join(out)
