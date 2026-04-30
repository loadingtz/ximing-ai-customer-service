"""无 LLM 切分——按 markdown 标题层级把页面切成"语义段（section）"，写入 Milvus。

⚠️ 重要：section 抽取产出的不是 schema.py 24 节点中的"结构化节点"，而是带元数据的**原始段**。
它的角色是：在没有 ANTHROPIC_API_KEY 时也能让 Milvus 跑起来、检索能命中相关内容；
LLM 抽取（extract.py）才会进一步把每个 section 抽成 24 个 schema 节点之一。

每个 section 字段：
  doc_id          source_url + section_idx 哈希
  text            section 正文（≤500 字，超长按句号切）
  source_type     brand / industry / private / unknown
  category        从原文 + URL 启发推断（岩茶/白茶/红茶/通用）
  node_type       raw_section（明确表明非 schema 节点）
  trust_level     来自 sources.yaml 的入参
  source_url      原 URL
  source_title    页面标题
  heading_path    "H1 > H2 > H3" 标题路径
  needs_llm_extract True   提示后续 LLM pass 还应该跑一遍
"""
from __future__ import annotations

import hashlib
import re

from .fetch_backends import FetchedPage

CATEGORY_KEYWORDS = {
    "岩茶": ["岩茶", "大红袍", "肉桂", "水仙", "武夷岩", "正岩", "三坑两涧", "牛魁", "岩凹", "金汤玉露", "焙火"],
    "白茶": ["白茶", "白毫银针", "白牡丹", "寿眉", "贡眉", "福鼎", "太姥山", "磻山雪芽", "老白茶"],
    "红茶": ["红茶", "金骏眉", "正山小种", "桐木关", "赤凝香"],
    "普洱": ["普洱", "莽野", "生普", "熟普", "陈化"],
    "绿茶": ["龙井", "西湖龙井"],
    "乌龙": ["乌龙", "铁观音", "醉雍隆"],
    "茉莉": ["茉莉", "子茉", "窨制"],
}

# ─── node_type 启发式推断（无 LLM 时用，对应 schema.py 的 6 大类）──
# 优先级从上到下，命中即返回
NODE_TYPE_RULES = [
    # 1. 售后政策 / 交易（commerce_*）
    ("commerce_return_7d",      ["7天无理由", "七天无理由", "无理由退换", "无理由退货"]),
    ("commerce_aftersale_sop",  ["发霉", "受潮", "破损", "漏发", "串味", "虫蛀", "投诉", "质量问题"]),
    ("commerce_logistics",      ["物流", "发货", "顺丰", "中通", "快递", "签收", "偏远地区"]),
    ("commerce_order",          ["发票", "下单", "支付", "赠品", "购物指南"]),
    ("commerce_loyalty",        ["积分", "会员", "复购", "评价"]),
    # 2. 冲泡（brewing_*）
    ("brewing_issue",           ["发苦", "苦涩", "味淡", "杂味", "不好喝", "为什么.*苦"]),
    ("brewing_params",          ["水温", "投茶量", "出汤", "几泡", "几道", "克", "盖碗", "冲泡"]),
    ("brewing_vessel",          ["盖碗", "紫砂壶", "玻璃杯", "飘逸杯"]),
    # 3. 饮用建议（advice_*）
    ("advice_storage",          ["储存", "保存", "密封", "避光", "防潮", "保质期", "陈化"]),
    ("advice_population",       ["孕妇", "哺乳期", "失眠", "空腹", "服药", "禁忌", "不宜"]),
    ("advice_constitution",     ["茶性", "温性", "凉性", "体质", "中医"]),
    ("advice_general",          ["提神", "解腻", "暖胃", "饮用感受"]),
    # 4. 工艺与口感（process_*）
    ("process_aroma",           ["桂皮香", "花香", "果香", "蜜香", "兰花香", "毫香", "栗香"]),
    ("process_flavor",          ["回甘", "醇厚", "岩骨花香", "甘润", "鲜爽"]),
    ("process_craft",           ["萎凋", "做青", "焙火", "炭焙", "工艺", "制作"]),
    # 5. 品牌（brand_*）
    ("brand_collab",            ["联名", "限量款", "大师款", "纪念款", "生肖"]),
    ("brand_story",             ["创立", "创始人", "朱陈松", "朱熹", "荣获", "获奖", "认证", "指定用茶", "龙头企业", "成立于"]),
    # 6. 产品（product_*）
    ("product_grade",           ["特级", "一级", "二级", "等级"]),
    ("product_origin",          ["三坑两涧", "慧苑坑", "牛栏坑", "马枕峰", "桐木关", "太姥山", "山场", "产区"]),
    ("product_sku",             ["规格", "净含量", "g/盒", "克/盒", "克/泡", "元/盒", "元/罐", "礼盒"]),
    ("product_category",        ["品类", "品种"]),
]


def _infer_node_type(text: str, url: str, hint: str | None, heading_path: str) -> str:
    """根据正文 / URL / sources.yaml hint / 标题路径 推断 node_type。"""
    blob = (heading_path + "\n" + text).lower()
    text_full = heading_path + "\n" + text

    # 1) URL 强提示
    if "showPro.aspx" in url:
        # SKU 详情页：默认 product_sku，但若正文重头讲品牌故事就转 brand_story
        for nt, kws in NODE_TYPE_RULES:
            if nt.startswith("product_") or nt.startswith("process_"):
                if any(re.search(kw, text_full) for kw in kws):
                    return nt
        return "product_sku"
    if "shouhou" in url or "/help" in url or "foodmate" in url:
        for nt, kws in NODE_TYPE_RULES:
            if nt.startswith("commerce_"):
                if any(kw in text_full for kw in kws):
                    return nt
        return "commerce_aftersale_sop"
    if "/paofa/" in url or "wulongchazhishi" in url or "ipucha" in url:
        for nt, kws in NODE_TYPE_RULES:
            if nt.startswith("brewing_"):
                if any(kw in text_full for kw in kws):
                    return nt
        return "brewing_params"

    # 2) 正文规则：按优先级命中第一条
    for nt, kws in NODE_TYPE_RULES:
        if any(re.search(kw, text_full) for kw in kws):
            return nt

    # 3) 退到 hint
    if hint:
        # 把 sources.yaml 的简写 hint 映射到 schema node_type
        hint_map = {
            "brand": "brand_story",
            "product": "product_sku",
            "product_list": "product_sku",
            "policy": "commerce_aftersale_sop",
            "brewing": "brewing_params",
            "faq": "commerce_order",
            "drinking_advice": "advice_general",
        }
        if hint in hint_map:
            return hint_map[hint]

    # 4) 最后兜底
    return "raw_section"

MAX_SECTION_LEN = 500
MIN_SECTION_LEN = 30   # 中文信息密度高，30 字已足够一个有意义段


def _detect_category(text: str, url: str = "") -> str:
    blob = text + " " + url
    counts = {cat: sum(1 for kw in kws if kw in blob) for cat, kws in CATEGORY_KEYWORDS.items()}
    counts = {k: v for k, v in counts.items() if v}
    if not counts:
        return "通用"
    return max(counts.items(), key=lambda x: x[1])[0]


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_NOISE_PATTERNS = [
    re.compile(r"^\[.*\]\(.*\)$"),     # 纯 markdown 链接
    re.compile(r"^!\[.*?\]"),          # 图片
    re.compile(r"^[-*]\s*\["),         # 链接列表项
    re.compile(r"copyright|版权|京ICP|沪ICP|闽ICP|备案号|skip to content", re.I),
]


def _clean(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    for p in _NOISE_PATTERNS:
        if p.match(line) or p.search(line):
            return ""
    # 行内链接降级为纯文本
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_sections(page: FetchedPage, node_type_hint: str | None = None, trust_level: int = 4) -> list[dict]:
    """按 markdown 标题切段。无标题时按双换行切。"""
    if not page.text:
        return []

    text = page.text
    headings = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []   # (heading_path, body)

    if headings:
        heading_stack: list[str] = []
        for i, m in enumerate(headings):
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
            heading_path = " > ".join(heading_stack)
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end]
            sections.append((heading_path, body))
    else:
        # 无标题：按双换行切
        for chunk in re.split(r"\n\s*\n+", text):
            sections.append(("", chunk))

    # 清洗 + 截短
    out: list[dict] = []
    for idx, (path, body) in enumerate(sections):
        cleaned_lines = [c for c in (_clean(l) for l in body.split("\n")) if c]
        clean_body = "\n".join(cleaned_lines).strip()
        if len(clean_body) < MIN_SECTION_LEN:
            continue
        # 超长按句号切
        for sub in _split_long(clean_body):
            if len(sub) < MIN_SECTION_LEN:
                continue
            out.append(_to_node(page, idx, path, sub, node_type_hint, trust_level))
    return out


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_SECTION_LEN:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[。！？!?])", text):
        if not sent:
            continue
        if len(cur) + len(sent) <= MAX_SECTION_LEN:
            cur += sent
        else:
            if cur:
                pieces.append(cur)
            cur = sent
    if cur:
        pieces.append(cur)
    return pieces


def _to_node(page: FetchedPage, idx: int, heading_path: str, body: str,
             node_type_hint: str | None = None, trust_level: int = 4) -> dict:
    h = hashlib.sha1((page.url + "|" + str(idx) + "|" + body[:120]).encode("utf-8")).hexdigest()[:16]
    full_text = (heading_path + "\n" + body) if heading_path else body
    category = _detect_category(full_text, page.url)
    node_type = _infer_node_type(body, page.url, node_type_hint, heading_path)
    return {
        "doc_id":     h,
        "node_type":  node_type,
        "category":   category,
        "text":       (heading_path + "：" + body) if heading_path else body,
        "trust_level": trust_level,
        "source_url": page.url,
        "source_title": page.title,
        "heading_path": heading_path,
        "needs_llm_extract": node_type == "raw_section",   # 仍是兜底类型才需 LLM 升级
    }
