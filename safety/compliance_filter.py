"""出站合规过滤：拦截功效宣称、绝对化用语、医疗/疾病话术。

双层：
1) 关键词正则——快、可解释，覆盖《广告法》和食品广告常见违规词
2) 留口子调用小模型做语义复核（默认未启用，留接口）
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 疾病/医疗类（食品广告法严禁）
DISEASE_TERMS = [
    r"治[愈疗好]",
    r"防治",
    r"抗癌|防癌|肿瘤",
    r"降三高|降血压|降血脂|降血糖|降胆固醇",
    r"治(?:疗|愈)?(?:糖尿病|高血压|心脏病|失眠|脱发|便秘)",
    r"包治",
    r"延年益寿",
    r"消炎|杀菌(?:作用)?",
    r"抗病毒",
]

# 绝对化用语（《广告法》第九条）——只命中明确违规组合，避免误伤客服中性用法
# "最佳冲泡参数" / "盖碗 110ml 最佳" 是中性推荐，不算广告法绝对化
ABSOLUTE_TERMS = [
    # "最 X 品牌/茶/产品" 才是广告法红线
    r"最[好佳优棒]\s*(?:的)?\s*(?:茶|品牌|岩茶|白茶|红茶|普洱|乌龙|品质|产品|选择)",
    r"全国\s*(?:最|第一)",
    r"第一品牌",
    r"国家级\s*(?:茶|品牌|品质)",
    r"独家\s*(?:秘方|技术|工艺)",
    r"唯一\s*(?:认证|品牌)",
    r"顶级\s*(?:品质|品牌|享受)",
    r"史上最",
    r"100%(?:有效|纯天然)",
]

# 神秘/夸大类
EXAGGERATION_TERMS = [
    r"神奇功效",
    r"立竿见影",
    r"包(?:瘦|赢|对)",
    r"不(?:得不|可不)信",
]

# 竞品贬损（占位，按需扩展）
COMPETITOR_DISS = [
    r"比.{0,20}(?:好喝|强|靠谱)得多",
]

ALL_PATTERNS = [(p, "disease") for p in DISEASE_TERMS] \
             + [(p, "absolute") for p in ABSOLUTE_TERMS] \
             + [(p, "exaggeration") for p in EXAGGERATION_TERMS] \
             + [(p, "competitor") for p in COMPETITOR_DISS]


@dataclass
class Violation:
    category: str
    matched: str


@dataclass
class FilterResult:
    safe: bool
    violations: list[Violation]
    sanitized: str   # 把违规词替换为 ▢ 后的文本（兜底用）


def check(text: str) -> FilterResult:
    vs: list[Violation] = []
    sanitized = text
    for pat, cat in ALL_PATTERNS:
        for m in re.finditer(pat, text):
            vs.append(Violation(category=cat, matched=m.group(0)))
            sanitized = sanitized.replace(m.group(0), "▢" * len(m.group(0)))
    return FilterResult(safe=not vs, violations=vs, sanitized=sanitized)
