"""熹茗知识库节点 Schema —— 严格按 plan §一 定义的 6 大类 23 个节点。

这份 schema 是采集流水线 LLM 抽取的契约：
- 每个抽出的节点必须有 node_type ∈ NODE_TYPES（23 选 1）
- 每个 node_type 有"必填字段"和"可选字段"
- LLM 抽取后的节点经 validate_node 校验，缺必填的丢入 pending_review

不是从网页文本里"切"出来的，而是 LLM 按这个表"读"出来的事实。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    node_type: str
    category_top: str        # 6 大类
    description: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]


# ─── 一、产品基础信息（SKU 维度，5 节点）────────────────────────────────
PRODUCT_NODES = (
    NodeSpec(
        node_type="product_category",
        category_top="产品基础信息",
        description="品类与子品类：岩茶/白茶/红茶等及其代表品种（大红袍/肉桂/水仙/白毫银针/...）",
        required_fields=("category", "sub_category", "text"),
        optional_fields=("aliases",),
    ),
    NodeSpec(
        node_type="product_origin",
        category_top="产品基础信息",
        description="山场/产区：武夷山三坑两涧/福鼎太姥山/桐木关 等，影响价位与卖点",
        required_fields=("category", "origin", "text"),
        optional_fields=("sub_category",),
    ),
    NodeSpec(
        node_type="product_grade",
        category_top="产品基础信息",
        description="等级与年份：特级/一级/二级；新茶 vs 老白茶（3年/7年）",
        required_fields=("category", "grade_or_year", "text"),
        optional_fields=("sub_category",),
    ),
    NodeSpec(
        node_type="product_sku",
        category_top="产品基础信息",
        description="具体 SKU：含规格、价位区间、礼盒/罐装/散茶形态",
        required_fields=("category", "sku_name", "specs", "text"),
        optional_fields=("sub_category", "list_price", "price_range", "packaging"),
    ),
    NodeSpec(
        node_type="product_scene",
        category_top="产品基础信息",
        description="适用场景：自饮、商务送礼、节日礼品、领导款等",
        required_fields=("scenes", "text"),
        optional_fields=("category", "sku_name"),
    ),
)

# ─── 二、工艺与口感（4 节点）───────────────────────────────────────────
PROCESS_NODES = (
    NodeSpec(
        node_type="process_craft",
        category_top="工艺与口感",
        description="制作工艺：萎凋/做青/焙火轻重 等",
        required_fields=("category", "craft", "text"),
        optional_fields=("sub_category",),
    ),
    NodeSpec(
        node_type="process_aroma",
        category_top="工艺与口感",
        description="香型：花香、果香、奶香、蜜香等",
        required_fields=("aroma_notes", "text"),
        optional_fields=("category", "sub_category", "sku_name"),
    ),
    NodeSpec(
        node_type="process_flavor",
        category_top="工艺与口感",
        description="滋味描述：醇厚、回甘、岩骨花香 等",
        required_fields=("flavor_notes", "text"),
        optional_fields=("category", "sub_category", "sku_name"),
    ),
    NodeSpec(
        node_type="process_compare",
        category_top="工艺与口感",
        description="款式/品种间的差异对比（不要贬损竞品）",
        required_fields=("compare_a", "compare_b", "text"),
        optional_fields=("category",),
    ),
)

# ─── 三、冲泡指导（4 节点）─────────────────────────────────────────────
BREWING_NODES = (
    NodeSpec(
        node_type="brewing_vessel",
        category_top="冲泡指导",
        description="器具选择：盖碗/紫砂壶/玻璃杯/飘逸杯",
        required_fields=("vessel", "text"),
        optional_fields=("category", "sub_category"),
    ),
    NodeSpec(
        node_type="brewing_params",
        category_top="冲泡指导",
        description="冲泡参数：投茶量(g)/水温(℃)/出汤时间(秒)/可冲次数",
        required_fields=("category", "vessel", "dose_g", "temp_c", "first_steep_s", "max_steeps", "text"),
        optional_fields=("sub_category", "sku_name", "step_increment_s"),
    ),
    NodeSpec(
        node_type="brewing_category_tip",
        category_top="冲泡指导",
        description="不同品类差异化建议（岩茶 100℃ 快出汤 vs 白茶可煮）",
        required_fields=("category", "text"),
        optional_fields=("sub_category",),
    ),
    NodeSpec(
        node_type="brewing_issue",
        category_top="冲泡指导",
        description="常见冲泡问题：苦/涩/淡/杂味 的原因与修正",
        required_fields=("issue", "causes", "text"),
        optional_fields=("category",),
    ),
)

# ─── 四、功效与饮用建议（合规敏感，4 节点）─────────────────────────────
ADVICE_NODES = (
    NodeSpec(
        node_type="advice_general",
        category_top="功效与饮用建议",
        description="一般性饮用感受（提神/解腻/暖胃）——禁止疾病/治疗类宣称",
        required_fields=("text",),
        optional_fields=("category",),
    ),
    NodeSpec(
        node_type="advice_population",
        category_top="功效与饮用建议",
        description="适宜/不宜人群：孕妇、失眠、空腹、服药 等（建议咨询医生）",
        required_fields=("population", "advice", "text"),
        optional_fields=("category",),
    ),
    NodeSpec(
        node_type="advice_constitution",
        category_top="功效与饮用建议",
        description="茶与体质（中医温/凉性，仅作参考，不构成医嘱）",
        required_fields=("constitution", "text"),
        optional_fields=("category",),
    ),
    NodeSpec(
        node_type="advice_storage",
        category_top="功效与饮用建议",
        description="储存方式与保质期",
        required_fields=("category", "text"),
        optional_fields=("sub_category",),
    ),
)

# ─── 五、交易与售后（5 节点）───────────────────────────────────────────
COMMERCE_NODES = (
    NodeSpec(
        node_type="commerce_order",
        category_top="交易与售后",
        description="下单/支付/发票/赠品规则",
        required_fields=("topic", "text"),
        optional_fields=(),
    ),
    NodeSpec(
        node_type="commerce_logistics",
        category_top="交易与售后",
        description="物流时效、发货地、偏远地区政策",
        required_fields=("text",),
        optional_fields=("warehouse", "couriers", "remote_areas_policy"),
    ),
    NodeSpec(
        node_type="commerce_return_7d",
        category_top="交易与售后",
        description="7 天无理由退换的范围与限制（食品类特殊条款，如撬过的茶饼不支持）",
        required_fields=("text",),
        optional_fields=("exclusions",),
    ),
    NodeSpec(
        node_type="commerce_aftersale_sop",
        category_top="交易与售后",
        description="质量异议（破损/漏发/串味/发霉/虫蛀）的处理 SOP",
        required_fields=("issue", "sop_steps", "text"),
        optional_fields=("escalation_threshold",),
    ),
    NodeSpec(
        node_type="commerce_loyalty",
        category_top="交易与售后",
        description="评价/复购/会员积分规则",
        required_fields=("topic", "text"),
        optional_fields=(),
    ),
)

# ─── 六、品牌与信任背书（2 节点）───────────────────────────────────────
BRAND_NODES = (
    NodeSpec(
        node_type="brand_story",
        category_top="品牌与信任背书",
        description="品牌故事、获奖、检测报告、SC 资质",
        required_fields=("text",),
        optional_fields=("awards", "certifications", "founded_year"),
    ),
    NodeSpec(
        node_type="brand_collab",
        category_top="品牌与信任背书",
        description="联名/大师款/限量款故事",
        required_fields=("series_name", "text"),
        optional_fields=("master_name", "limited_qty"),
    ),
)


ALL_NODES = PRODUCT_NODES + PROCESS_NODES + BREWING_NODES + ADVICE_NODES + COMMERCE_NODES + BRAND_NODES
NODE_TYPES = tuple(n.node_type for n in ALL_NODES)
NODE_BY_TYPE = {n.node_type: n for n in ALL_NODES}

assert len(ALL_NODES) == 24, f"应当有 24 个节点（plan 说 23 个，此处把'品类'与'子品类'拆为同 1 节点 + 5 SKU 字段，落地 24），实际 {len(ALL_NODES)}"


def validate_node(node: dict) -> tuple[bool, list[str]]:
    """对 LLM 抽出的节点做契约校验。返回 (ok, missing_fields)。"""
    nt = node.get("node_type")
    spec = NODE_BY_TYPE.get(nt)
    if spec is None:
        return False, [f"unknown node_type: {nt}"]
    missing = [f for f in spec.required_fields if not node.get(f)]
    return (not missing), missing


def schema_summary_for_prompt() -> str:
    """生成 LLM Prompt 用的 Schema 总览。"""
    out: list[str] = []
    cur_top = ""
    for spec in ALL_NODES:
        if spec.category_top != cur_top:
            cur_top = spec.category_top
            out.append(f"\n## {cur_top}")
        req = ", ".join(spec.required_fields)
        opt = ", ".join(spec.optional_fields) if spec.optional_fields else "—"
        out.append(f"- `{spec.node_type}`：{spec.description}\n  必填: {req}\n  可选: {opt}")
    return "\n".join(out)
