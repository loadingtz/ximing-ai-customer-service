"""业务工具：订单查询、物流查询、售后工单创建、转人工。

每个函数都是真实的 HTTP/数据库调用占位——示意如何与电商平台 ERP/OMS 对接。
默认走环境变量 XIMING_OMS_BASE_URL；未配置时返回结构化的 NOT_CONFIGURED 错误，
让 Agent 自然地降级为"暂未连通系统，请提供订单号截图"的话术。

工具描述（tool schemas）即下方 TOOL_SCHEMAS——给 Claude 的 messages.create(tools=...)。
"""
from __future__ import annotations

import os
from typing import Any

import httpx

OMS_BASE = os.getenv("XIMING_OMS_BASE_URL")
OMS_TOKEN = os.getenv("XIMING_OMS_TOKEN")


def _not_configured() -> dict[str, Any]:
    return {
        "status": "NOT_CONFIGURED",
        "message": "OMS 接入信息未配置（XIMING_OMS_BASE_URL / XIMING_OMS_TOKEN）。"
                   "请告知用户暂时无法实时查询，引导其提供订单号截图，由人工客服跟进。",
    }


def _call(path: str, params: dict | None = None, payload: dict | None = None, method: str = "GET") -> dict[str, Any]:
    if not OMS_BASE:
        return _not_configured()
    headers = {"Authorization": f"Bearer {OMS_TOKEN}"} if OMS_TOKEN else {}
    url = OMS_BASE.rstrip("/") + path
    try:
        if method == "GET":
            r = httpx.get(url, params=params, headers=headers, timeout=10)
        else:
            r = httpx.request(method, url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        return {"status": "OK", "data": r.json()}
    except httpx.HTTPError as e:
        return {"status": "ERROR", "message": str(e)}


# ---- 工具实现 -------------------------------------------------

def order_query(order_id: str) -> dict[str, Any]:
    return _call("/orders", params={"order_id": order_id})


def logistics_query(order_id: str | None = None, tracking_no: str | None = None) -> dict[str, Any]:
    return _call("/logistics", params={k: v for k, v in {"order_id": order_id, "tracking_no": tracking_no}.items() if v})


def stock_check(sku_name: str) -> dict[str, Any]:
    return _call("/inventory", params={"sku": sku_name})


def aftersale_ticket(order_id: str, issue: str, photos: list[str] | None = None, severity: str = "normal") -> dict[str, Any]:
    return _call(
        "/aftersale/tickets",
        method="POST",
        payload={"order_id": order_id, "issue": issue, "photos": photos or [], "severity": severity},
    )


def handoff_to_human(reason: str, transcript_excerpt: str = "") -> dict[str, Any]:
    return _call(
        "/handoff",
        method="POST",
        payload={"reason": reason, "transcript": transcript_excerpt},
    )


# ---- 给 Claude 的工具 schemas ----------------------------------

TOOL_SCHEMAS = [
    {
        "name": "order_query",
        "description": "通过订单号查询订单详情（下单时间、商品、金额、收货地址、状态）。仅当用户提供了订单号时才调用。",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "logistics_query",
        "description": "查询订单物流轨迹（已发货、已签收、运输中、异常等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "tracking_no": {"type": "string", "description": "快递单号（可选）"},
            },
        },
    },
    {
        "name": "stock_check",
        "description": "检查 SKU 当前库存与活动价。回答价格/是否有货前必须调用，禁止凭知识库内的旧价格作答。",
        "input_schema": {
            "type": "object",
            "properties": {"sku_name": {"type": "string", "description": "SKU 名称（如 正岩肉桂礼盒 250g）"}},
            "required": ["sku_name"],
        },
    },
    {
        "name": "aftersale_ticket",
        "description": "为质量异议（受潮/发霉/串味/破损）创建售后工单。须先拿到订单号 + 用户拍照后再调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "issue": {"type": "string", "description": "问题描述：发霉/受潮/串味/虫蛀/破损/漏发等"},
                "photos": {"type": "array", "items": {"type": "string"}, "description": "用户提供的图片 URL/标识"},
                "severity": {"type": "string", "enum": ["normal", "high"], "description": "金额>500、媒体身份、315 关键词出现 → high"},
            },
            "required": ["order_id", "issue"],
        },
    },
    {
        "name": "handoff_to_human",
        "description": "转人工坐席。触发条件：用户情绪激动、投诉、索赔、媒体身份、连续 2 轮未解决、AI 置信度低。",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "为什么转人工"},
                "transcript_excerpt": {"type": "string", "description": "最近 2-3 轮对话摘要，便于人工接手"},
            },
            "required": ["reason"],
        },
    },
]


TOOL_DISPATCH = {
    "order_query": lambda **kw: order_query(**kw),
    "logistics_query": lambda **kw: logistics_query(**kw),
    "stock_check": lambda **kw: stock_check(**kw),
    "aftersale_ticket": lambda **kw: aftersale_ticket(**kw),
    "handoff_to_human": lambda **kw: handoff_to_human(**kw),
}


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"status": "ERROR", "message": f"unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"status": "ERROR", "message": f"invalid arguments: {e}"}
