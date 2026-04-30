"""LLM 抽象层——同时支持 Anthropic 直连和 OpenRouter（OpenAI 兼容）。

后端选择优先级：
1. OPENROUTER_API_KEY 设置 → 走 OpenRouter（OpenAI compatible API）
   - model 默认 anthropic/claude-sonnet-4.5
   - tools 转 OpenAI tools format
2. ANTHROPIC_API_KEY 设置 → 走 anthropic SDK 直连
   - model 默认 claude-opus-4-7
3. 都没设 → raise RuntimeError

返回 unified dict:
    {"text": str, "tool_calls": [{id, name, input}], "stop_reason": str}
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


def _backend() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def default_model(*, fast: bool = False) -> str:
    """fast=True 时优先 XIMING_CHAT_MODEL（客服会话用快模型），否则 XIMING_MODEL（抽取用慢但准的）。"""
    if fast:
        if env := os.getenv("XIMING_CHAT_MODEL"):
            return env
        # 默认客服模型：Haiku（4 秒 vs Sonnet 9 秒）
        if _backend() == "openrouter":
            return "anthropic/claude-haiku-4.5"
        return "claude-haiku-4-5"
    if env := os.getenv("XIMING_MODEL"):
        return env
    if _backend() == "openrouter":
        return "anthropic/claude-sonnet-4.5"
    return "claude-opus-4-7"


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """anthropic tools → OpenAI tools 格式。"""
    out = []
    for t in anthropic_tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


def _system_to_string(system: Any) -> str:
    """anthropic system block list → 单个 string（OpenRouter 不支持 cache_control）。"""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n\n".join(parts)
    return str(system)


def _flatten_messages(messages: list[dict]) -> list[dict]:
    """anthropic messages content 可能是 list (含 tool_use/tool_result)，转 OpenAI 格式。"""
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        # content 是 list of blocks
        text_parts = []
        tool_calls = []
        for blk in content:
            # SDK 对象（响应块）→ 用属性；dict（手构）→ 用 .get
            btype = getattr(blk, "type", None) or (blk.get("type") if isinstance(blk, dict) else None)
            if btype == "text":
                text_parts.append(getattr(blk, "text", None) or blk.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": getattr(blk, "id", None) or blk.get("id"),
                    "type": "function",
                    "function": {
                        "name": getattr(blk, "name", None) or blk.get("name"),
                        "arguments": json.dumps(
                            getattr(blk, "input", None) or blk.get("input", {}),
                            ensure_ascii=False,
                        ),
                    },
                })
            elif btype == "tool_result":
                # tool_result 的 content 在 OpenAI 里要单独的 message role=tool
                out.append({
                    "role": "tool",
                    "tool_call_id": getattr(blk, "tool_use_id", None) or blk.get("tool_use_id"),
                    "content": getattr(blk, "content", None) or blk.get("content", ""),
                })
        if role == "assistant":
            msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "user" and (text_parts or not tool_calls):
            # tool_result 已经单独 append 过了；剩下的 user 文本/混合内容
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
    return out


def chat_stream(
    *,
    system: Any,
    messages: list[dict],
    max_tokens: int = 1024,
    model: Optional[str] = None,
):
    """流式生成——只 yield 文本 delta；用于客服 chat 体感优化。
    生成器结束后，函数最后一次 yield 一个 dict {"_final": True, "text":..., "stop_reason":...}
    内部不支持 tools；调用方在 stream 完成后若需 tool 循环，再走 chat() 非 stream 版。
    """
    backend = _backend()
    if backend != "openrouter":
        # anthropic 路径（用 SDK stream API）
        import anthropic
        client = anthropic.Anthropic()
        kwargs = {"model": model or default_model(fast=True), "max_tokens": max_tokens, "system": system, "messages": messages}
        text_acc = []
        with client.messages.stream(**kwargs) as s:
            for delta in s.text_stream:
                text_acc.append(delta)
                yield delta
            final = s.get_final_message()
        yield {"_final": True, "text": "".join(text_acc), "stop_reason": final.stop_reason}
        return

    # OpenRouter / OpenAI compatible
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    sys_str = _system_to_string(system)
    oai_msgs = [{"role": "system", "content": sys_str}] + _flatten_messages(messages)
    text_acc = []
    finish = "end_turn"
    stream = client.chat.completions.create(
        model=model or default_model(fast=True),
        messages=oai_msgs,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.delta and ch.delta.content:
            text_acc.append(ch.delta.content)
            yield ch.delta.content
        if ch.finish_reason:
            finish = ch.finish_reason
    yield {"_final": True, "text": "".join(text_acc), "stop_reason": finish}


def chat(
    *,
    system: Any,
    messages: list[dict],
    max_tokens: int = 1024,
    model: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    json_object: bool = False,
) -> dict:
    """统一 chat 接口。"""
    backend = _backend()
    if backend == "none":
        raise RuntimeError(
            "未配置 LLM key——填 OPENROUTER_API_KEY 或 ANTHROPIC_API_KEY"
        )
    model = model or default_model()

    if backend == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        sys_str = _system_to_string(system)
        oai_msgs = [{"role": "system", "content": sys_str}] + _flatten_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_msgs,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenRouter call failed: {e}") from e
        choice = resp.choices[0]
        msg = choice.message
        tc_list = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tc_list.append({"id": tc.id, "name": tc.function.name, "input": args})
        return {
            "text": msg.content or "",
            "tool_calls": tc_list,
            "stop_reason": choice.finish_reason or "end_turn",
            "_raw": resp,
            "_backend": "openrouter",
            "_model": model,
        }

    # anthropic 直连
    import anthropic
    client = anthropic.Anthropic()
    kwargs = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    resp = client.messages.create(**kwargs)
    text = next((b.text for b in resp.content if b.type == "text"), "")
    tool_calls = [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in resp.content
        if b.type == "tool_use"
    ]
    return {
        "text": text,
        "tool_calls": tool_calls,
        "stop_reason": resp.stop_reason,
        "_raw": resp,
        "_backend": "anthropic",
        "_model": model,
    }
