"""熹茗 AI 客服——FastAPI Web 服务。

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

接口：
    GET  /             静态首页（Web UI）
    POST /chat         {message, session_id?} → {reply, cite, need_human, intent, confidence, tool_calls?}
    GET  /stats        知识库 / 模型信息
    GET  /healthz      存活探针
"""
from __future__ import annotations

import os
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import sys
sys.path.insert(0, str(ROOT))

from agent.orchestrator import Agent
from retrieval.hybrid_search import HybridIndex

app = FastAPI(title="熹茗 AI 客服", description="基于 RAG + Tool Use 的茶叶电商客服 · 小茗", version="0.1")

# 单例：进程启动时加载 Agent + 索引；session_id → 历史 deque（最多 6 轮）
_index: Optional[HybridIndex] = None
_agent: Optional[Agent] = None
_history: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=12))   # 6 轮 × 2 条


def _get_agent() -> Agent:
    global _index, _agent
    if _agent is None:
        _index = HybridIndex.load()
        _agent = Agent(index=_index)
    return _agent


def _autoreindex_if_needed() -> None:
    """clone-and-run 友好：如果 vector.db 不存在或为空，但 nodes.jsonl 有内容，
    自动重建一次 Milvus 索引——让 clone 之后 git 跟踪的 nodes.jsonl 立即可用。
    """
    nodes_jsonl = ROOT / "data" / "knowledge" / "nodes.jsonl"
    if not nodes_jsonl.exists() or nodes_jsonl.stat().st_size < 100:
        return    # 没数据可重建
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        if vs.count() > 0:
            return    # Milvus 已经有数据
        # 自动 reindex
        print("[startup] 检测到 nodes.jsonl 但 Milvus 为空，自动 reindex …")
        from ingestion.pipeline import _build_index_from_jsonl
        n = _build_index_from_jsonl()
        print(f"[startup] 自动 reindex 完成：{n} 节点 → Milvus")
    except Exception as e:
        print(f"[startup] 自动 reindex 失败（可手动 python -m ingestion.pipeline --reindex-only）: {e}")


@app.on_event("startup")
def _warmup() -> None:
    _autoreindex_if_needed()
    _get_agent()    # 进程启动就加载，避免首次请求长等待


# ───── Models ─────────────────────────────────────────────────────
class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
    debug: bool = False


class ChatOut(BaseModel):
    reply: str
    session_id: str
    cite: list[str] = []
    need_human: bool = False
    intent: Optional[str] = None
    confidence: float = 0.0
    tool_calls: list[dict] = []
    raw_violations: list[str] = []
    next_action: Optional[str] = None


# ───── Endpoints ──────────────────────────────────────────────────
@app.get("/healthz")
def healthz() -> dict:
    agent = _get_agent()
    return {"ok": True, "kb_nodes": len(agent.index.nodes)}


@app.get("/stats")
def stats() -> dict:
    agent = _get_agent()
    nodes = agent.index.nodes
    by_cat: dict[str, int] = {}
    by_host: dict[str, int] = {}
    for n in nodes:
        cat = n.meta.get("category") or "未知"
        by_cat[cat] = by_cat.get(cat, 0) + 1
        src = n.meta.get("source_url", "")
        host = src.split("//")[-1].split("/")[0] if "//" in src else "unknown"
        by_host[host] = by_host.get(host, 0) + 1
    sku_pages = {n.meta.get("source_url") for n in nodes if "showPro.aspx" in (n.meta.get("source_url") or "")}
    # LLM 后端识别
    if os.getenv("OPENROUTER_API_KEY"):
        llm_backend = "openrouter"
        llm_model = os.getenv("XIMING_MODEL", "anthropic/claude-sonnet-4.5")
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm_backend = "anthropic"
        llm_model = os.getenv("XIMING_MODEL", "claude-opus-4-7")
    else:
        llm_backend, llm_model = None, None
    return {
        "kb_nodes": len(nodes),
        "sku_pages": len(sku_pages),
        "by_category": dict(sorted(by_cat.items(), key=lambda x: -x[1])),
        "by_host_top5": dict(sorted(by_host.items(), key=lambda x: -x[1])[:5]),
        "embed_model": os.getenv("XIMING_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "llm_ready": llm_backend is not None,
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
    }


@app.get("/nodes")
def list_nodes(
    q: Optional[str] = None,
    category: Optional[str] = None,
    node_type: Optional[str] = None,
    host: Optional[str] = None,
    sku_only: bool = False,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """列表/搜索 Milvus 节点。q 走 hybrid_search；其余字段走纯过滤。"""
    agent = _get_agent()
    nodes = agent.index.nodes
    limit = max(1, min(200, limit))

    def _row(n, score=None, score_v=None, score_b=None) -> dict:
        m = n.meta
        return {
            "doc_id": n.doc_id,
            "score": round(score, 3) if score is not None else None,
            "score_vector": round(score_v, 3) if score_v is not None else None,
            "score_bm25": round(score_b, 3) if score_b is not None else None,
            "trust_level": int(m.get("trust_level") or 3),
            "text": n.text[:300],
            "category": m.get("category"),
            "node_type": m.get("node_type"),
            "source_url": m.get("source_url"),
            "heading_path": m.get("heading_path"),
        }

    # 1) 关键词检索 → 命中后再过滤
    if q:
        hits = agent.index.search(q, k=200, category=category)
        items = []
        for h in hits:
            n, m = h.node, h.node.meta
            if node_type and m.get("node_type") != node_type:
                continue
            if host and host not in (m.get("source_url") or ""):
                continue
            if sku_only and "showPro.aspx" not in (m.get("source_url") or ""):
                continue
            items.append(_row(n, h.score, h.score_vector, h.score_bm25))
        total = len(items)
        return {"total": total, "offset": 0, "limit": limit, "items": items[:limit]}

    # 2) 纯过滤：category / node_type / host / sku_only
    items = []
    for n in nodes:
        m = n.meta
        if category and m.get("category") != category:
            continue
        if node_type and m.get("node_type") != node_type:
            continue
        if host and host not in (m.get("source_url") or ""):
            continue
        if sku_only and "showPro.aspx" not in (m.get("source_url") or ""):
            continue
        items.append(_row(n))
    # 浏览模式按 trust_level desc 排，让权威源排前
    items.sort(key=lambda r: -r["trust_level"])
    total = len(items)
    return {"total": total, "offset": offset, "limit": limit, "items": items[offset:offset + limit]}


@app.get("/nodes/{doc_id}")
def get_node(doc_id: str) -> dict:
    agent = _get_agent()
    for n in agent.index.nodes:
        if n.doc_id == doc_id:
            return {"doc_id": n.doc_id, "text": n.text, "meta": n.meta}
    raise HTTPException(404, "doc_id not found")


@app.get("/pending")
def list_pending(limit: int = 50) -> dict:
    """命中敏感词、待人审的节点。"""
    import json as _json
    path = ROOT / "data" / "knowledge" / "pending_review.jsonl"
    if not path.exists():
        return {"total": 0, "items": []}
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(_json.loads(line))
            except _json.JSONDecodeError:
                pass
    return {"total": len(items), "items": items[:limit]}


@app.post("/chat/stream")
async def chat_stream(req: ChatIn):
    """SSE 流式 /chat——异步路径无 buffer，首字节真实 ≈ OpenRouter TTFT 2-3s。"""
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "message 不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:12]
    agent = _get_agent()

    import json as _json
    import anyio
    from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream

    send_stream, recv_stream = anyio.create_memory_object_stream(max_buffer_size=64)

    def producer():
        """同步线程里跑 agent.chat_stream，把 chunks 推到 anyio 流。"""
        try:
            for chunk in agent.chat_stream(msg, session_id=session_id):
                if isinstance(chunk, dict):
                    payload = chunk.get("meta", {})
                    payload["session_id"] = session_id
                    line = f"data: {_json.dumps({'meta': payload}, ensure_ascii=False)}\n\n"
                else:
                    line = f"data: {_json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
                anyio.from_thread.run(send_stream.send, line)
        except Exception as e:
            err = f"LLM 调用失败：{type(e).__name__} {e}"
            try:
                anyio.from_thread.run(
                    send_stream.send,
                    f"data: {_json.dumps({'delta': err, 'error': True}, ensure_ascii=False)}\n\n",
                )
            except Exception:
                pass
        finally:
            try:
                anyio.from_thread.run(send_stream.send, "data: [DONE]\n\n")
                anyio.from_thread.run(send_stream.aclose)
            except Exception:
                pass

    async def streamer():
        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, producer)
            async for line in recv_stream:
                yield line

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn) -> ChatOut:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "message 不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:12]

    agent = _get_agent()
    try:
        resp = agent.chat(msg, session_id=session_id)
    except Exception as e:
        # 没 ANTHROPIC_API_KEY 时降级——直接展示检索 top-3，证明 RAG 链路工作
        hits = agent.index.search(msg, k=3)
        if hits:
            reply_lines = [
                f"⚠️ 当前未配置 ANTHROPIC_API_KEY（{type(e).__name__}），"
                "无法走完整 LLM 生成。下面是 RAG 检索到的 top-3 相关知识——"
                "配置 key 后即可由 Claude Opus 4.7 综合成话术：\n",
            ]
            for i, h in enumerate(hits, 1):
                src = h.node.meta.get("source_url", "")
                reply_lines.append(f"\n【{i}】 score={h.score:.2f}  {src}")
                reply_lines.append(h.node.text[:240])
            return ChatOut(
                reply="\n".join(reply_lines),
                session_id=session_id,
                cite=[h.node.meta.get("doc_id") or h.node.doc_id for h in hits[:3]],
                confidence=round(agent.index.confidence(hits), 3),
            )
        return ChatOut(
            reply=f"暂时无法生成回复（{type(e).__name__}: {e}）。已为您接通专属顾问。",
            session_id=session_id,
            need_human=True,
        )

    # 记录历史（仅做日志，本版没把历史回写给 LLM）
    h = _history[session_id]
    h.append({"role": "user", "content": msg})
    h.append({"role": "assistant", "content": resp.reply})

    return ChatOut(
        reply=resp.reply,
        session_id=session_id,
        cite=resp.cite,
        need_human=resp.need_human,
        intent=resp.intent,
        confidence=round(resp.confidence, 3),
        tool_calls=resp.tool_calls if req.debug else [],
        raw_violations=resp.raw_violations if req.debug else [],
        next_action=resp.next_action,
    )


# ───── Static UI ──────────────────────────────────────────────────
WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"hint": "POST /chat with {message:'你好'}", "stats": "/stats", "health": "/healthz", "admin": "/admin"})


@app.get("/admin")
def admin_page() -> FileResponse:
    page = WEB_DIR / "admin.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(404, "admin page missing")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
