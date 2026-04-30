"""多后端资料获取层——融合多个 AI 工具 API。

后端优先级（自动 fallback）：
1. Jina Reader     — 免 key 默认，r.jina.ai/<url> 返回干净 markdown
2. Tavily Extract  — 需 TAVILY_API_KEY；对反爬严的页面更可靠
3. Firecrawl       — 需 FIRECRAWL_API_KEY；JS 渲染最强

任何一个后端拿到长度 ≥ 200 字的 markdown 即停止；
全部失败时 retry 最多 3 次再放弃。
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_JINA_BASE = os.getenv("XIMING_READER_BASE", "https://r.jina.ai/")


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    fetched_at: float
    status: int
    backend: str


# ─── 后端 1：Jina Reader ────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
async def _jina(client: httpx.AsyncClient, url: str) -> tuple[str, str, int]:
    headers = {}
    if jk := os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {jk}"
    r = await client.get(_JINA_BASE.rstrip("/") + "/" + url, headers=headers)
    r.raise_for_status()
    return _parse_jina(r.text), "", r.status_code


def _parse_jina(raw: str) -> str:
    # 分段解析：Title / URL Source / Markdown Content
    title = ""
    body_lines: list[str] = []
    in_body = False
    for line in raw.splitlines():
        if not in_body:
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
            elif line.startswith("Markdown Content:"):
                in_body = True
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip() if in_body else raw.strip()
    return f"@@TITLE@@{title}\n{body}"


# ─── 后端 2：Tavily Extract ─────────────────────────────────────────
async def _tavily(client: httpx.AsyncClient, url: str) -> tuple[str, str, int]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    r = await client.post(
        "https://api.tavily.com/extract",
        json={"api_key": key, "urls": [url], "extract_depth": "advanced"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("results") or []
    if not items:
        raise RuntimeError("Tavily empty result")
    return f"@@TITLE@@\n{items[0].get('raw_content', '')}", "", 200


# ─── 后端 3：Firecrawl ──────────────────────────────────────────────
async def _firecrawl(client: httpx.AsyncClient, url: str) -> tuple[str, str, int]:
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        raise RuntimeError("FIRECRAWL_API_KEY not set")
    r = await client.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers={"Authorization": f"Bearer {key}"},
        json={"url": url, "formats": ["markdown"]},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json().get("data") or {}
    md = data.get("markdown") or ""
    title = data.get("metadata", {}).get("title", "")
    return f"@@TITLE@@{title}\n{md}", "", 200


_BACKENDS: list[tuple[str, Callable]] = [
    ("jina", _jina),
    ("tavily", _tavily),
    ("firecrawl", _firecrawl),
]


def _split_title(payload: str) -> tuple[str, str]:
    if payload.startswith("@@TITLE@@"):
        nl = payload.find("\n")
        title = payload[len("@@TITLE@@"):nl].strip()
        body = payload[nl + 1:].strip()
        return title, body
    return "", payload.strip()


async def fetch_one(
    client: httpx.AsyncClient, url: str, *, min_len: int = 200
) -> FetchedPage | None:
    """按 _BACKENDS 顺序 fallback；任一返回 ≥min_len 字符即停止。"""
    last_err = None
    for name, fn in _BACKENDS:
        try:
            payload, _, status = await fn(client, url)
            title, body = _split_title(payload)
            if len(body) >= min_len:
                return FetchedPage(
                    url=url, title=title, text=body, fetched_at=time.time(),
                    status=status, backend=name,
                )
        except Exception as e:
            last_err = (name, e)
            continue
    if last_err:
        print(f"[fetch] 全部后端失败 {url} (last: {last_err[0]} {last_err[1]})")
    return None


async def fetch_many(
    urls: Iterable[str],
    *,
    user_agent: str = "Ximing-CSBot/0.1",
    max_concurrent: int = 4,
    per_host_delay_s: float = 1.0,
    request_timeout_s: float = 60,
    respect_robots: bool = True,   # 由 Jina/Tavily/Firecrawl 各自处理
    max_pages: int | None = None,
) -> list[FetchedPage]:
    sem = asyncio.Semaphore(max_concurrent)
    last_hit = [0.0]
    results: list[FetchedPage] = []
    backend_stats: dict[str, int] = {}

    headers = {"User-Agent": user_agent, "Accept-Language": "zh-CN,zh;q=0.9"}
    async with httpx.AsyncClient(headers=headers, timeout=request_timeout_s) as client:
        async def _one(url: str) -> None:
            async with sem:
                wait = per_host_delay_s - (time.time() - last_hit[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                last_hit[0] = time.time()
                page = await fetch_one(client, url)
                if page:
                    results.append(page)
                    backend_stats[page.backend] = backend_stats.get(page.backend, 0) + 1
                    # 每抓 10 页打一次进度，让用户看得到
                    if len(results) % 10 == 0:
                        print(f"[fetch] 进度 {len(results)} / {len(urls_list)} (后端: {backend_stats})", flush=True)

        urls_list = list(urls)
        if max_pages:
            urls_list = urls_list[:max_pages]
        await asyncio.gather(*[_one(u) for u in urls_list])

    if backend_stats:
        print(f"[fetch] 后端使用次数：{backend_stats}")
    return results
