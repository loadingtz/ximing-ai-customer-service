"""多后端搜索发现层——融合多个 AI 工具 API。

后端优先级（自动 fallback）：
1. Tavily Search API     — 需 TAVILY_API_KEY，最稳，含 AI 摘要
2. SerpAPI              — 需 SERPAPI_API_KEY，覆盖百度/Google
3. Jina-via-DDG         — 通过 Jina Reader 抓 DuckDuckGo HTML 结果页（免 key）
4. Jina-via-Bing        — 通过 Jina Reader 抓 Bing 搜索结果页（免 key 兜底）
5. duckduckgo-search    — 老库，从这台沙盒外的常规 client 调；偶尔超时

任何一条 query 命中其中一个后端的有效结果即停止，
失败则按顺序往下试。
"""
from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

import httpx


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    query: str
    backend: str
    node_type_hint: str | None = None


# ─── 后端 1：Tavily ────────────────────────────────────────────────
def _tavily(query: str, max_results: int) -> list[dict]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    r = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return [
        {"url": it["url"], "title": it.get("title", ""), "snippet": it.get("content", "")}
        for it in data.get("results", [])[:max_results]
    ]


# ─── 后端 2：SerpAPI ───────────────────────────────────────────────
def _serpapi(query: str, max_results: int) -> list[dict]:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        return []
    r = httpx.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": key, "engine": "baidu", "num": max_results},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("organic_results") or []
    return [
        {"url": it.get("link"), "title": it.get("title", ""), "snippet": it.get("snippet", "")}
        for it in items[:max_results]
        if it.get("link")
    ]


# ─── 后端 3+4：Jina Reader 抓搜索结果页 ─────────────────────────────
_JINA = "https://r.jina.ai/"

_DDG_LINK = re.compile(r"##\s*\[([^\]]+)\]\(https://duckduckgo\.com/l/\?uddg=([^&]+)[^)]*\)")
_BING_LINK = re.compile(r"##\s*\[([^\]]+)\]\((https?://[^)]+?)\)")


def _jina_via_search_engine(engine: str, query: str, max_results: int) -> list[dict]:
    if engine == "ddg":
        target = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        regex = _DDG_LINK
    else:
        target = "https://www.bing.com/search?q=" + urllib.parse.quote_plus(query)
        regex = _BING_LINK
    headers = {}
    if jk := os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {jk}"
    r = httpx.get(_JINA + target, headers=headers, timeout=30)
    r.raise_for_status()
    text = r.text

    results: list[dict] = []
    for match in regex.finditer(text):
        title = match.group(1).strip()
        raw_url = match.group(2)
        if engine == "ddg":
            try:
                url = urllib.parse.unquote(raw_url)
            except Exception:
                continue
        else:
            url = raw_url
            # Bing 结果会混进很多内部 ck/redirect 链接 + Bing 自身 URL，过滤
            if "bing.com" in url or "bingj.com" in url or "msn.com" in url or url.startswith("javascript"):
                continue
        # 段落里下一段是 snippet（启发式：title 后第一段非链接文本）
        # 简化处理：从原文中找 title 后 200 字
        idx = text.find(match.group(0))
        snippet_zone = text[idx + len(match.group(0)) : idx + len(match.group(0)) + 600]
        # 取首段干净文本
        snippet = ""
        for line in snippet_zone.split("\n"):
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("!["):
                continue
            snippet = line
            break
        results.append({"url": url, "title": title, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


# ─── 后端 5：duckduckgo-search 老库 ─────────────────────────────────
def _duckduckgo(query: str, max_results: int) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region="cn-zh"))
        return [
            {"url": it.get("href") or it.get("url"), "title": it.get("title", ""), "snippet": it.get("body", "")}
            for it in results
            if it.get("href") or it.get("url")
        ]
    except Exception:
        return []


# ─── 编排：按优先级 fallback ───────────────────────────────────────
BRAND_KEYWORDS = ("熹茗", "ximing", "ximingcha", "ximingtea")


def _is_brand_relevant(url: str, title: str, snippet: str, query: str) -> bool:
    blob = " ".join([url, title, snippet, query]).lower()
    return any(kw.lower() in blob for kw in BRAND_KEYWORDS)


_ALL_BACKENDS = [
    ("tavily", _tavily),
    ("serpapi", _serpapi),
    ("jina-ddg", lambda q, n: _jina_via_search_engine("ddg", q, n)),
    ("jina-bing", lambda q, n: _jina_via_search_engine("bing", q, n)),
    ("duckduckgo-lib", _duckduckgo),
]


def discover(queries: Iterable[dict], *, require_brand: bool = True) -> list[SearchHit]:
    """queries: [{q, max_results, node_type_hint?}, ...]

    require_brand=True：召回结果必须含"熹茗"才保留（用于品牌特定检索）；
    reference_urls 类查询应传 require_brand=False。
    """
    hits: list[SearchHit] = []
    seen: set[str] = set()
    dropped = 0
    backend_stats: dict[str, int] = {}
    for entry in queries:
        q = entry["q"]
        n = int(entry.get("max_results", 5))
        # 按优先级 fallback：第一个返回非空就用它
        used_backend = None
        used_results: list[dict] = []
        for name, fn in _ALL_BACKENDS:
            try:
                results = fn(q, n)
            except Exception as e:
                print(f"[search] {name} 异常 q={q!r}: {e}")
                continue
            if results:
                used_backend, used_results = name, results
                backend_stats[name] = backend_stats.get(name, 0) + 1
                break
        if not used_results:
            print(f"[search] 所有后端失败 q={q!r}")
            continue

        for r in used_results:
            url = r["url"]
            if not url or url in seen:
                continue
            seen.add(url)
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if require_brand and not _is_brand_relevant(url, title, snippet, q):
                dropped += 1
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=title,
                    snippet=snippet,
                    query=q,
                    backend=used_backend,
                    node_type_hint=entry.get("node_type_hint"),
                )
            )

    if dropped:
        print(f"[search] 品牌相关性过滤丢弃 {dropped} 条")
    if backend_stats:
        print(f"[search] 后端使用次数：{backend_stats}")
    return hits
