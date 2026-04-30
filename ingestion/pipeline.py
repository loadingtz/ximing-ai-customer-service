"""采集流水线主入口：discover（多 API 搜索）→ fetch（多 API 抓取）→ 抽取 → 落地 → build_index（推 Milvus）。

抽取两种模式：
- --mode llm（默认）：按 schema.py 24 节点契约 LLM 结构化抽取，需 ANTHROPIC_API_KEY，质量高
- --mode section    ：按 markdown 标题切原始段，每段一个 raw_section 节点，无需 key，可让 Milvus 跑起来

用法：
    python -m ingestion.pipeline                                # llm 模式（默认）
    python -m ingestion.pipeline --mode section                 # 无 key 也能填库
    python -m ingestion.pipeline --dry-run                      # 只列候选 URL
    python -m ingestion.pipeline --only-fetch URL ...           # 只抓不抽
    python -m ingestion.pipeline --reindex-only                 # 跳过抓取，从 nodes.jsonl 重建向量库

输出：
    data/knowledge/nodes.jsonl              抽取通过的节点
    data/knowledge/pending_review.jsonl     敏感/契约校验失败，待运营+法务复核
    data/knowledge/manifest.json            本次运行 metadata
    data/vector.db                          Milvus Lite 向量库
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .discover import SearchHit, discover
from .fetch import FetchedPage, fetch_many

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "knowledge"
SOURCES_YAML = ROOT / "ingestion" / "sources.yaml"


def _load_sources() -> dict:
    with open(SOURCES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _doc_id(node: dict) -> str:
    h = hashlib.sha1()
    h.update((node.get("source_url", "") + "|" + (node.get("text") or "")[:200]).encode("utf-8"))
    return h.hexdigest()[:16]


def _is_sensitive(node: dict, sensitive_keywords: list[str]) -> bool:
    if node.get("sensitive"):
        return True
    text = (node.get("text") or "") + " " + (node.get("topic") or "")
    return any(kw in text for kw in sensitive_keywords)


def _build_index_from_jsonl() -> int:
    """从 nodes.jsonl 重建 Milvus 向量库。"""
    from retrieval.vector_store import VectorStore

    nodes: list[dict] = []
    path = KB_DIR / "nodes.jsonl"
    if not path.exists():
        print("[index] nodes.jsonl 不存在，跳过向量索引")
        return 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))
    if not nodes:
        return 0

    vs = VectorStore()
    vs.reset()
    print(f"[index] 编码 {len(nodes)} 个节点 (model={vs.embedder.model_name}, dim={vs.embedder.dim}) …")
    n = vs.upsert(nodes)
    print(f"[index] 已写入 Milvus collection={vs.collection} URI={vs.client._using or 'lite-file'} count={n}")
    return n


async def _run_async(args) -> int:
    load_dotenv(ROOT / ".env")

    if args.reindex_only:
        n = _build_index_from_jsonl()
        print(f"[pipeline] reindex 完成：{n} 节点")
        return 0

    needs_llm = (args.mode == "llm") and not (args.only_fetch or args.dry_run)
    if needs_llm and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print(
            "ERROR: --mode llm 需要 ANTHROPIC_API_KEY 或 OPENROUTER_API_KEY。\n"
            "       要么填 .env，要么改用 --mode section（按 markdown 标题切原始段，无需 key）。",
            file=sys.stderr,
        )
        return 2

    cfg = _load_sources()
    policy = cfg.get("crawl_policy", {})
    sensitive_kw = cfg.get("sensitive_keywords", [])
    user_agent = os.getenv("XIMING_CRAWLER_UA") or policy.get("user_agent", "Ximing-CSBot/0.1")
    KB_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Discover
    if args.only_fetch:
        urls = list(args.only_fetch)
        hits_by_url: dict[str, dict] = {u: {"node_type_hint": None, "query": None} for u in urls}
    else:
        # 1A) 品牌特定查询：搜出来必须含"熹茗"
        brand_queries = list(cfg.get("search_queries", []))
        # 1B) 行业通用查询：不强制品牌过滤
        ref_queries = list(cfg.get("reference_queries", []))

        hits: list[SearchHit] = []
        if brand_queries:
            print(f"[pipeline] 搜索（品牌）{len(brand_queries)} 条 query …")
            hits.extend(discover(brand_queries, require_brand=True))
        if ref_queries:
            print(f"[pipeline] 搜索（行业通用）{len(ref_queries)} 条 query …")
            hits.extend(discover(ref_queries, require_brand=False))

        # 1C) seed_urls / reference_urls 直接加进来
        for s in cfg.get("seed_urls", []) or []:
            if s.get("enabled", True) and s.get("url"):
                hits.append(SearchHit(url=s["url"], title="", snippet="", query="seed",
                                      backend="seed", node_type_hint=s.get("node_type")))
        for s in cfg.get("reference_urls", []) or []:
            if s.get("url"):
                hits.append(SearchHit(url=s["url"], title="", snippet="", query="reference",
                                      backend="reference", node_type_hint=s.get("node_type_hint")))

        urls = [h.url for h in hits]
        hits_by_url = {h.url: {"node_type_hint": h.node_type_hint, "query": h.query, "backend": h.backend} for h in hits}

        # host_allowlist 强制过滤——防止 search 后端把 UGC 噪音带回来
        allowlist = cfg.get("host_allowlist") or []
        if allowlist:
            import urllib.parse
            before = len(urls)
            def _allowed(u: str) -> bool:
                host = urllib.parse.urlsplit(u).netloc.lower()
                return any(host == h or host.endswith("." + h) for h in allowlist)
            urls = [u for u in urls if _allowed(u)]
            hits_by_url = {u: hits_by_url[u] for u in urls}
            dropped = before - len(urls)
            if dropped:
                print(f"[pipeline] host_allowlist 过滤丢弃 {dropped} 个非白名单 URL")
        print(f"[pipeline] 共 {len(urls)} 个候选 URL")

    if args.dry_run:
        for u in urls:
            print(u)
        return 0

    if not urls:
        print("[pipeline] 没有候选 URL — 编辑 ingestion/sources.yaml 添加 seed_urls 或 search_queries。")
        return 1

    # 2) Fetch
    print(f"[pipeline] 抓取 {len(urls)} 页 …")
    pages: list[FetchedPage] = await fetch_many(
        urls,
        user_agent=user_agent,
        max_concurrent=int(policy.get("max_concurrent", 4)),
        per_host_delay_s=float(policy.get("per_host_delay_s", 1.5)),
        request_timeout_s=float(policy.get("request_timeout_s", 20)),
        respect_robots=bool(policy.get("respect_robots_txt", True)),
        max_pages=int(policy.get("max_pages_per_run", 200)),
    )
    print(f"[pipeline] 实际抓到 {len(pages)} 页")

    if args.only_fetch:
        for p in pages:
            print(f"\n=== {p.url} ===\n{p.text[:1000]}\n…(truncated)")
        return 0

    # 3) Extract——双模式
    # 把 sources.yaml 的 trust_level 也传下去（用于 section 模式的元数据；hits_by_url 已存 hint）
    seed_trust_by_url = {}
    for s in cfg.get("seed_urls", []) or []:
        if s.get("url"):
            seed_trust_by_url[s["url"]] = int(s.get("trust_level", 4))
    for s in cfg.get("reference_urls", []) or []:
        if s.get("url"):
            seed_trust_by_url[s["url"]] = int(s.get("trust_level", 3))

    if args.mode == "llm":
        from .extract import extract_nodes
        def extractor(page, hint):
            return extract_nodes(page, source_hint=hint)
    else:
        from .section_extract import split_sections
        def extractor(page, hint):
            return split_sections(
                page,
                node_type_hint=hint,
                trust_level=seed_trust_by_url.get(page.url, 4),
            )

    approved_path = KB_DIR / "nodes.jsonl"
    pending_path = KB_DIR / "pending_review.jsonl"
    n_ok, n_pending, n_total = 0, 0, 0
    seen_ids: set[str] = set()

    with approved_path.open("w", encoding="utf-8") as fout, pending_path.open("w", encoding="utf-8") as fpend:
        for p in pages:
            hint = (hits_by_url.get(p.url) or {}).get("node_type_hint")
            try:
                nodes = extractor(p, hint)
            except Exception as e:
                print(f"[pipeline] 抽取失败 {p.url}: {e}")
                continue
            for node in nodes:
                node.setdefault("doc_id", _doc_id(node))
                if node["doc_id"] in seen_ids:
                    continue
                seen_ids.add(node["doc_id"])
                n_total += 1
                if _is_sensitive(node, sensitive_kw):
                    node["needs_review"] = True
                    fpend.write(json.dumps(node, ensure_ascii=False) + "\n")
                    n_pending += 1
                else:
                    fout.write(json.dumps(node, ensure_ascii=False) + "\n")
                    n_ok += 1
            print(f"[pipeline] {p.url} → {len(nodes)} 节点")

    print(f"[pipeline] 抽取完成：已入库 {n_ok}，待人审 {n_pending}，写入 {approved_path}")

    # 4) Build index (Milvus)
    if not args.skip_index:
        try:
            indexed = _build_index_from_jsonl()
            print(f"[pipeline] 已建立向量索引：{indexed} 节点")
        except Exception as e:
            print(f"[pipeline] WARN 向量索引失败（先存 nodes.jsonl，可后续 --reindex-only 重建）：{e}")
            indexed = 0
    else:
        indexed = 0

    manifest = {
        "pages_fetched": len(pages),
        "nodes_total": n_total,
        "nodes_approved": n_ok,
        "nodes_pending_review": n_pending,
        "nodes_indexed": indexed,
        "fetch_backend": "multi (jina/tavily/firecrawl)",
        "search_backend": "multi (tavily/serpapi/jina-ddg/jina-bing/ddg-lib)",
        "extract_mode": args.mode,
        "model_llm": os.getenv("XIMING_MODEL", "claude-opus-4-7") if args.mode == "llm" else None,
        "embed_model": os.getenv("XIMING_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
    }
    (KB_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只列出候选 URL")
    ap.add_argument("--only-fetch", nargs="*", help="只抓取指定 URL 并打印纯文本")
    ap.add_argument("--mode", choices=["llm", "section"], default="llm",
                    help="llm: 按 schema 24 节点契约 LLM 抽（需 ANTHROPIC_API_KEY）；section: 按 markdown 标题切原始段（无需 key）")
    ap.add_argument("--skip-index", action="store_true", help="抽完不建向量索引")
    ap.add_argument("--reindex-only", action="store_true", help="跳过抓取与抽取，从 nodes.jsonl 重建向量库")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run_async(args)))


if __name__ == "__main__":
    main()
