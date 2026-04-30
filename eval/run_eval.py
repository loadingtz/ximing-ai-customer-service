"""黄金集回归。每周 / 每次 prompt 改动后跑一次。

通过 HTTP /chat 调用——避免和 uvicorn 抢 Milvus 文件锁。

用法：
    # 1) 先启动 uvicorn： python -m uvicorn app:app --port 8000
    # 2) 跑评测：
    python eval/run_eval.py
    python eval/run_eval.py --limit 3        # 只测前 3 题
    python eval/run_eval.py --base http://...:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.jsonl"


def chat_once(base: str, message: str, session_id: str | None = None, timeout: int = 60) -> dict:
    body = json.dumps({"message": message, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", type=str, default="http://127.0.0.1:8000")
    args = ap.parse_args()

    # 健康检查——服务必须在跑
    try:
        with urllib.request.urlopen(args.base.rstrip("/") + "/healthz", timeout=5) as r:
            ok = json.loads(r.read())
        print(f"✓ 后端在 {args.base} 运行（{ok.get('kb_nodes')} 节点）")
    except Exception as e:
        print(f"✗ 后端未启动 / 不可达：{args.base} ({e})", file=sys.stderr)
        print(f"  请先：python -m uvicorn app:app --port 8000", file=sys.stderr)
        return 2

    cases: list[dict] = []
    with GOLDEN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]

    total, ok, wrong_intent, compliance_hits = 0, 0, 0, 0
    fail_cases: list[dict] = []

    for c in cases:
        total += 1
        t0 = time.time()
        try:
            resp = chat_once(args.base, c["input"])
        except urllib.error.HTTPError as e:
            fail_cases.append({"id": c["id"], "error": f"HTTP {e.code}"})
            continue
        except Exception as e:
            fail_cases.append({"id": c["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        elapsed = time.time() - t0

        reply = resp.get("reply") or ""
        passed = True
        reasons: list[str] = []

        # must_contain_any: list of list（外 AND, 内 OR）—— 每组至少命中一个近义词
        for group in c.get("must_contain_any", []):
            if not any(w in reply for w in group):
                passed = False
                reasons.append(f"缺[{ '|'.join(group) }]中任一词")
        # must_contain: 旧格式向后兼容（严格字面匹配）
        for w in c.get("must_contain", []):
            if w not in reply:
                passed = False
                reasons.append(f"缺关键词:{w}")
        for w in c.get("must_not_contain", []):
            if w in reply:
                passed = False
                compliance_hits += 1
                reasons.append(f"违禁词:{w}")
        if c.get("expect_human") is not None and bool(c["expect_human"]) != bool(resp.get("need_human")):
            passed = False
            reasons.append(f"need_human 期望 {c['expect_human']} 实际 {resp.get('need_human')}")
        if resp.get("intent") and resp.get("intent") != c.get("intent"):
            wrong_intent += 1

        marker = "✅" if passed else "❌"
        print(f"  {marker} [{c['id']}] {elapsed:.1f}s  {c['input'][:30]}")
        if passed:
            ok += 1
        else:
            fail_cases.append({"id": c["id"], "reply": reply[:200], "reasons": reasons})

    print(f"\n=== 黄金集回归（{total} 题）===")
    print(f"  通过率:    {ok}/{total} = {ok / total * 100:.1f}%")
    print(f"  意图错分:  {wrong_intent}/{total}")
    print(f"  违禁词命中: {compliance_hits}")
    if fail_cases:
        print("\n失败用例：")
        for c in fail_cases:
            print(f"  [{c['id']}] {c.get('reasons') or c.get('error')}")
            if c.get("reply"):
                print(f"    reply: {c['reply'][:120]}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
