"""熹茗 AI 客服 CLI——多轮交互式问答。

用法：
    python cli.py                  # 进入交互式
    python cli.py --once "问题"     # 单条问答（可脚本化）
    python cli.py --debug          # 打印检索分数、工具调用、置信度
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator import Agent
from retrieval.hybrid_search import HybridIndex


def _print_response(resp, debug: bool) -> None:
    print(f"\n小茗 ▶ {resp.reply}")
    if resp.next_action:
        print(f"        ↳ next_action: {resp.next_action}")
    if resp.cite:
        print(f"        ↳ cite: {resp.cite}")
    if resp.need_human:
        print("        ↳ ⚠️ 已转人工")
    if debug:
        print(f"        ↳ intent={resp.intent} conf={resp.confidence:.2f}")
        if resp.tool_calls:
            print(f"        ↳ tool_calls: {json.dumps(resp.tool_calls, ensure_ascii=False)}")
        if resp.raw_violations:
            print(f"        ↳ ⛔ 违规命中: {resp.raw_violations}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", type=str, help="单条问答")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    index = HybridIndex.load()
    if index.empty:
        print(
            "⚠️  知识库为空 (data/knowledge/nodes.jsonl 不存在或没数据)。\n"
            "   请先运行：python -m ingestion.pipeline\n"
            "   然后编辑 ingestion/sources.yaml 填入真实的搜索 query 与 seed_urls。\n"
            "   现在仍可启动客服，但所有提问都会直接转人工。\n",
            file=sys.stderr,
        )

    agent = Agent(index=index)

    if args.once:
        resp = agent.chat(args.once)
        _print_response(resp, args.debug)
        return

    print("欢迎使用熹茗 AI 客服「小茗」（输入 quit 退出）")
    while True:
        try:
            line = input("\n你 ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"quit", "exit", ":q"}:
            break
        try:
            resp = agent.chat(line)
        except Exception as e:
            print(f"出错了：{e}", file=sys.stderr)
            continue
        _print_response(resp, args.debug)


if __name__ == "__main__":
    main()
