"""Ask the RAG pipeline a question from the terminal (no API or frontend needed).

Usage:
    python scripts/ask.py "How did Karna die?"
    python scripts/ask.py --context "How did Karna die?"   # also print what was retrieved
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import rag  # noqa: E402


async def run(question, show_context):
    rag.load()
    ctx = rag.retrieve(question)
    if show_context:
        print(f"entities: {ctx['entities']}  best score: {ctx['best_score']:.3f}  on topic: {ctx['on_topic']}")
        for f in ctx["facts"]:
            print(f"  fact: {f['source']} {f['type']} {f['target']} ({f['origins']})")
        for p in ctx["passages"]:
            tag = "COMMENTARY " if p["section"] == "commentary" else ""
            print(f"  passage {p['score']:.3f} {tag}{p['citation']} [{p['source']}]")
        print()
    if not ctx["on_topic"]:
        print(rag.REFUSAL)
    else:
        async for token in rag.stream_answer(rag.build_messages(question, ctx)):
            print(token, end="", flush=True)
        print()
    rag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--context", action="store_true", help="print retrieved facts and passages")
    args = parser.parse_args()
    asyncio.run(run(args.question, args.context))
