"""Ingestion CLI.

    python -m src.ingest              # text sources -> vector index, CSVs -> Neo4j
    python -m src.ingest text         # only the vector index (incremental)
    python -m src.ingest graph        # only the graph (full rebuild)
    python -m src.ingest text --rebuild   # ignore the existing index, re-embed everything
    python -m src.ingest stats        # chunk counts and token lengths, writes nothing
"""

import argparse
import logging
import os
import warnings

# Keep the output readable: silence model-loading chatter.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(prog="python -m src.ingest")
    parser.add_argument("step", nargs="?", default="all", choices=["all", "text", "graph", "stats"])
    parser.add_argument("--rebuild", action="store_true", help="re-embed every chunk from scratch")
    args = parser.parse_args()

    if args.step == "stats":
        from src.ingest.text import print_stats
        print_stats()
        return
    if args.step in ("all", "text"):
        from src.ingest.text import ingest_text
        print("Text -> vector index")
        ingest_text(rebuild=args.rebuild)
    if args.step in ("all", "graph"):
        from src.ingest.graph import ingest_graph
        print("Graph -> Neo4j")
        ingest_graph()


if __name__ == "__main__":
    main()
