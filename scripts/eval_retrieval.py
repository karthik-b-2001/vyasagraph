"""Retrieval benchmark: do the right Ganguli sections come back for a question?

Each question has gold section ranges in the Ganguli translation, located by
searching the text (e.g. Karna's wheel sinking and his death: Karna Parva
90-91). Ranking is within the translation only, which is what fills the
translation quota (PASSAGES_TRANSLATION) given to the LLM. A hit is a
retrieved chunk inside a gold range.

Embeds the corpus in memory, so any model can be compared without touching
the vector store:
    python scripts/eval_retrieval.py                       # configured model
    python scripts/eval_retrieval.py BAAI/bge-small-en-v1.5 "Represent this sentence for searching relevant passages: "
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.ingest.text import build_chunks, load_sources  # noqa: E402

GOLD = [
    ("How did Karna die?", [("Karna Parva", 90, 91)]),
    ("Who killed Duryodhana and how?", [("Shalya Parva", 57, 59), ("Sauptika Parva", 9, 9)]),
    ("What happened at the dice game?", [("Sabha Parva", 45, 80)]),
    ("How was Abhimanyu killed?", [("Drona Parva", 33, 52)]),
    ("What did Krishna teach Arjuna on the battlefield?", [("Bhishma Parva", 25, 42)]),
    ("How did the Pandavas escape the house of lac?", [("Adi Parva", 143, 153)]),
    ("Who won Draupadi at her swayamvara and how?", [("Adi Parva", 186, 194)]),
    ("What questions did the Yaksha ask Yudhishthira at the lake?", [("Vana Parva", 310, 313)]),
    ("Why did Karna give his armour and earrings to Indra?", [("Vana Parva", 298, 310)]),
    ("How did the Yadavas destroy themselves?", [("Mausala Parva", 1, 8)]),
    ("What happened to the dog that followed Yudhishthira?", [("Mahaprasthanika Parva", 1, 3), ("Svargarohana Parva", 1, 6)]),
    ("How did Bhishma fall in battle?", [("Bhishma Parva", 119, 121)]),
    ("Why was the Khandava forest burned?", [("Adi Parva", 224, 236)]),
    ("Tell me the story of Savitri and Satyavan", [("Vana Parva", 292, 298)]),
    ("Tell me the story of Nala and Damayanti", [("Vana Parva", 52, 79)]),
    ("How did Bhima kill Kichaka?", [("Virata Parva", 21, 23)]),
    ("Why did Krishna kill Shishupala?", [("Sabha Parva", 39, 44)]),
    ("How was Jarasandha killed?", [("Sabha Parva", 20, 24)]),
    ("How did Arjuna kill Jayadratha?", [("Drona Parva", 145, 146)]),
    ("How did Ghatotkacha die?", [("Drona Parva", 178, 180)]),
    ("How was Drona killed?", [("Drona Parva", 191, 193)]),
    ("What did Ashwatthama do in the night attack on the camp?", [("Sauptika Parva", 5, 9)]),
    ("Why did Gandhari curse Krishna?", [("Stri Parva", 25, 25)]),
    ("How did Bhima kill Hidimba?", [("Adi Parva", 154, 157)]),
    ("How did Arjuna obtain the Pashupata weapon from Shiva?", [("Vana Parva", 38, 41)]),
]


def is_hit(meta, gold):
    return any(meta["parva"] == p and a <= meta["chapter"] <= b for p, a, b in gold)


def main():
    from sentence_transformers import SentenceTransformer

    model_name = sys.argv[1] if len(sys.argv) > 1 else config.EMBEDDING_MODEL
    prefix = sys.argv[2] if len(sys.argv) > 2 else config.EMBEDDING_QUERY_PREFIX
    model = SentenceTransformer(model_name)

    texts, metas = [], []
    for source in load_sources():
        if source.kind != "translation":
            continue
        for text, meta in build_chunks(source).values():
            texts.append(text)
            metas.append(meta)
    print(f"model {model_name}, {len(texts)} chunks, query prefix {prefix!r}")
    docs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    queries = model.encode([prefix + q for q, _ in GOLD], normalize_embeddings=True)

    hit5 = hit10 = mrr = 0.0
    for (question, gold), qv in zip(GOLD, queries):
        top = np.argsort(-(docs @ qv))[:10]
        ranks = [i for i, idx in enumerate(top) if is_hit(metas[idx], gold)]
        first = ranks[0] + 1 if ranks else None
        hit5 += bool(first and first <= 5)
        hit10 += bool(first)
        mrr += 1 / first if first else 0
        print(f"  {'rank ' + str(first) if first else 'miss  '}  {question}")
    n = len(GOLD)
    print(f"hit@5 {hit5 / n:.0%}   hit@10 {hit10 / n:.0%}   MRR@10 {mrr / n:.2f}")


if __name__ == "__main__":
    main()
