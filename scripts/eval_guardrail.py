"""Measure the guardrail's retrieval gate on labelled questions.

For each question, prints whether a known character was detected and the
best passage similarity, then reports how many questions each candidate
GUARD_MIN_SCORE would get right. Needs the vector index and Neo4j, not the LLM.

Usage:
    python scripts/eval_guardrail.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, rag  # noqa: E402

# Questions a user might ask about the epic, phrased without and with names.
ON_TOPIC = [
    "Who were Arjuna's parents?",
    "How did Karna die?",
    "Who killed Duryodhana?",
    "What happened at the dice game?",
    "What was Abhimanyu's fate?",
    "Who is Vrikodara?",
    "Why did the great war happen?",
    "What did Krishna teach on the battlefield?",
    "What happened to the Pandavas after the war?",
    "How did the five brothers end up sharing one wife?",
    "What vow did the grandsire take?",
    "Who was the teacher of the princes?",
    "What is dharma according to the epic?",
    "Describe the battle formation called the Chakravyuha",
    "What happened during the exile in the forest?",
    "Why did the queen curse the god after her sons died?",
    "Who won the archery contest at the swayamvara?",
    "What happened in the year of hiding at the king's court?",
    "How did the Yadavas destroy themselves?",
    "What happened on the final journey to the Himalayas?",
    # Added after the threshold was first chosen, to check it on unseen questions.
    "Why was Bhishma lying on a bed of arrows?",
    "Who was Ekalavya and what did he give his teacher?",
    "Why did Gandhari blindfold herself?",
    "What did Yudhishthira answer the Yaksha at the lake?",
    "How was the wax house plot foiled?",
    "Who was Shikhandi in his previous life?",
    "What boon did Kunti receive from Durvasa?",
    "Why did Ashwatthama attack the sleeping camp?",
    "Tell me about the burning of the Khandava forest",
    "Who drove Arjuna's chariot?",
    "What happened to Jayadratha?",
    "What is the story of Savitri and Satyavan?",
    "Why did Karna give away his armour?",
    "What did the dog do at the end?",
    "What is the Vishvarupa?",
    "How did the war end?",
    "who was ganga",
    "explain the dice match",
    "What advice did Vidura give?",
    "What weapons did Arjuna get from the gods?",
]

OFF_TOPIC = [
    "What is the capital of France?",
    "Write me a Python function to reverse a list",
    "Who won the 2022 FIFA World Cup?",
    "How do I bake sourdough bread?",
    "Explain quantum entanglement",
    "What is the stock price of Apple?",
    "Who is Elon Musk?",
    "Summarize the plot of Harry Potter",
    "What is the best laptop for programming?",
    "How do I fix a flat bicycle tyre?",
    "Translate 'good morning' into Spanish",
    "What are the side effects of ibuprofen?",
    "Recommend a good movie for tonight",
    "How does a car engine work?",
    "Who wrote Pride and Prejudice?",
    "What is the weather like tomorrow?",
    "Give me tips for a job interview",
    "What is machine learning?",
    "How many calories are in an apple?",
    "Who was the first president of the United States?",
    # Added later, including topics adjacent to the epic (yoga, karma, Shiva).
    "What is the population of India?",
    "How do I learn guitar?",
    "Explain how neural networks are trained",
    "Who is the prime minister of the UK?",
    "Write a poem about the ocean",
    "What is the meaning of life?",
    "Tell me a joke",
    "What are the rules of cricket?",
    "How do I make biryani?",
    "What is the Ramayana about?",
    "Who is Taylor Swift?",
    "What year did World War 2 end?",
    "What's the difference between TCP and UDP?",
    "How do vaccines work?",
    "Who was Napoleon?",
    "Help me write a cover letter",
    "What is yoga good for?",
    "How to meditate?",
    "What is karma in Buddhism?",
    "Who is Lord Shiva?",
]


def main():
    rag.load()
    rows = []
    for label, questions in (("on", ON_TOPIC), ("off", OFF_TOPIC)):
        for q in questions:
            ctx = rag.retrieve(q)
            rows.append((label, q, ctx["entities"], ctx["best_score"]))

    print(f"{'label':<5} {'best':>5}  entities / question")
    for label, q, entities, best in sorted(rows, key=lambda r: r[3]):
        print(f"{label:<5} {best:5.3f}  {entities or '-'}  {q}")

    print("\nthreshold  on-topic passed  off-topic refused")
    for t in [round(0.30 + 0.05 * i, 2) for i in range(10)]:
        on = sum(1 for l, _, e, b in rows if l == "on" and (e or b >= t))
        off = sum(1 for l, _, e, b in rows if l == "off" and not (e or b >= t))
        mark = "  <- current" if abs(t - config.GUARD_MIN_SCORE) < 1e-9 else ""
        print(f"  {t:.2f}     {on:2}/{len(ON_TOPIC)}            {off:2}/{len(OFF_TOPIC)}{mark}")
    rag.close()


if __name__ == "__main__":
    main()
