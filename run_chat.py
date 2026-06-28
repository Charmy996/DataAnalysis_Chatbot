"""
Terminal chat interface.
Fix: variable was named 'prompt' but answer_question returns (answer, results).
"""
from rag_chain import answer_question

while True:
    q = input("\nAsk question (or 'exit'): ").strip()
    if q.lower() in ("exit", "quit", ""):
        break

    answer, results = answer_question(q)

    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")
    for r in results:
        year_str = f" | {r['year']}" if r.get("year") else ""
        print(f"  {r['company']}{year_str} | Page {r['page']} [{r.get('source','text')}] score={r['score']:.2f}")
        print(f"  {r['chunk'][:150]}\n---")
