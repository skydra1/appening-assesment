"""Running 5-6 benchmark queries against the API.

Usage:
    # Terminal 1: python -m uvicorn app:app --reload
    # Terminal 2: python tests_sample_queries.py
    # Or directly (no server): python tests_sample_queries.py --direct
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  
]

API_URL = "http://127.0.0.1:8000/chat"


def run_via_api(queries):
    import httpx

    results = []
    for q in queries:
        try:
            resp = httpx.post(API_URL, json={"query": q}, timeout=90.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            data = {"final_answer": f"REQUEST FAILED: {exc}", "retrieved_context": [], "confidence_score": 0.0}
        results.append((q, data))
    return results


def run_direct(queries):
    from src.graph import answer_question

    results = []
    for q in queries:
        try:
            r = answer_question(q)
            data = {
                "final_answer": r.get("answer", ""),
                "retrieved_context": r.get("context", []),
                "confidence_score": float(r.get("score", 0.0)),
            }
        except Exception as exc:
            data = {"final_answer": f"PIPELINE FAILED: {exc}", "retrieved_context": [], "confidence_score": 0.0}
        results.append((q, data))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="Call LangGraph directly instead of HTTP API")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("WARNING: OPENAI_API_KEY / PINECONE_API_KEY not set. Calls will fail until .env is configured.\n")

    results = run_direct(SAMPLE_QUERIES) if args.direct else run_via_api(SAMPLE_QUERIES)
    for i, (q, data) in enumerate(results, start=1):
        print(f"\n=== Q{i}: {q} ===")
        print(f"Answer: {data.get('final_answer', '')[:1200]}")
        ctx = data.get("retrieved_context", [])
        print(f"Chunks: {len(ctx)} | Score: {data.get('confidence_score')}")
        if ctx:
            print(f"Chunk[0] preview: {ctx[0][:300]}...")
    # Save JSON 
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump([{"query": q, **d} for q, d in results], f, indent=2, ensure_ascii=False)
    print("\nFull results saved to test_results.json")


if __name__ == "__main__":
    main()
