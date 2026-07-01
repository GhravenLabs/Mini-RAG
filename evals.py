#!/usr/bin/env python3
"""
evals.py -- measure the finance RAG (retrieval quality + guardrail accuracy)
============================================================================
"Eval literacy" is the single cheapest hiring signal for an LLM builder (2026 research). This harness
scores the demo the way you'd score any real RAG system:

  * Retrieval quality on in-scope questions:  Hit@k (did the right passage make the top-k?) and MRR.
  * Guardrail accuracy on out-of-scope questions:  did it correctly REFUSE (not hallucinate)?

Run:  python evals.py        (pure stdlib, no API key)
"""
from __future__ import annotations
import json, sys
import finance_rag

K = 3
MIN_SCORE = finance_rag.DEFAULT_MIN_SCORE


def _hit_rank(passages, keywords):
    """1-based rank of the first retrieved passage containing ANY expected keyword, else 0."""
    for i, (_score, text) in enumerate(passages, start=1):
        low = text.lower()
        if any(kw.lower() in low for kw in keywords):
            return i
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    doc = open("finance-handbook.md", encoding="utf-8").read()
    data = json.load(open("eval_set.json", encoding="utf-8"))
    idx = finance_rag.build_index(doc)

    # --- Retrieval quality (in-scope) ---
    hits, rr, rows = 0, 0.0, []
    for item in data["in_scope"]:
        r = finance_rag.answer(item["q"], idx, k=K)
        rank = _hit_rank(r["passages"], item["expect_keywords"])
        ok = rank != 0
        hits += 1 if ok else 0
        rr += (1.0 / rank) if rank else 0.0
        rows.append((ok, rank, item["q"]))
    n_in = len(data["in_scope"])
    hit_at_k = hits / n_in if n_in else 0.0
    mrr = rr / n_in if n_in else 0.0

    # --- Guardrail accuracy (out-of-scope must refuse) ---
    refused, grows = 0, []
    for item in data["out_of_scope"]:
        r = finance_rag.answer(item["q"], idx, k=K)
        correct = not r["grounded"]  # correct = refused
        refused += 1 if correct else 0
        grows.append((correct, r["top_score"], item["q"]))
    n_out = len(data["out_of_scope"])
    guard_acc = refused / n_out if n_out else 0.0

    # --- Scorecard ---
    print("=" * 62)
    print("  FINANCE RAG -- EVALUATION SCORECARD")
    print("=" * 62)
    print(f"  Retrieval Hit@{K}:  {hit_at_k*100:5.1f}%   ({hits}/{n_in} in-scope questions)")
    print(f"  Retrieval MRR:    {mrr:5.2f}")
    print(f"  Guardrail acc.:   {guard_acc*100:5.1f}%   ({refused}/{n_out} out-of-scope refused)")
    print("-" * 62)
    print("  In-scope detail (rank of correct passage; lower = better):")
    for ok, rank, q in rows:
        print(f"    {'PASS' if ok else 'MISS'}  rank={rank or '-'}  {q[:48]}")
    print("  Out-of-scope detail (must refuse):")
    for ok, score, q in grows:
        print(f"    {'REFUSED' if ok else 'LEAKED '}  top={score:.3f}  {q[:44]}")
    print("=" * 62)
    overall = (hit_at_k >= 0.75 and guard_acc >= 0.75)
    print("  RESULT:", "PASS -- retrieval solid + guardrail holds" if overall else "NEEDS TUNING")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
