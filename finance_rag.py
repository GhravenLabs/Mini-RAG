#!/usr/bin/env python3
"""
finance_rag.py -- RAG with a grounding GUARDRAIL (on top of Mini-RAG)
=====================================================================
Adds the two things that separate a toy RAG from a trustworthy one, and that hiring managers
actually look for: a *guardrail* (refuse when the answer isn't grounded in the document) and a
clean seam for *evaluation* (see evals.py). Finance-flavored, because that's the domain edge.

The guardrail: if the best retrieved passage scores below `min_score`, we DO NOT answer -- we say we
couldn't find it. In finance you must never confidently make up a number. This is the human-in-the-loop
/ "know what you don't know" behavior that the 2026 hiring research flags as the #1 cheap signal.
"""
from __future__ import annotations
import math
from pathlib import Path
import mini_rag  # vendored Mini-RAG engine (pure stdlib)

DEFAULT_MIN_SCORE = 0.16  # tuned via evals.py (a "policy handbook" leaks generic "policy" queries at ~0.13); below this = "not in the document"
DEFAULT_MIN_EVIDENCE_TERMS = 2


def build_index(doc: str):
    """Chunk + index a document once; reuse for many questions."""
    chunks = mini_rag.chunk(doc)
    vectors, idf = mini_rag.build_index(chunks)
    return {"chunks": chunks, "vectors": vectors, "idf": idf}


def evidence_terms(question: str, passages: list[tuple[float, str]]) -> list[str]:
    """Return meaningful question terms that appear in retrieved evidence."""
    query_terms = set(mini_rag.tokenize(question))
    evidence_tokens = set(mini_rag.tokenize(" ".join(text for _score, text in passages)))
    return sorted(query_terms & evidence_tokens)


def answer(
    question: str,
    index: dict,
    k: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    min_evidence_terms: int = DEFAULT_MIN_EVIDENCE_TERMS,
) -> dict:
    """Return a grounded result or a guardrailed refusal.

    Returns dict:
    {
      grounded: bool,
      top_score: float,
      evidence_terms: list[str],
      evidence_coverage: float,
      passages: [(score, text)],
      answer: str|None
    }
    """
    if (isinstance(min_score, bool) or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score) or not 0 <= min_score <= 1):
        raise ValueError("min_score must be a finite number between 0 and 1")
    if (isinstance(min_evidence_terms, bool) or not isinstance(min_evidence_terms, int)
            or min_evidence_terms < 0):
        raise ValueError("min_evidence_terms must be a nonnegative integer")
    results = mini_rag.retrieve(question, index["chunks"], index["vectors"], index["idf"], k)
    top = results[0][0] if results else 0.0
    terms = evidence_terms(question, results)
    query_terms = set(mini_rag.tokenize(question))
    coverage = len(terms) / len(query_terms) if query_terms else 0.0
    if not results or top < min_score or len(terms) < min_evidence_terms:
        return {"grounded": False, "top_score": round(top, 3), "evidence_terms": terms,
                "evidence_coverage": round(coverage, 3), "passages": results,
                "answer": "I couldn't find that in the document."}
    # grounded: retrieval is trustworthy. (A grounded LLM answer can be added here if an API key is set;
    # mini_rag.ai_answer(question, [p for _s, p in results]) -- kept optional so the demo needs no key.)
    return {"grounded": True, "top_score": round(top, 3), "evidence_terms": terms,
            "evidence_coverage": round(coverage, 3), "passages": results, "answer": None}


if __name__ == "__main__":
    import sys
    doc = Path(__file__).with_name("finance-handbook.md").read_text(encoding="utf-8")
    idx = build_index(doc)
    q = " ".join(sys.argv[1:]) or "What are the standard payment terms?"
    r = answer(q, idx)
    print("Q:", q)
    print("grounded:", r["grounded"], "| top_score:", r["top_score"])
    print("evidence_terms:", ", ".join(r["evidence_terms"]) or "-",
          "| coverage:", r["evidence_coverage"])
    if r["grounded"]:
        for s, p in r["passages"]:
            print(f"  [{s:.2f}] {p.splitlines()[0][:70]}")
    else:
        print("  ->", r["answer"], "(guardrail refused -- not in the document)")
