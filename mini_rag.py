#!/usr/bin/env python3
"""Mini-RAG — a tiny "chat with your document" retriever in pure Python.

Demonstrates the core Retrieval-Augmented Generation pattern with **zero
dependencies**: chunk a document, rank chunks against a question with TF-IDF +
cosine similarity, and return the most relevant passages. With an API key it
will also write a grounded answer that cites the passages it used.

Examples:
    python mini_rag.py handbook.md "what is the return policy?"
    python mini_rag.py handbook.md "how do I contact support?" --k 2
    python mini_rag.py handbook.md "do you ship overseas?" --ai   # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

STOP = set("a an the and or of to in on for is are was were be been being it its this that "
           "with as at by from we you your our i if how do does can will".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def chunk(doc: str) -> list[str]:
    """Split into passages on blank lines; merge very short ones forward."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
    merged, buf = [], ""
    for p in parts:
        buf = (buf + "\n" + p).strip() if buf else p
        if len(buf) >= 120:           # keep chunks a reasonable size
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def build_index(chunks: list[str]):
    """Return (tfidf_vectors, idf) for the chunks."""
    docs_tokens = [tokenize(c) for c in chunks]
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    n = len(chunks)
    idf = {t: math.log((n + 1) / (df_t + 1)) + 1 for t, df_t in df.items()}
    vectors = []
    for toks in docs_tokens:
        tf = Counter(toks)
        total = sum(tf.values()) or 1
        vectors.append({t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()})
    return vectors, idf


def vectorize(text: str, idf: dict) -> dict:
    toks = tokenize(text)
    tf = Counter(toks)
    total = sum(tf.values()) or 1
    return {t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()}


def cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query: str, chunks: list[str], vectors, idf, k: int = 3):
    qv = vectorize(query, idf)
    scored = sorted(((cosine(qv, v), i) for i, v in enumerate(vectors)),
                    key=lambda x: x[0], reverse=True)
    return [(score, chunks[i]) for score, i in scored[:k] if score > 0]


def ai_answer(query: str, passages: list[str]):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = (f"Answer the question using ONLY the context passages below. Cite the passage "
              f"number(s) you used like [1]. If the answer isn't in the context, say so.\n\n"
              f"Context:\n{context}\n\nQuestion: {query}")
    body = json.dumps({"model": "claude-opus-4-8", "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["content"][0]["text"]
    except Exception as e:  # noqa
        return f"(AI answer failed: {e})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tiny RAG retriever over a local document.")
    ap.add_argument("document", help="path to a .txt/.md document")
    ap.add_argument("question", nargs="?", help="your question (omit for interactive mode)")
    ap.add_argument("--k", type=int, default=3, help="how many passages to retrieve (default 3)")
    ap.add_argument("--ai", action="store_true", help="also generate a grounded answer (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass

    if not os.path.isfile(args.document):
        print(f"No such file: {args.document}", file=sys.stderr)
        return 2
    doc = open(args.document, encoding="utf-8", errors="replace").read()
    chunks = chunk(doc)
    vectors, idf = build_index(chunks)
    print(f"Indexed {len(chunks)} passages from {args.document}.\n")

    def answer(q):
        hits = retrieve(q, chunks, vectors, idf, args.k)
        if not hits:
            print("  No relevant passage found.")
            return
        print("Top passages:")
        for score, passage in hits:
            snippet = " ".join(passage.split())
            print(f"  ({score:.2f}) {snippet[:200]}{'…' if len(snippet) > 200 else ''}")
        if args.ai:
            a = ai_answer(q, [p for _, p in hits])
            if a is None:
                print("\n  (set ANTHROPIC_API_KEY and use --ai for a written answer)")
            else:
                print("\nAnswer:\n  " + a.replace("\n", "\n  "))

    if args.question:
        answer(args.question)
    else:
        print("Interactive mode — ask a question (blank line to quit).")
        while True:
            try:
                q = input("\n? ").strip()
            except EOFError:
                break
            if not q:
                break
            answer(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
