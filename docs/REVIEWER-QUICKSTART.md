# Reviewer Quickstart

Mini-RAG is a tiny pure-Python retrieval demo. It is meant to make the retrieval and grounding loop easy to inspect.

## Suggested review path

1. Read `README.md` for the core flow.
2. Inspect `mini_rag.py` for chunking, TF-IDF ranking, and CLI behavior.
3. Inspect `finance_rag.py` for the evidence-term guardrail.
4. Run the tests before changing scoring or grounding behavior.

## Local verification

```bash
python -m pytest
python mini_rag.py handbook.md "what is the return policy?" --k 2
```

The retrieval path does not require an API key. AI answer generation is optional and should stay graceful when no key is configured.

