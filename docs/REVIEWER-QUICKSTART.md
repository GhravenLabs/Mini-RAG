# Reviewer Quickstart

Mini-RAG is a tiny pure-Python retrieval demo. It is meant to make the retrieval and grounding loop easy to inspect.

## Suggested review path

1. Read `README.md` for the core flow.
2. Inspect `mini_rag.py` for chunking, TF-IDF ranking, and CLI behavior.
3. Inspect `finance_rag.py` for the evidence-term guardrail.
4. Run the tests before changing scoring or grounding behavior.

## Local verification

```bash
python -m venv .venv
# Windows PowerShell; on macOS/Linux use .venv/bin/python instead.
.venv/Scripts/python -m pip install "pytest>=8,<10"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python mini_rag.py handbook.md "what is the return policy?" --k 2
.venv/Scripts/python evals.py
```

The retrieval path does not require an API key. AI answer generation is optional and should stay graceful when no key is configured.

Pytest is a development dependency, not a retrieval runtime dependency. The evaluation fixture has
eight in-scope questions and four out-of-scope questions; passing it does not guarantee correct
answers on arbitrary documents. Review source passages yourself before relying on generated text.

