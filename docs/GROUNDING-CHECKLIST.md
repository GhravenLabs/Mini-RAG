# Grounding Checklist

Use this checklist when changing retrieval, ranking, or answer-generation behavior.

## Retrieval quality

- The returned passages contain terms that answer the question.
- Scores are deterministic for the same document and query.
- Short, generic overlaps do not count as strong grounding by themselves.
- Empty or unrelated questions fail gracefully.

## Answer behavior

- Optional AI answers cite retrieved passages.
- The prompt constrains answers to the supplied context.
- Missing API keys do not break retrieval-only usage.
- Tests cover at least one grounded and one weakly grounded case.

