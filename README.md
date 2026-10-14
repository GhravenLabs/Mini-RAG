# Mini-RAG

A tiny **"chat with your document"** retriever in **pure Python** — zero dependencies. It
demonstrates the core **Retrieval-Augmented Generation** pattern: chunk a document, rank passages
against a question with **TF-IDF + cosine similarity**, and return the most relevant ones. With an
API key it also writes a grounded answer that **cites** the passages it used. The finance demo
adds an evidence-term guardrail so a question is not treated as grounded just because one generic
passage scored above the threshold.

![Mini-RAG screenshot](assets/screenshot.png)

## Portfolio proof
- [Live browser demo](https://mini-rag-ju6l.onrender.com/) — try the retrieval flow without cloning the repo.
- [Case study](PORTFOLIO-CASE-STUDY.md) — how this tiny implementation explains the RAG pattern without heavy tooling.
- GitHub Actions smoke check compiles the script and verifies a sample retrieval run on every push.

## Why it's interesting
RAG is usually shown with heavy stacks (vector DBs, embedding APIs, LangChain). This strips it to
the essentials so the *mechanism* is clear and it runs anywhere with just Python — no install, no
keys needed for retrieval.

## Usage
Documents must be UTF-8 text. Invalid encoding is reported before indexing; the CLI
does not silently replace unreadable bytes or modify the original file.

```bash
python mini_rag.py handbook.md "what is the return policy?"
python mini_rag.py handbook.md "do you ship overseas?" --k 2
python mini_rag.py handbook.md "return policy" --json > results.json
python mini_rag.py handbook.md "return policy" --json --out results.json
python mini_rag.py handbook.md                      # interactive Q&A loop
python mini_rag.py handbook.md "how long is the warranty?" --ai   # grounded answer (needs ANTHROPIC_API_KEY)
```

## Example
```
$ python mini_rag.py handbook.md "when can I contact support?" --k 1
Indexed 6 passages from handbook.md.

Top passages:
  [1] (0.46) ## Support hours and contact  Customer support is available Monday to Friday, 9am to 6pm...
```

Passage labels such as `[1]` match the context numbers sent to the optional AI
answer. They restart for each question and are not permanent document identifiers.
The CLI shows shortened previews; generated citations still need checking against
the source document.

`--json` requires a question and emits a single JSON object containing the document,
question, and full passages with citation numbers and similarity scores. No matches
produces an empty `passages` list. With `--ai`, an `answer` field is included (`null`
when no answer is available). Errors go to stderr, keeping successful output parseable.

Use `--min-score 0.4` to exclude weak lexical matches (range 0–1). The default
retains every positive match up to `--k`. This is a similarity filter, not an
answer-confidence estimate or a guarantee that the passages answer the question.
`--out` requires `--json` and atomically writes UTF-8 output, preserving an existing
report if writing fails. It refuses to overwrite the source document.

## How it works
1. **Chunk** the document into passages (blank-line split, short chunks merged).
2. **Index** — build TF-IDF vectors (term frequency × inverse document frequency) per passage.
3. **Retrieve** — vectorize the question and rank passages by cosine similarity.
4. **Audit grounding** — the finance demo reports which meaningful question terms are actually
   supported by the retrieved passages before allowing an answer.
5. **(Optional) Answer** — with `--ai`, send the top passages + question to the Anthropic API
   (via `urllib`, no SDK) and request an answer that cites the source passages. Prompting and
   citations do not eliminate hallucinations; verify generated claims against the document.

Pure standard library: `re`, `math`, `collections`, `urllib`. No `pip install` for retrieval.

## Limitations and verification

- Retrieval uses lexical overlap, so it can miss paraphrases or return passages that share words
  without answering the question. The finance guardrail is a score/term heuristic, not a factuality guarantee.
- Finance guardrail settings require a finite `min_score` between 0 and 1 and a nonnegative
  integer `min_evidence_terms`; invalid values raise `ValueError`. Zero disables that threshold,
  but a query with no matching passage still receives a refusal.
- `--ai` sends the question and retrieved passages to an external model API. Use public or synthetic
  documents for the hosted demo; keep confidential documents out of it.
- Generated answers stopped by a token or context limit are marked incomplete. Partial text is
  retained for inspection; the tool does not automatically retry or incur another API request.
- The browser demo may need time to wake up on its hosting service. Local retrieval works without
  a hosted service or API key.
- Run `python evals.py` for the bundled eight retrieval and four refusal examples. These small
  fixture results are not evidence of general accuracy. See [the reviewer quickstart](docs/REVIEWER-QUICKSTART.md)
  for isolated test setup.
- The finance demo and evaluation script locate their bundled data beside the scripts,
  so launching them by path from another directory does not load unrelated local files.

## Real-world version
For production you'd swap TF-IDF for embeddings + a vector store (e.g. pgvector/Chroma) — but the
retrieve → ground → cite loop is exactly this. Good base for an SMB "chat with your SOPs" tool.

## License
MIT © Rolly Calma ([Ghraven](https://github.com/Ghraven))

---
_By **Rolly Calma** — see live demos & services at **[rollycalma.com](https://rollycalma.com/)**._
