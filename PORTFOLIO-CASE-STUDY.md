# Portfolio Case Study: Mini-RAG

## Problem
RAG demos often hide the important mechanics behind frameworks, vector databases, and embedding APIs. For learning and client explanation, it helps to show the retrieval loop in plain Python.

## Build
Mini-RAG chunks a document, builds TF-IDF vectors, ranks passages with cosine similarity, and returns the best source passages for a question. With an API key, it can also write a grounded answer using only retrieved context.

## Why it is useful
- Demonstrates retrieval, grounding, and citation without dependencies.
- Works offline for the retrieval portion.
- Gives Raven a teachable base for "chat with your SOPs/docs" service ideas.
- Shows Python fundamentals: parsing text, indexing, ranking, CLI UX, and optional API integration.

## Verification
- Sample terminal proof: `assets/screenshot.png`
- Demo document: `handbook.md`
- Smoke check: `.github/workflows/smoke.yml`

## Next upgrades
- Add multi-file folder ingestion.
- Add JSON output and a tiny browser UI.
- Add an embeddings-backed production variant as a separate advanced demo.
