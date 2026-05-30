# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-29

Initial release. A fully-local [MCP](https://modelcontextprotocol.io) server
for question-answering over your PDFs. Point it at a PDF (or a folder), ask
questions in plain language, and Claude retrieves only the relevant passages —
with page citations — instead of ingesting the whole document. No API keys,
nothing leaves your machine.

### Added

- **Fully local by default.** Embeddings run on-device via
  `sentence-transformers` (`all-MiniLM-L6-v2`); vectors persist to disk with
  ChromaDB. No cloud, no keys.
- **Page-cited retrieval.** Chunks never span a page boundary, so every result
  carries an unambiguous `source, p.N` citation.
- **Cross-encoder reranking.** Search over-fetches ~20 candidates and reranks
  them with a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision
  before returning the top results. Disable with `PDF_RAG_RERANK=0`.
- **Four MCP tools:** `ingest_pdf`, `list_collections`, `search`, and
  `delete_collection`.
- **Returns chunks, not answers.** The server hands back raw passages and lets
  the model synthesize — keeping it simple and model-agnostic.
- **Swappable embedding model** via `PDF_RAG_EMBED_MODEL`.
- **Opt-in table-aware extraction** (`PDF_RAG_TABLES=1`, default off): detects
  ruled tables and serializes them one record per row so a query matches a
  single row's cells directly.
- Chunker test suite and GitHub Actions CI across Python 3.10–3.12.

### Requirements

- Python 3.10+
- ~170 MB disk for the two default models (embedder + reranker), downloaded
  and cached on first use.

### Known limitations

- **No OCR.** Scanned / image-only PDFs have no extractable text and fail
  loudly with a clear error rather than indexing nothing.
- **Ruled tables only.** With `PDF_RAG_TABLES=1`, only tables with visible grid
  lines are detected; borderless (whitespace-aligned) tables fall back to
  prose. Table extraction is experimental and remains opt-in pending
  validation against real ruled-table documents.
- **Encrypted PDFs** open only if they use an empty password.
- Tuned for a single-machine, single-user workflow.

[0.1.0]: https://github.com/arjun7965/local-pdf-rag-mcp/releases/tag/v0.1.0
