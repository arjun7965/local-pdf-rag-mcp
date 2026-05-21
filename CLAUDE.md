# CLAUDE.md

Project context for Claude Code. Read this first when resuming work.

## What this project is

`local-pdf-rag-mcp` — a fully-local MCP server that lets Claude (or any MCP client)
answer questions over arbitrary PDFs. The user points it at a PDF (or folder of
PDFs), then asks questions in plain language; the server retrieves only the
relevant chunks (with page citations) so the model answers without ingesting
the whole document. Intended to be published publicly on GitHub under MIT for
others to use.

The original motivating use case was a ~700-page PCIe specification, but the
tool was deliberately built to be document-agnostic.

## Locked-in design decisions

These were settled with the user — do not relitigate without asking:

1. **Fully local by default.** Embeddings run on-device via
   `sentence-transformers` (default model `all-MiniLM-L6-v2`); vectors persist
   to disk via ChromaDB. No API keys, no data leaves the machine. Embedding
   model is swappable via `PDF_RAG_EMBED_MODEL`.
2. **`search` returns chunks, not answers.** The server returns raw passages
   and lets the model synthesize the answer. This keeps it simple and
   model-agnostic. Do NOT add answer-generation inside the server.
3. **Python.** Matches the user's WSL2 + Claude Code workflow.
4. **ChromaDB, not a hosted vector DB.** Single-user, single-machine,
   privacy-preserving, zero-setup. A hosted DB only matters at multi-user /
   very-large scale, which is out of scope for v1.

## Architecture

- `local_pdf_rag_mcp/chunking.py` — page-aware PDF text extraction (`pypdf`) and
  token-budgeted, paragraph-respecting chunking. Chunks never span pages, so
  each carries an unambiguous page number for citations. Oversized paragraphs
  are split by sentence, then by a hard character budget as a last resort.
  Overlap is carried as a bounded trailing text slice (NOT whole units — see
  "Known gotchas").
- `local_pdf_rag_mcp/store.py` — `VectorStore`, a thin wrapper over a persistent
  ChromaDB client with a local embedding function. Handles add/search/list and
  batches inserts (256) to bound memory on large docs.
- `local_pdf_rag_mcp/server.py` — the FastMCP server. Three tools:
  - `ingest_pdf(path, collection="default")` — file or folder of PDFs.
  - `list_collections()` — names + chunk counts.
  - `search(query, collection="default", top_k=5)` — top chunks with
    `source, p.N` citations.
- `pyproject.toml` — pinned deps (`mcp`, `chromadb`, `sentence-transformers`,
  `pypdf`); console entry point `local-pdf-rag-mcp = local_pdf_rag_mcp.server:main`.
- `README.md`, `LICENSE` (MIT), `.gitignore`.

## Current state

- All four modules compile cleanly and the chunker is tested (see below).
- NOT yet run end-to-end with the real ChromaDB / sentence-transformers / mcp
  stack — those pull large models and weren't installed in the build env.
- Git repo initialized with an initial commit on `main`. Remote
  https://github.com/arjun7965/local-pdf-rag-mcp exists but nothing has been
  pushed yet.

## What's been verified

- Chunking holds its token budget across targets of 80–350 tokens.
- Edge cases pass: all-blank pages → 0 chunks; tiny single-line doc → 1 chunk;
  page numbers always in range; no empty chunks.
- The no-extractable-text path raises `PdfExtractionError` cleanly (so
  scanned/image-only PDFs fail loudly instead of indexing nothing).

## Known gotchas (hard-won — don't regress these)

- **Token budget was silently blown in early versions.** Two causes, both
  fixed: (1) paragraphs larger than the target weren't being split — fixed with
  `_split_oversized` (sentence-pack, then hard char split); (2) overlap carried
  *whole units* forward, so a single target-sized unit became a full extra
  chunk's worth of overlap — fixed by carrying a bounded trailing text slice
  (`eff_overlap = min(overlap_tokens, target_tokens // 3)`). If you touch
  chunking, re-run the budget test across multiple target sizes.
- Token counting is a `~4 chars/token` approximation to avoid a tokenizer
  dependency. Allow ~1.4x overshoot when asserting budgets.

## Next steps / open questions

Immediate (before publishing):
1. **Smoke test on a real machine:** `pip install -e .`, ingest any PDF,
   confirm `search` returns sensible chunks. This is the main untested path.
2. **Add the remote and push to GitHub.** The repo at
   https://github.com/arjun7965/local-pdf-rag-mcp is empty; run
   `git remote add origin https://github.com/arjun7965/local-pdf-rag-mcp.git`
   then `git push -u origin main`. User performs the push — Claude does not
   push on the user's behalf without explicit per-session confirmation.

Possible v2 (user was asked, hasn't decided):
- **Reranking step** after vector search to improve precision before chunks go
  to the model.
- **Table-aware extraction.** Current chunking flattens tables and register
  layouts into plain text — fine for prose, lossy for dense technical specs
  like PCIe. Real table-structure extraction needs a heavier parser. Document
  as a known limitation; natural v2 if the repo gets traction.
- Optional OCR path for scanned PDFs (currently explicitly out of scope).

## Testing notes

Quick chunker regression check (no heavy deps needed beyond `pypdf` +
`reportlab` to synthesize a test PDF):

```python
from pathlib import Path
from local_pdf_rag_mcp.chunking import extract_pages, chunk_pages
pages = extract_pages(Path("some.pdf"))
for target in (80, 120, 200, 350):
    chunks = chunk_pages(pages, source="some.pdf",
                         target_tokens=target, overlap_tokens=target // 4)
    sizes = [len(c.text) // 4 for c in chunks]
    assert max(sizes) <= target * 1.4, f"over budget at {target}: {max(sizes)}"
    assert all(c.text.strip() for c in chunks)
```
