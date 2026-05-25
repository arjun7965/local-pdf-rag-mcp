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

- `local_pdf_rag_mcp/chunking.py` — page-aware PDF text extraction (`pdfplumber`)
  and token-budgeted, paragraph-respecting chunking. Chunks never span pages,
  so each carries an unambiguous page number for citations. Oversized
  paragraphs are split by sentence, then by a hard character budget as a last
  resort. Overlap is carried as a bounded trailing text slice (NOT whole
  units — see "Known gotchas"). Chose `pdfplumber` (MIT, on pdfminer.six)
  over `pymupdf` (AGPL-3.0) to keep the project's MIT license clean for
  downstream users. Internals are unified on a `Segment` abstraction (prose
  or table); `chunk_pages` wraps each page as a single prose segment, so the
  default path is unchanged. Opt-in table-aware extraction (`PDF_RAG_TABLES=1`)
  detects ruled tables, serializes them one record per row, interleaves them
  with prose by vertical position, and tags those chunks `chunk_type="table"`.
  Table chunks pack whole rows (never split mid-row) and carry no overlap.
- `local_pdf_rag_mcp/store.py` — `VectorStore`, a thin wrapper over a persistent
  ChromaDB client with a local embedding function. Handles add/search/list and
  batches inserts (256) to bound memory on large docs. `search` over-fetches
  candidates (default 20) and runs a cross-encoder reranker
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over them to improve precision
  before returning top_k. The reranker is lazy-loaded on first search and
  can be disabled via `PDF_RAG_RERANK=0`.
- `local_pdf_rag_mcp/server.py` — the FastMCP server. Four tools:
  - `ingest_pdf(path, collection="default")` — file or folder of PDFs.
  - `list_collections()` — names + chunk counts.
  - `search(query, collection="default", top_k=8)` — top chunks with
    `source, p.N` citations (reranked from ~20 candidates).
  - `delete_collection(collection)` — drop one collection + its chunks
    (irreversible); for rebuilding a single collection without wiping others.
- `pyproject.toml` — pinned deps (`mcp`, `chromadb`, `sentence-transformers`,
  `pdfplumber`); console entry point `local-pdf-rag-mcp = local_pdf_rag_mcp.server:main`.
- `README.md`, `LICENSE` (MIT), `.gitignore`.

## Current state

- All modules compile cleanly; chunker has a pytest suite (see "Testing notes").
- Verified end-to-end: ingest → embed → rerank → search confirmed on a real
  PDF (bitcoin_as_macro.pdf, 21 pages → 71 chunks) with sensible, well-ranked
  results.
- Published at https://github.com/arjun7965/local-pdf-rag-mcp; `main` tracks
  `origin/main`. Registered with Claude Code as an MCP server via
  `uvx --from git+<repo>.git local-pdf-rag-mcp`.

## What's been verified

- Chunking holds its token budget across targets of 80–350 tokens.
- Edge cases pass: all-blank pages → 0 chunks; tiny single-line doc → 1 chunk;
  page numbers always in range; no empty chunks.
- The no-extractable-text path raises `PdfExtractionError` cleanly (so
  scanned/image-only PDFs fail loudly instead of indexing nothing).
- Table-aware extraction (opt-in): a synthesized ruled table is serialized
  one record per row (cells stay together, keyed by header), tagged
  `chunk_type="table"`, and stays within the token budget. With the flag on
  but no table detected, output is identical to the prose path — confirmed
  on bitcoin_as_macro.pdf (21 pages → 71 prose chunks, 0 table segments,
  same as flag-off).

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
- **Chunk size is coupled to the embedding model's `max_seq_length`.**
  sentence-transformers silently truncates inputs above that limit, so a
  chunk longer than the model can ingest gets its tail dropped from the
  embedding (the text is still stored and returned — only the vector is
  short, so queries that should match the tail wouldn't retrieve the chunk).
  MiniLM caps at 256 tokens, so the chunker's `target_tokens` default is
  set to 250 to sit just under that. If you swap the embedding model,
  retune `target_tokens` to the new model's window.

## Re-ingesting and refreshing the deployed server

When you change extraction, chunking, or the embedding model, existing
ChromaDB collections keep their old representation — vectors don't
auto-update. A collection embeds and stores chunks at ingest time, so a
new parser, a different `target_tokens`, or a swapped embedding model only
affects *future* ingests. To benefit from such a change you must rebuild
the affected collections.

- **Rebuilding one collection.** Use the `delete_collection` tool to drop
  just that collection, then re-ingest its PDF. Leaves other collections
  intact.
- **Wiping everything.** To clear all collections at once, remove the whole
  on-disk store: `Remove-Item -Recurse $env:USERPROFILE\.local_pdf_rag_mcp`
  (PowerShell) or `rm -rf ~/.local_pdf_rag_mcp` (bash). Then re-ingest.
- **Switching embedding models** changes the vector space, so it also
  invalidates existing vectors — same wipe-and-re-ingest applies.
- **Shared store.** The uvx-launched MCP server and any direct
  `VectorStore()` call (e.g. the venv used for smoke tests) read/write the
  same `~/.local_pdf_rag_mcp/chroma`, so a collection ingested from one is
  visible to the other.

Deploying code changes to the Claude Code MCP server (registered as
`uvx --from git+<repo>.git@main local-pdf-rag-mcp`):

- After pushing to `main`, uvx may reuse a stale cached env. Force a
  one-time rebuild, then restart Claude Code:
  `echo "" | uvx --refresh --from git+<repo>.git local-pdf-rag-mcp`
  (the piped empty string EOFs the stdio server so it builds the env and
  exits instead of hanging). Prefer this over `uv cache clean`, which
  nukes the entire uv cache.
- Or pin a commit SHA in the registration (`...@<sha>`) for reproducibility
  and to stop the server picking up untested commits; bump the SHA when you
  want it to move forward.

## Next steps / open questions

In progress:
- **Table-aware extraction — implemented, behind `PDF_RAG_TABLES=1` (default
  off).** Decisions locked with the user: row-as-record serialization, keep
  the "never cross a page" invariant (no header-carry across page breaks),
  opt-in flag during the trial. Remaining before flipping the default on:
  validate on a table-heavy real PDF (the bitcoin doc has no tables, so it
  only exercised the fallback). Out of scope: merged/nested cells, spanning
  headers, scanned-image tables. Once proven, flip the default and delete
  the flag.

Possible v2 (user was asked, hasn't decided):
- Optional OCR path for scanned PDFs (currently explicitly out of scope).

## Testing notes

Chunker tests live in `tests/test_chunking.py`. They cover the token-budget
regression across multiple target sizes plus the edge cases listed under
"What's been verified" (blank pages, tiny doc, page-number range, no-text
PDFs). Synthesize PDFs via `reportlab` so there are no binary fixtures in
the repo.

To run:

```bash
pip install -e ".[dev]"
pytest
```

If you touch chunking, this is the suite to re-run.
