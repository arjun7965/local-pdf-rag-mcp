# local-pdf-rag-mcp

[![CI](https://github.com/arjun7965/local-pdf-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/arjun7965/local-pdf-rag-mcp/actions/workflows/ci.yml)

A fully-local [MCP](https://modelcontextprotocol.io) server that lets Claude
(or any MCP client) answer questions over your PDFs. Point it at a PDF, ask
questions in plain language, and the model fetches only the relevant passages
— with page-level citations — instead of swallowing the whole document.

- **Fully local by default.** Embeddings run on-device (sentence-transformers)
  and vectors are stored on disk (ChromaDB). No API keys, nothing leaves your
  machine.
- **Token-cheap.** Only a handful of relevant chunks are sent to the model per
  question, not the entire document.
- **Cited answers.** Every retrieved chunk carries its source filename and page
  number.
- **Any PDF, many PDFs.** Index a single file or a whole folder, organized into
  named collections.

## How it works

Ingestion (once per document) extracts the text page by page, splits it into
overlapping ~250-token chunks that respect paragraph boundaries, embeds each
chunk locally, and stores them in ChromaDB. At query time the server embeds
your question, pulls the top ~20 chunks from vector search, then reranks
them with a local cross-encoder so the most relevant ones surface first.
The model reads those chunks and writes the answer — the server deliberately
does **not** generate answers itself, which keeps it simple and model-agnostic.

## Requirements

- Python 3.10+
- ~170 MB disk for the two default models, downloaded automatically and
  cached: the embedding model (~80 MB, fetched on first ingest/search) and
  the cross-encoder reranker (~80 MB, fetched on first search). Reranking
  can be turned off with `PDF_RAG_RERANK=0` if you'd rather skip the
  second download.

## Install

```bash
git clone https://github.com/arjun7965/local-pdf-rag-mcp.git
cd local-pdf-rag-mcp
pip install -e .
```

## Register with Codex

If you installed it with `pip install -e .`, register the console command:

```bash
codex mcp add pdf-rag -- local-pdf-rag-mcp
```

Or run it straight from GitHub without cloning, using
[uv](https://docs.astral.sh/uv/)'s `uvx`:

```bash
codex mcp add pdf-rag -- uvx --from git+https://github.com/arjun7965/local-pdf-rag-mcp.git local-pdf-rag-mcp
```

Verify the registration:

```bash
codex mcp get pdf-rag
```

Codex CLI and the Codex IDE extension share MCP configuration. To configure
the server manually, add this to `~/.codex/config.toml`, or to
`.codex/config.toml` for a trusted project:

```toml
[mcp_servers.pdf-rag]
command = "local-pdf-rag-mcp"
```

Start a new Codex session after registering the server. In the Codex terminal
UI, use `/mcp` to check that it is active. See the
[Codex MCP documentation](https://developers.openai.com/codex/mcp).

## Register with Claude Code

If you installed it (the `pip install -e .` above), point Claude Code at the
console command:

```bash
claude mcp add pdf-rag -- local-pdf-rag-mcp
```

Or run it straight from GitHub without cloning, using
[uv](https://docs.astral.sh/uv/)'s `uvx` — it fetches and caches the package
on first launch:

```bash
claude mcp add pdf-rag -- uvx --from git+https://github.com/arjun7965/local-pdf-rag-mcp.git local-pdf-rag-mcp
```

Or add it manually to your Claude MCP config:

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "local-pdf-rag-mcp"
    }
  }
}
```

Restart Claude Code so it picks up the new server.

## Usage

The server exposes four tools. In practice you just talk to Claude and it
calls them for you:

> **You:** Ingest the spec at ~/docs/pcie-5.0.pdf into a collection called "pcie".
>
> *Claude calls `ingest_pdf` → "Ingested into collection 'pcie': pcie-5.0.pdf: 712 pages, 2{,}480 chunks"*
>
> **You:** How does link equalization work during training?
>
> *Claude calls `search` with your question, reads the returned passages, and answers — citing e.g. `pcie-5.0.pdf, p.412`.*

### Tools

| Tool | What it does |
| --- | --- |
| `ingest_pdf(path, collection="default")` | Chunk + embed a PDF file, or every PDF in a folder, into a collection. |
| `list_collections()` | Show indexed collections and their chunk counts. |
| `search(query, collection="default", top_k=8)` | Return the most relevant chunks with citations. |
| `delete_collection(collection)` | Delete one collection and all its chunks (irreversible). |

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF_RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model name. |
| `PDF_RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to rerank vector hits. |
| `PDF_RAG_RERANK` | `1` | Set to `0` to skip reranking and use raw vector ranking. |
| `PDF_RAG_TABLES` | `0` | Set to `1` to enable table-aware extraction (see below). |
| `PDF_RAG_DB_PATH` | `~/.local_pdf_rag_mcp/chroma` | Where the vector store lives on disk. |

### Table-aware extraction (opt-in)

By default, text is extracted linearly — tables get flattened into prose,
which scatters a row's cells and hurts retrieval on dense technical docs.
Set `PDF_RAG_TABLES=1` to detect ruled tables and serialize them one record
per row (`Field: Foo; Bits: 0-3; Description: ...`), so a query about a
single row matches that row's record directly. Detection is conservative
(it relies on ruling lines, so whitespace-aligned prose isn't misread as a
table), and any page with no detected table falls back to the normal prose
path. Re-ingest after enabling, since the change only affects future
ingests.

**Only ruled tables are detected.** Because detection requires visible grid
lines, *borderless* tables — columns aligned by whitespace with no ruling
lines — are not recognized and fall back to the prose path, where their
cells get flattened into linear text. This is a deliberate tradeoff:
alignment-based detection would catch borderless tables but also misreads
ordinary prose layouts as tables, shredding them into junk cells. If your
documents rely on borderless tables, `PDF_RAG_TABLES` won't help with them.

### Embedding model

The default is `all-MiniLM-L6-v2` and the rest of the project is tuned
around it:

- **Why this model.** Small (~80 MB), fast on CPU, no GPU required, decent
  general-English retrieval quality, Apache-2.0 licensed. Standard default
  in the sentence-transformers ecosystem.
- **Input limit: 256 tokens.** sentence-transformers silently truncates
  inputs above the model's `max_seq_length`. Chunks are sized to stay
  within this window so the embedding reflects the whole chunk, not just
  its head. If you swap in a model with a different limit (e.g.
  `BAAI/bge-large-en-v1.5` at 512), consider raising `target_tokens` in
  `chunk_pages` to match — otherwise you're paying for capacity you don't
  use.
- **Output: a 384-dim vector.** Each chunk embeds to a fixed
  384-dimensional vector regardless of its length (the model produces a
  vector, not text, so there's no output-token limit). ChromaDB infers this
  dimensionality automatically; a different model with a different size just
  works, but mixing vectors of different sizes in one collection does not —
  re-ingest after switching models.
- **Swapping.** Any sentence-transformers model from HuggingFace works:
  ```bash
  PDF_RAG_EMBED_MODEL=BAAI/bge-large-en-v1.5 local-pdf-rag-mcp
  ```
  Larger models (BGE-large, E5-large) improve retrieval quality at the
  cost of more disk, more RAM, and slower embedding. Switching models
  invalidates any existing vectors — delete `~/.local_pdf_rag_mcp/chroma`
  and re-ingest.
- **Offline / quiet operation.** Both models are cached after first use (in
  `~/.cache/huggingface`). To run fully offline afterward — and silence the
  `Warning: You are sending unauthenticated requests to the HF Hub` message —
  set `HF_HUB_OFFLINE=1`. Unset it temporarily if you switch to a model you
  haven't downloaded yet, since offline mode blocks new fetches.

## Limitations

- **No OCR.** Scanned or image-only PDFs have no extractable text; the server
  detects this and returns a clear error rather than indexing nothing.
- **Encrypted PDFs** open only if they use an empty password.
- **Borderless tables.** Even with `PDF_RAG_TABLES=1`, only tables with
  visible grid/border lines are detected. Whitespace-aligned tables are
  flattened into prose like any other text (see Configuration → Table-aware
  extraction).
- **Embedding input cap.** Chunks longer than the embedding model's
  `max_seq_length` (256 tokens for the default MiniLM) are silently
  truncated by sentence-transformers — the full text is still stored and
  returned, but the vector reflects only the head. Keep `target_tokens`
  aligned with whatever model you use.
- Tuned for a single-machine, single-user workflow. For multi-user or
  very-large-scale deployments you'd swap ChromaDB for a hosted vector store.

## License

MIT — see [LICENSE](LICENSE).
