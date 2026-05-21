# local-pdf-rag-mcp

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
overlapping ~350-token chunks that respect paragraph boundaries, embeds each
chunk locally, and stores them in ChromaDB. At query time the server embeds
your question, finds the closest chunks, and returns them. The model reads
those chunks and writes the answer — the server deliberately does **not**
generate answers itself, which keeps it simple and model-agnostic.

## Requirements

- Python 3.10+
- ~100 MB disk for the default embedding model (downloaded on first run)

## Install

```bash
git clone https://github.com/arjun7965/local-pdf-rag-mcp.git
cd local-pdf-rag-mcp
pip install -e .
```

## Register with Claude Code

Add the server to Claude Code:

```bash
claude mcp add pdf-rag -- local-pdf-rag-mcp
```

Or add it manually to your MCP config:

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

The server exposes three tools. In practice you just talk to Claude and it
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
| `search(query, collection="default", top_k=5)` | Return the most relevant chunks with citations. |

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF_RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model name. |
| `PDF_RAG_DB_PATH` | `~/.local_pdf_rag_mcp/chroma` | Where the vector store lives on disk. |

## Limitations

- **No OCR.** Scanned or image-only PDFs have no extractable text; the server
  detects this and returns a clear error rather than indexing nothing.
- **Encrypted PDFs** open only if they use an empty password.
- Tuned for a single-machine, single-user workflow. For multi-user or
  very-large-scale deployments you'd swap ChromaDB for a hosted vector store.

## License

MIT — see [LICENSE](LICENSE).
