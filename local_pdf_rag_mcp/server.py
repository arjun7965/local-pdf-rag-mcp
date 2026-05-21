"""MCP server exposing PDF retrieval tools to Claude Code (or any MCP client).

Three tools:
  - ingest_pdf:        chunk + embed a PDF (or every PDF in a folder) into a collection
  - list_collections:  show indexed collections and their chunk counts
  - search:            return the top matching chunks for a query (with citations)

By design `search` returns raw chunks and lets the model synthesize the
answer, rather than generating an answer inside the tool. This keeps the
server simple, model-agnostic, and token-cheap.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .chunking import PdfExtractionError, chunk_pages, extract_pages
from .store import VectorStore

mcp = FastMCP("pdf-rag")
_store = VectorStore()


def _ingest_one(pdf_path: Path, collection: str) -> dict:
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, source=pdf_path.name)
    added = _store.add_chunks(collection, chunks)
    return {"file": pdf_path.name, "pages": len(pages), "chunks_added": added}


@mcp.tool()
def ingest_pdf(path: str, collection: str = "default") -> str:
    """Ingest a PDF file, or every PDF in a folder, into a named collection.

    Args:
        path: Path to a .pdf file or a directory containing PDFs.
        collection: Name of the collection to store chunks in. Use separate
            collections to keep unrelated documents from mixing in search.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return f"Error: path does not exist: {target}"

    pdfs = (
        [target]
        if target.is_file() and target.suffix.lower() == ".pdf"
        else sorted(target.glob("*.pdf"))
        if target.is_dir()
        else []
    )
    if not pdfs:
        return f"Error: no PDF files found at {target}"

    results, errors = [], []
    for pdf in pdfs:
        try:
            results.append(_ingest_one(pdf, collection))
        except PdfExtractionError as exc:
            errors.append(str(exc))
        except Exception as exc:  # keep going on a single bad file
            errors.append(f"{pdf.name}: unexpected error: {exc}")

    lines = [f"Ingested into collection '{collection}':"]
    for r in results:
        lines.append(f"  - {r['file']}: {r['pages']} pages, {r['chunks_added']} chunks")
    if errors:
        lines.append("Skipped:")
        lines.extend(f"  - {e}" for e in errors)
    return "\n".join(lines)


@mcp.tool()
def list_collections() -> str:
    """List all indexed collections and how many chunks each contains."""
    cols = _store.list_collections()
    if not cols:
        return "No collections yet. Use ingest_pdf to index a PDF first."
    return "\n".join(f"- {c['name']}: {c['chunks']} chunks" for c in cols)


@mcp.tool()
def search(query: str, collection: str = "default", top_k: int = 5) -> str:
    """Search a collection and return the most relevant chunks with citations.

    Returns raw passages for the model to read and synthesize an answer from.
    Each result includes its source filename and page number.

    Args:
        query: The question or search phrase.
        collection: Which collection to search.
        top_k: How many chunks to return (keep small to stay token-cheap).
    """
    hits = _store.search(collection, query, top_k=top_k)
    if not hits:
        return (
            f"No results in collection '{collection}'. "
            f"Check the collection name with list_collections, or ingest a PDF first."
        )
    blocks = []
    for i, h in enumerate(hits, start=1):
        cite = f"{h['source']}, p.{h['page']}"
        blocks.append(f"[{i}] ({cite})\n{h['text']}")
    return "\n\n".join(blocks)


def main() -> None:
    """Entry point for the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    sys.exit(main())
