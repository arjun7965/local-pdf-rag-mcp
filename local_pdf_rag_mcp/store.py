"""Vector store backed by ChromaDB with a local embedding model.

Fully local by default: ChromaDB persists to disk and the embedding model
(sentence-transformers all-MiniLM-L6-v2) runs on-device, so no API keys and
no data leaves the machine. The embedding model is swappable via the
PDF_RAG_EMBED_MODEL env var for users who want a different local model.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .chunking import Chunk

DEFAULT_EMBED_MODEL = os.environ.get(
    "PDF_RAG_EMBED_MODEL", "all-MiniLM-L6-v2"
)


def _default_db_path() -> Path:
    # Honor an override, else a stable per-user location.
    override = os.environ.get("PDF_RAG_DB_PATH")
    if override:
        return Path(override)
    return Path.home() / ".local_pdf_rag_mcp" / "chroma"


class VectorStore:
    """Thin wrapper over a persistent ChromaDB client."""

    def __init__(self, db_path: Path | None = None, embed_model: str | None = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self._embed = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model or DEFAULT_EMBED_MODEL
        )

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, embedding_function=self._embed
        )

    def add_chunks(self, collection: str, chunks: list[Chunk]) -> int:
        """Embed and store chunks. Returns the number added."""
        if not chunks:
            return 0
        col = self._collection(collection)
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        # Batch to keep memory bounded on large docs.
        batch = 256
        for i in range(0, len(chunks), batch):
            col.add(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )
        return len(chunks)

    def search(self, collection: str, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most similar chunks with metadata and distance."""
        try:
            col = self.client.get_collection(
                name=collection, embedding_function=self._embed
            )
        except Exception:
            return []
        res = col.query(query_texts=[query], n_results=top_k)
        out: list[dict] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "distance": dist,
                }
            )
        return out

    def list_collections(self) -> list[dict]:
        result = []
        for col in self.client.list_collections():
            c = self.client.get_collection(
                name=col.name, embedding_function=self._embed
            )
            result.append({"name": col.name, "chunks": c.count()})
        return result
