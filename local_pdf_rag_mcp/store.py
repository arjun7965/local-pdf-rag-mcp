"""Vector store backed by ChromaDB with a local embedding model.

Fully local by default: ChromaDB persists to disk and the embedding model
(sentence-transformers all-MiniLM-L6-v2) runs on-device, so no API keys and
no data leaves the machine. The embedding model is swappable via the
PDF_RAG_EMBED_MODEL env var for users who want a different local model.

A cross-encoder reranker runs over the top-N vector hits to improve
precision before the chunks reach the model. This catches relevant
passages that bi-encoder cosine similarity misses (semantically close but
phrased differently). Reranking can be disabled with PDF_RAG_RERANK=0 if
you want pure vector-search behaviour.
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
DEFAULT_RERANK_MODEL = os.environ.get(
    "PDF_RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_ENABLED = os.environ.get("PDF_RAG_RERANK", "1") != "0"


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
        # Reranker is lazy-loaded on first search so server startup stays
        # snappy when the user only ingests.
        self._reranker = None

    def _get_reranker(self):
        if not RERANK_ENABLED:
            return None
        if self._reranker is None:
            # Local import keeps the heavy sentence-transformers import path
            # off the critical startup path when reranking is disabled.
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(DEFAULT_RERANK_MODEL)
        return self._reranker

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
            {
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
            }
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

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 8,
        candidates: int = 20,
    ) -> list[dict]:
        """Return the top_k most relevant chunks.

        When reranking is enabled (default), pulls `candidates` chunks from
        vector search, scores each (query, chunk) pair with a cross-encoder,
        and returns the top_k after rerank. When disabled, returns the
        top_k from vector search directly.

        Each returned dict has: text, source, page, distance (vector
        distance from query), and rerank_score (cross-encoder relevance,
        only present when reranking ran).
        """
        try:
            col = self.client.get_collection(
                name=collection, embedding_function=self._embed
            )
        except Exception:
            return []

        reranker = self._get_reranker()
        # Over-fetch when reranking so the cross-encoder has real choices.
        n_retrieve = max(top_k, candidates) if reranker is not None else top_k
        res = col.query(query_texts=[query], n_results=n_retrieve)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits: list[dict] = []
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "distance": dist,
                }
            )

        if reranker is None or not hits:
            return hits[:top_k]

        # Cross-encoder scores (query, chunk) pairs; higher = more relevant.
        pairs = [(query, h["text"]) for h in hits]
        scores = reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]

    def list_collections(self) -> list[dict]:
        result = []
        for col in self.client.list_collections():
            c = self.client.get_collection(
                name=col.name, embedding_function=self._embed
            )
            result.append({"name": col.name, "chunks": c.count()})
        return result

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its chunks. Returns False if it
        didn't exist; lets any other failure propagate as a real error."""
        existing = {c.name for c in self.client.list_collections()}
        if name not in existing:
            return False
        self.client.delete_collection(name=name)
        return True
