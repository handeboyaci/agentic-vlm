"""ChromaDB-backed vector store for RAG-based target identification.

Requires optional dependencies: ``chromadb``, ``sentence-transformers``.
Install with: ``pip install chromadb sentence-transformers``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _check_deps():
  """Check that chromadb and sentence_transformers are installed."""
  try:
    import chromadb  # noqa: F401
    from sentence_transformers import (  # noqa: F401
      SentenceTransformer,
    )

    return True
  except ImportError:
    logger.warning(
      "chromadb or sentence-transformers not installed. "
      "RAG vector store unavailable. Install with: "
      "pip install chromadb sentence-transformers"
    )
    return False


class VectorStore:
  """Semantic search over drug discovery knowledge base."""

  def __init__(
    self,
    persist_dir: str = "data/rag_vectordb",
    collection_name: str = "drug_discovery_kb",
  ):
    if not _check_deps():
      raise ImportError("chromadb and sentence-transformers required")

    import chromadb
    from sentence_transformers import SentenceTransformer

    self._client = chromadb.PersistentClient(path=persist_dir)
    self._model = SentenceTransformer("all-MiniLM-L6-v2")
    self._collection = self._client.get_or_create_collection(name=collection_name)

  def add_documents(self, documents: list[dict[str, Any]]) -> int:
    """Index documents into ChromaDB."""
    texts = [d["text"] for d in documents]
    metadatas = [{"source": d["source"], "disease": d["disease"]} for d in documents]
    ids = [f"doc_{i}_{hash(d['text'])}" for i, d in enumerate(documents)]
    embeddings = self._model.encode(texts).tolist()
    self._collection.add(
      ids=ids,
      embeddings=embeddings,
      documents=texts,
      metadatas=metadatas,  # type: ignore[arg-type]
    )
    return len(ids)

  def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve top-K relevant documents for a query."""
    query_emb = self._model.encode([query]).tolist()
    res = self._collection.query(query_embeddings=query_emb, n_results=top_k)
    results = []
    if res["documents"]:
      for i in range(len(res["documents"][0])):
        results.append(
          {
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],  # type: ignore[index]
            "score": res["distances"][0][i],  # type: ignore[index]
          }
        )
    return results

  def reset(self):
    """Clear the collection."""
    self._client.delete_collection(self._collection.name)
    self._collection = self._client.get_or_create_collection(self._collection.name)
