import chromadb
from sentence_transformers import SentenceTransformer
from typing import Any


class VectorStore:
  def __init__(
    self, persist_dir="data/rag_vectordb", collection_name="drug_discovery_kb"
  ):
    self._client = chromadb.PersistentClient(path=persist_dir)
    self._model = SentenceTransformer("all-MiniLM-L6-v2")
    self._collection = self._client.get_or_create_collection(name=collection_name)

  def add_documents(self, documents: list[dict[str, Any]]) -> int:
    texts = [d["text"] for d in documents]
    metadatas = [{"source": d["source"], "disease": d["disease"]} for d in documents]
    ids = [f"doc_{i}_{hash(d['text'])}" for i, d in enumerate(documents)]
    embeddings = self._model.encode(texts).tolist()
    self._collection.add(
      ids=ids,
      embeddings=embeddings,
      documents=texts,
      metadatas=metadatas,  # type: ignore
    )
    return len(ids)

  def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    query_emb = self._model.encode([query]).tolist()
    res = self._collection.query(query_embeddings=query_emb, n_results=top_k)
    results = []
    if res["documents"]:
      for i in range(len(res["documents"][0])):
        results.append(
          {
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],  # type: ignore
            "score": res["distances"][0][i],  # type: ignore
          }
        )
    return results

  def reset(self):
    self._client.delete_collection(self._collection.name)
    self._collection = self._client.get_or_create_collection(self._collection.name)
