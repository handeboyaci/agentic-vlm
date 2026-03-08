import argparse
import logging
from rag.knowledge_base import build_knowledge_base
from rag.vector_store import VectorStore

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Build RAG index for drug discovery.")
  parser.add_argument(
    "--diseases",
    type=str,
    default="Alzheimer's,Cancer,COVID-19",
    help="Comma-separated diseases",
  )
  parser.add_argument(
    "--reset", action="store_true", help="Reset version store before adding"
  )
  args = parser.parse_args()

  diseases = [d.strip() for d in args.diseases.split(",")]
  logger.info("Building knowledge base for: %s", diseases)

  docs = build_knowledge_base(diseases)
  logger.info("Fetched %d documents.", len(docs))

  store = VectorStore()
  if args.reset:
    logger.info("Resetting vector store...")
    store.reset()

  n = store.add_documents(docs)
  logger.info("Successfully indexed %d documents into ChromaDB.", n)

  # Smoke test
  logger.info("Running smoke test for 'Alzheimer's'...")
  results = store.retrieve("Alzheimer's", top_k=2)
  for r in results:
    logger.info(" - [%s] %s...", r["metadata"]["source"], r["text"][:100])


if __name__ == "__main__":
  main()
