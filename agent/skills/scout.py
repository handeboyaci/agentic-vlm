"""Scout agent skills: target identification and constraint derivation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from config.settings import ScoutConfig

logger = logging.getLogger(__name__)

_DEFAULT_CFG = ScoutConfig()

# Static knowledge base (fallback if RAG is unavailable).
_TARGET_DB: dict[str, dict[str, str]] = {
  "Alzheimer's": {
    "name": "BACE1",
    "pdb_id": "4B7R",
    "uniprot": "P56817",
    "location": "CNS",
  },
  "Cancer": {
    "name": "EGFR",
    "pdb_id": "1M17",
    "uniprot": "P00533",
    "location": "Systemic",
  },
  "Diabetes": {
    "name": "DPP4",
    "pdb_id": "1X70",
    "uniprot": "P27487",
    "location": "Systemic",
  },
  "COVID-19": {
    "name": "MPro",
    "pdb_id": "6LU7",
    "uniprot": "P0DTD1",
    "location": "Systemic",
  },
}


def rag_search(disease_name: str) -> dict[str, Any]:
  """Orchestrate RAG: retrieve from ChromaDB → synthesise with LLM."""
  try:
    from rag.vector_store import VectorStore
    from rag.llm_synthesizer import synthesise

    store = VectorStore()
    docs = store.retrieve(disease_name, top_k=5)
    if not docs:
      logger.warning(
        "No RAG documents found for: %s",
        disease_name,
      )
      return {}

    result = synthesise(disease_name, docs)
    # Normalise key: LLM returns "target_name", pipeline expects "name"
    if "target_name" in result and "name" not in result:
      result["name"] = result.pop("target_name")
    logger.info(
      "Target identified via RAG: %s",
      result.get("name"),
    )
    return result
  except Exception as exc:
    logger.warning("RAG search failed: %s", exc)
    return {}


def identify_target(
  disease_name: str,
  search_func: Optional[Callable[[str], dict[str, Any]]] = None,
) -> dict[str, Any]:
  """Identify the primary protein target for *disease_name*.

  Args:
    disease_name: Human-readable disease string.
    search_func: Optional callable injected for testing or
      alternative search backends.  Defaults to ``rag_search``.

  Returns:
    Dict with at least ``name`` and ``location`` keys.
  """
  if search_func is None:
    search_func = rag_search

  result = search_func(disease_name)
  if result and result.get("name", "Unknown") != "Unknown":
    return result

  # Fallback to static DB
  target = _TARGET_DB.get(disease_name)
  if target:
    logger.info(
      "Falling back to static DB for %s: %s",
      disease_name,
      target["name"],
    )
    return target

  raise ValueError(
    f"Could not identify a target protein for '{disease_name}'. "
    "RAG search failed and no static fallback exists."
  )


def determine_constraints(
  target_info: dict[str, Any],
) -> dict[str, Any]:
  """Derive chemical constraints from target metadata.

  CNS targets receive stricter BBB-penetrant constraints.
  """
  constraints: dict[str, Any] = {
    "max_mw": 500.0,
    "max_logp": 5.0,
    "min_hbd": 0,
    "max_hbd": 5,
    "min_hba": 0,
    "max_hba": 10,
  }
  location = target_info.get(
    "location",
    _DEFAULT_CFG.default_location,
  )
  if location == "CNS":
    constraints.update(
      {
        "max_mw": _DEFAULT_CFG.cns_max_mw,
        "max_logp": _DEFAULT_CFG.cns_max_logp,
        "max_hbd": _DEFAULT_CFG.cns_max_hbd,
        "max_psa": _DEFAULT_CFG.cns_max_psa,
      }
    )
    logger.info(
      "CNS target detected — applying stricter BBB constraints.",
    )
  return constraints
