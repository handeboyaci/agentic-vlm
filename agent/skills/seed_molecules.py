"""Fetch seed molecules from ChEMBL for a given protein target."""

from __future__ import annotations

import logging
from typing import Optional

import requests
from rdkit import Chem

logger = logging.getLogger(__name__)

CHEMBL_ACTIVITY_URL = (
  "https://www.ebi.ac.uk/chembl/api/data/activity.json"
)
CHEMBL_TARGET_URL = (
  "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
)

# Fallback library if ChEMBL is unreachable
DEFAULT_SEEDS = [
  "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
  "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
  "c1ccc2[nH]c(-c3ccncc3)nc2c1",  # Benzimidazole
  "O=C(O)c1cccnc1",  # Nicotinic acid
]


def _resolve_target_id(
  query: str,
) -> Optional[tuple[str, str]]:
  """Resolve a protein name to a ChEMBL target ID.

  Args:
    query: Protein name (e.g. "Acetylcholinesterase") or
      a ChEMBL ID (e.g. "CHEMBL220").

  Returns:
    Tuple of (chembl_id, preferred_name) or None.
  """
  # If it's already a ChEMBL ID, return as-is
  if query.upper().startswith("CHEMBL"):
    return query.upper(), query.upper()

  resp = requests.get(
    CHEMBL_TARGET_URL,
    params={"q": query, "format": "json", "limit": 5},
    timeout=10,
  )
  resp.raise_for_status()
  targets = resp.json().get("targets", [])

  if not targets:
    return None

  # Prefer SINGLE PROTEIN targets over protein families
  for t in targets:
    if t.get("target_type") == "SINGLE PROTEIN":
      return t["target_chembl_id"], t.get("pref_name", "unknown")

  # Fallback to first result
  t = targets[0]
  return t["target_chembl_id"], t.get("pref_name", "unknown")


def fetch_seed_molecules(
  target_name: str,
  disease: str | None = None,
  max_molecules: int = 20,
  min_activity_um: float = 10.0,
) -> list[str]:
  """Fetch bioactive seed molecules from ChEMBL.

  Queries ChEMBL for known active compounds against the specified
  protein target. Returns validated SMILES strings suitable as
  starting population for the genetic algorithm.

  Args:
    target_name: Protein target name (e.g. "Acetylcholinesterase")
      or a ChEMBL target ID (e.g. "CHEMBL220"). This is the
      primary search key.
    disease: Optional disease name, used as fallback if target
      lookup fails.
    max_molecules: Maximum number of seed molecules to return.
    min_activity_um: Maximum IC50/Ki threshold in µM to filter
      for potent compounds only.

  Returns:
    List of valid SMILES strings. Falls back to DEFAULT_SEEDS
    if ChEMBL is unreachable or returns no results.
  """
  try:
    # Step 1: Resolve the target protein to a ChEMBL ID
    result = _resolve_target_id(target_name)
    if result is None and disease:
      logger.info(
        "Target '%s' not found, falling back to disease '%s'",
        target_name,
        disease,
      )
      result = _resolve_target_id(disease)

    if result is None:
      logger.warning(
        "No ChEMBL target found for '%s', using defaults",
        target_name,
      )
      return DEFAULT_SEEDS

    target_id, pref_name = result
    logger.info(
      "Fetching seeds for %s (%s)", target_id, pref_name
    )

    # Step 2: Fetch bioactive molecules for the target
    act_resp = requests.get(
      CHEMBL_ACTIVITY_URL,
      params={
        "target_chembl_id": target_id,
        "standard_type__in": "IC50,Ki,Kd",
        "standard_value__lte": min_activity_um * 1000,  # nM
        "format": "json",
        "limit": max_molecules * 3,  # extra for filtering
      },
      timeout=15,
    )
    act_resp.raise_for_status()
    activities = act_resp.json().get("activities", [])

    # Step 3: Extract and validate unique SMILES
    seen: set[str] = set()
    seeds: list[str] = []

    for act in activities:
      smi = act.get("canonical_smiles")
      if not smi or smi in seen:
        continue
      mol = Chem.MolFromSmiles(smi)
      if mol is None:
        continue
      # Filter out very large or tiny molecules
      n_atoms = mol.GetNumHeavyAtoms()
      if n_atoms < 5 or n_atoms > 70:
        continue
      seen.add(smi)
      seeds.append(smi)
      if len(seeds) >= max_molecules:
        break

    if seeds:
      logger.info(
        "Fetched %d seed molecules from ChEMBL (%s)",
        len(seeds),
        pref_name,
      )
      return seeds

    logger.warning(
      "No valid molecules from ChEMBL for '%s', using defaults",
      target_name,
    )
    return DEFAULT_SEEDS

  except Exception as exc:
    logger.warning(
      "ChEMBL fetch failed (%s), using default seeds", exc
    )
    return DEFAULT_SEEDS
