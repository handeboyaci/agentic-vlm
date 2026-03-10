"""Fetch seed molecules from ChEMBL for a given disease target."""

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


def fetch_seed_molecules(
  disease: str,
  max_molecules: int = 20,
  min_activity_um: float = 10.0,
) -> list[str]:
  """Fetch bioactive seed molecules from ChEMBL for a disease.

  Queries ChEMBL for known active compounds against targets
  associated with the disease. Returns validated SMILES strings
  suitable as starting population for the genetic algorithm.

  Args:
    disease: Disease name (e.g. "Alzheimer's", "Cancer").
    max_molecules: Maximum number of seed molecules to return.
    min_activity_um: Maximum IC50/Ki threshold in µM to filter
      for potent compounds only.

  Returns:
    List of valid SMILES strings. Falls back to DEFAULT_SEEDS
    if ChEMBL is unreachable or returns no results.
  """
  try:
    # Step 1: Find ChEMBL target IDs for the disease
    resp = requests.get(
      CHEMBL_TARGET_URL,
      params={"q": disease, "format": "json", "limit": 5},
      timeout=10,
    )
    resp.raise_for_status()
    targets = resp.json().get("targets", [])

    if not targets:
      logger.warning(
        "No ChEMBL targets found for '%s', using defaults", disease
      )
      return DEFAULT_SEEDS

    # Step 2: Fetch bioactive molecules for the top target
    target_id = targets[0].get("target_chembl_id")
    logger.info(
      "Fetching seeds from ChEMBL target %s (%s)",
      target_id,
      targets[0].get("pref_name", "unknown"),
    )

    act_resp = requests.get(
      CHEMBL_ACTIVITY_URL,
      params={
        "target_chembl_id": target_id,
        "standard_type__in": "IC50,Ki,Kd",
        "standard_value__lte": min_activity_um * 1000,  # nM
        "format": "json",
        "limit": max_molecules * 3,  # fetch extra for filtering
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
        "Fetched %d seed molecules from ChEMBL", len(seeds)
      )
      return seeds

    logger.warning(
      "No valid molecules from ChEMBL for '%s', using defaults",
      disease,
    )
    return DEFAULT_SEEDS

  except Exception as exc:
    logger.warning(
      "ChEMBL fetch failed (%s), using default seeds", exc
    )
    return DEFAULT_SEEDS
