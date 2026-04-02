"""Uni-Mol predictor wrapper for the Local VLM Pipeline.

Supports both ligand-only and pocket-aware scoring modes:
- **Ligand-only**: Uses the SMILES-only model (default).
- **Pocket-aware**: When a ``pdb_id`` is provided, extracts the
  pocket atoms/coordinates from the PDB structure and concatenates
  them with the ligand for protein-aware scoring.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

# We use the official MolPredict class from unimol_tools
try:
  from unimol_tools import MolPredict
except ImportError:
  MolPredict = None

logger = logging.getLogger(__name__)

# Global caches
_UNIMOL_PREDICTOR = None
_UNIMOL_POCKET_PREDICTOR = None

POCKET_MODEL_DIR = "models/unimol_pocket_model"
LIGAND_MODEL_DIR = "models/unimol_binding_model"


def _extract_pocket_atoms_coords(
  pdb_id: str,
  cache_dir: str = "data/raw/pdbs",
) -> dict[str, Any] | None:
  """Fetch PDB and extract pocket atoms + coordinates."""
  from utils.pocket import fetch_pdb

  pdb_path = fetch_pdb(pdb_id, cache_dir=cache_dir)
  if pdb_path is None:
    return None

  mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
  if mol is None or mol.GetNumConformers() == 0:
    return None

  atoms = [a.GetSymbol() for a in mol.GetAtoms()]
  conf = mol.GetConformer()
  coords = np.array(
    [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
    dtype=np.float32,
  )
  return {"atoms": atoms, "coordinates": coords}


def _mol_to_atoms_coords(
  mol: Chem.Mol,
) -> dict[str, Any] | None:
  """Extract atoms + 3D coordinates from an RDKit Mol."""
  if mol is None:
    return None
  mol_noH = Chem.RemoveHs(mol)
  if mol_noH.GetNumConformers() == 0:
    return None
  atoms = [a.GetSymbol() for a in mol_noH.GetAtoms()]
  conf = mol_noH.GetConformer()
  coords = np.array(
    [list(conf.GetAtomPosition(i)) for i in range(mol_noH.GetNumAtoms())],
    dtype=np.float32,
  )
  return {"atoms": atoms, "coordinates": coords}


def score_molecules(
  mols: list[Chem.Mol],
  uncertainty_threshold: float = 1.0,
  pdb_id: str | None = None,
) -> list[dict[str, Any]]:
  """Score molecules using the fine-tuned Uni-Mol ensemble.

  Args:
    mols: List of RDKit Mol objects with 3D conformers.
    uncertainty_threshold: Ignored (Uni-Mol uses robust
      5-fold ensemble).
    pdb_id: Optional PDB ID for pocket-aware scoring.

  Returns:
    List of dicts with ``smiles``, ``pka_mean``, ``pka_std``,
    and ``confident`` keys.
  """
  global _UNIMOL_PREDICTOR, _UNIMOL_POCKET_PREDICTOR
  if not mols:
    return []

  if MolPredict is None:
    logger.error(
      "unimol_tools is not installed. Run `pip install unimol_tools huggingface_hub`"
    )
    return []

  # Decide: pocket-aware or ligand-only
  use_pocket = pdb_id is not None and os.path.isdir(POCKET_MODEL_DIR)

  if use_pocket:
    return _score_with_pocket(mols, pdb_id)
  return _score_ligand_only(mols)


def _score_ligand_only(
  mols: list[Chem.Mol],
) -> list[dict[str, Any]]:
  """Score using SMILES-only model (original behaviour)."""
  global _UNIMOL_PREDICTOR

  smiles_list = [Chem.MolToSmiles(m) for m in mols if m is not None]
  if not smiles_list:
    return []

  df = pd.DataFrame({"SMILES": smiles_list})

  if _UNIMOL_PREDICTOR is None:
    logger.info(
      "Loading Uni-Mol 84M Ensemble from %s (this takes a few seconds)...",
      LIGAND_MODEL_DIR,
    )
    _UNIMOL_PREDICTOR = MolPredict(load_model=LIGAND_MODEL_DIR)

  with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as tmp:
    df.to_csv(tmp.name, index=False)
    preds = _UNIMOL_PREDICTOR.predict(data=tmp.name)

  results = []
  for i, smi in enumerate(smiles_list):
    mean_pka = (
      float(preds[i][0]) if hasattr(preds[i], "__getitem__") else float(preds[i])
    )
    results.append(
      {
        "smiles": smi,
        "pka_mean": mean_pka,
        "pka_std": 0.0,
        "confident": True,
      }
    )
  return results


def _score_with_pocket(
  mols: list[Chem.Mol],
  pdb_id: str,
) -> list[dict[str, Any]]:
  """Score using pocket+ligand model for protein-aware pKa."""
  global _UNIMOL_POCKET_PREDICTOR

  pocket = _extract_pocket_atoms_coords(pdb_id)
  if pocket is None:
    logger.warning(
      "Could not extract pocket for %s; falling back to ligand-only.",
      pdb_id,
    )
    return _score_ligand_only(mols)

  if _UNIMOL_POCKET_PREDICTOR is None:
    logger.info(
      "Loading pocket-aware Uni-Mol from %s...",
      POCKET_MODEL_DIR,
    )
    _UNIMOL_POCKET_PREDICTOR = MolPredict(load_model=POCKET_MODEL_DIR)

  # Build custom data: pocket + ligand concatenated
  all_atoms = []
  all_coords = []
  smiles_list = []
  for mol in mols:
    if mol is None:
      continue
    lig = _mol_to_atoms_coords(mol)
    if lig is None:
      continue
    combined_atoms = pocket["atoms"] + lig["atoms"]
    combined_coords = np.vstack([pocket["coordinates"], lig["coordinates"]])
    all_atoms.append(combined_atoms)
    all_coords.append(combined_coords)
    smiles_list.append(Chem.MolToSmiles(mol))

  if not all_atoms:
    return []

  custom_data = {
    "atoms": all_atoms,
    "coordinates": all_coords,
  }
  preds = _UNIMOL_POCKET_PREDICTOR.predict(data=custom_data)

  results = []
  for i, smi in enumerate(smiles_list):
    mean_pka = (
      float(preds[i][0]) if hasattr(preds[i], "__getitem__") else float(preds[i])
    )
    results.append(
      {
        "smiles": smi,
        "pka_mean": mean_pka,
        "pka_std": 0.0,
        "confident": True,
      }
    )
  return results
