"""Vina docking skill: physics-based binding score via AutoDock Vina."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

_VINA_AVAILABLE: bool | None = None


def _check_vina() -> bool:
  """Check if AutoDock Vina and Meeko are available."""
  global _VINA_AVAILABLE
  if _VINA_AVAILABLE is not None:
    return _VINA_AVAILABLE
  try:
    import vina  # noqa: F401
    import meeko  # noqa: F401

    _VINA_AVAILABLE = True
  except ImportError:
    logger.warning(
      "Vina/Meeko not installed. "
      "Install with: conda install -c conda-forge vina meeko",
    )
    _VINA_AVAILABLE = False
  return _VINA_AVAILABLE


def _mol_to_pdbqt(mol: Chem.Mol) -> str | None:
  """Convert RDKit Mol to PDBQT string using Meeko."""
  try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    mol_h = Chem.AddHs(mol)

    if mol_h.GetNumConformers() == 0:
      AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
      AllChem.MMFFOptimizeMolecule(mol_h)

    mol_setup = preparator.prepare(mol_h)[0]
    pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(
      mol_setup,
    )
    return pdbqt_str if is_ok else None
  except Exception as exc:
    logger.warning("Meeko prep failed: %s", exc)
    return None


def _prepare_receptor(pdb_path: str) -> str | None:
  """Prepare receptor from PDB file (basic conversion)."""
  try:
    from meeko import PDBQTWriterLegacy

    # Simple conversion — production code should use
    # ADFR Suite's prepare_receptor for proper handling
    pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
    if os.path.exists(pdbqt_path):
      return pdbqt_path

    # Minimal conversion via Meeko/obabel fallback
    import subprocess
    import shutil

    obabel_cmd = shutil.which("obabel") or "/opt/homebrew/bin/obabel"

    result = subprocess.run(
      [
        obabel_cmd, pdb_path, "-O", pdbqt_path,
        "-xr", "--partialcharge", "gasteiger",
      ],
      capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.exists(pdbqt_path):
      return pdbqt_path
    return None
  except Exception as exc:
    logger.warning("Receptor prep failed: %s", exc)
    return None


def dock_molecule(
  mol: Chem.Mol,
  pdb_id: str | None = None,
  center: tuple[float, float, float] = (0.0, 0.0, 0.0),
  box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
  exhaustiveness: int = 8,
) -> dict[str, Any] | None:
  """Dock a molecule and return the binding score.

  Args:
    mol: RDKit Mol (with or without 3D conformer).
    pdb_id: PDB ID to fetch receptor structure.
    center: Docking box center (x, y, z) in Angstroms.
    box_size: Docking box dimensions.
    exhaustiveness: Vina search exhaustiveness.

  Returns:
    Dict with ``vina_score`` (kcal/mol) and ``pka_mean``
    (converted estimate), or None on failure.
  """
  if not _check_vina():
    return None

  from vina import Vina

  # Prepare ligand
  pdbqt_lig = _mol_to_pdbqt(mol)
  if pdbqt_lig is None:
    return None

  # Prepare receptor
  receptor_pdbqt = None
  if pdb_id:
    from utils.pocket import fetch_pdb, extract_ligand_center

    pdb_path = fetch_pdb(pdb_id, cache_dir="data/raw/pdbs")
    if pdb_path:
      receptor_pdbqt = _prepare_receptor(pdb_path)
      calc_center = extract_ligand_center(pdb_path)
      if calc_center:
        center = calc_center

  if receptor_pdbqt is None:
    logger.warning("No receptor available; scoring only.")
    return None

  try:
    v = Vina(sf_name="vina")
    v.set_receptor(receptor_pdbqt)

    # Write ligand to temp file
    with tempfile.NamedTemporaryFile(
      suffix=".pdbqt", mode="w", delete=False,
    ) as f:
      f.write(pdbqt_lig)
      lig_path = f.name

    v.set_ligand_from_file(lig_path)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness)

    score = v.score()[0]  # kcal/mol (negative = better)
    os.unlink(lig_path)

    # Rough conversion: pKa ≈ -score / 1.366
    # Based on ΔG = -RT ln(Kd), at 298K: 1 pKa unit ≈ 1.366 kcal/mol
    pka_estimate = -score / 1.366

    return {
      "vina_score": float(score),
      "pka_mean": float(pka_estimate),
      "pka_std": 0.0,  # Vina is deterministic
      "confident": True,
    }
  except Exception as exc:
    logger.warning("Vina docking failed: %s", exc)
    return None


def score_molecules(
  mols: list[Chem.Mol],
  pdb_id: str | None = None,
) -> list[dict[str, Any]]:
  """Score a list of molecules using Vina docking.

  Args:
    mols: List of RDKit Mol objects.
    pdb_id: PDB ID for receptor structure.

  Returns:
    List of result dicts with ``smiles``, ``pka_mean``,
    ``vina_score``, and ``confident`` keys.
  """
  if not _check_vina():
    return []

  results = []
  for mol in mols:
    smiles = Chem.MolToSmiles(mol)
    score = dock_molecule(mol, pdb_id=pdb_id)
    if score is None:
      logger.warning("Vina failed for: %s", smiles)
      continue
    score["smiles"] = smiles
    results.append(score)
  return results
