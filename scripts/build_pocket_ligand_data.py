"""Build pocket+ligand atom/coordinate data for Uni-Mol training.

Reads LP-PDBBind CSV + refined-set pocket/ligand files and
produces a pickle file containing the dict format expected by
``unimol_tools.MolTrain.fit()``.

Usage
-----
  uv run python scripts/build_pocket_ligand_data.py \
      --output data/unimol_pocket_ligand.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

CSV_PATH = "data/raw/LP_PDBBind.csv"
REFINED_DIR = "data/refined-set"


def extract_atoms_coords(
  pdb_id: str,
  refined_dir: str = REFINED_DIR,
) -> dict | None:
  """Extract pocket + ligand atoms and coords for one complex."""
  pocket_path = os.path.join(refined_dir, pdb_id, f"{pdb_id}_pocket.pdb")
  ligand_path = os.path.join(refined_dir, pdb_id, f"{pdb_id}_ligand.sdf")
  if not os.path.exists(pocket_path) or not os.path.exists(ligand_path):
    return None

  # Parse pocket PDB (keep hydrogens = False for cleaner input)
  pocket_mol = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
  if pocket_mol is None or pocket_mol.GetNumConformers() == 0:
    return None

  # Parse ligand SDF
  ligand_mol = Chem.MolFromMolFile(ligand_path, removeHs=True)
  if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
    return None

  # Extract pocket atoms + coords
  pocket_atoms = [a.GetSymbol() for a in pocket_mol.GetAtoms()]
  pocket_conf = pocket_mol.GetConformer()
  pocket_coords = np.array(
    [list(pocket_conf.GetAtomPosition(i)) for i in range(pocket_mol.GetNumAtoms())],
    dtype=np.float32,
  )

  # Extract ligand atoms + coords
  ligand_atoms = [a.GetSymbol() for a in ligand_mol.GetAtoms()]
  ligand_conf = ligand_mol.GetConformer()
  ligand_coords = np.array(
    [list(ligand_conf.GetAtomPosition(i)) for i in range(ligand_mol.GetNumAtoms())],
    dtype=np.float32,
  )

  # Concatenate: [pocket_atoms..., ligand_atoms...]
  all_atoms = pocket_atoms + ligand_atoms
  all_coords = np.vstack([pocket_coords, ligand_coords])

  return {"atoms": all_atoms, "coordinates": all_coords}


def build_dataset(
  csv_path: str = CSV_PATH,
  refined_dir: str = REFINED_DIR,
  clean_level: str = "CL1",
) -> dict:
  """Build the full pocket+ligand dataset as a dict.

  Returns a dict with keys: ``atoms``, ``coordinates``,
  ``target``, ``pdb_id``, ``split``.
  """
  df = pd.read_csv(csv_path, index_col=0)

  # Apply LP-PDBBind cleanup filters
  if clean_level in df.columns:
    df = df[df[clean_level] & ~df["covalent"]]
    logger.info(
      "After %s + non-covalent filter: %d complexes",
      clean_level,
      len(df),
    )

  all_atoms = []
  all_coords = []
  all_targets = []
  all_pdb_ids = []
  all_splits = []
  skipped = 0

  for pdb_id, row in df.iterrows():
    result = extract_atoms_coords(str(pdb_id), refined_dir)
    if result is None:
      skipped += 1
      continue

    all_atoms.append(result["atoms"])
    all_coords.append(result["coordinates"])
    all_targets.append(float(row["value"]))
    all_pdb_ids.append(str(pdb_id))
    all_splits.append(str(row["new_split"]))

  logger.info(
    "Built %d complexes (%d skipped due to missing/bad files)",
    len(all_atoms),
    skipped,
  )

  dataset = {
    "atoms": all_atoms,
    "coordinates": all_coords,
    "target": np.array(all_targets, dtype=np.float32),
    "pdb_id": all_pdb_ids,
    "split": all_splits,
  }
  return dataset


def main():
  parser = argparse.ArgumentParser(description="Build pocket+ligand data for Uni-Mol.")
  parser.add_argument(
    "--output",
    type=str,
    default="data/unimol_pocket_ligand.pkl",
    help="Output pickle path",
  )
  parser.add_argument(
    "--clean-level",
    type=str,
    default="CL1",
    choices=["CL1", "CL2", "CL3"],
  )
  args = parser.parse_args()

  dataset = build_dataset(clean_level=args.clean_level)

  # Save
  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  with open(args.output, "wb") as f:
    pickle.dump(dataset, f)
  logger.info("Saved to %s", args.output)

  # Print split stats
  splits = dataset["split"]
  for s in ["train", "val", "test"]:
    count = sum(1 for x in splits if x == s)
    logger.info("  %s: %d complexes", s, count)


if __name__ == "__main__":
  main()
