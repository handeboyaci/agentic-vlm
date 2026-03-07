"""Uni-Mol predictor skill: binding affinity via fine-tuned Uni-Mol."""
from __future__ import annotations

import logging
import os
from typing import Any

from rdkit import Chem

logger = logging.getLogger(__name__)

_predictor = None


def _ensure_unimol():
  """Lazy-load Uni-Mol predictor (heavy import)."""
  global _predictor
  if _predictor is not None:
    return True
  try:
    from unimol_tools import MolPredict
  except ImportError:
    logger.warning(
      "unimol_tools not installed. "
      "Install with: pip install unimol_tools",
    )
    return False

  model_dir = os.environ.get(
    "UNIMOL_MODEL_DIR", "unimol_binding_model",
  )
  if not os.path.isdir(model_dir):
    logger.warning(
      "Uni-Mol model directory not found: %s. "
      "Fine-tune using notebooks/finetune_unimol.ipynb.",
      model_dir,
    )
    return False
  _predictor = MolPredict(load_model=model_dir)
  logger.info("Loaded Uni-Mol model from %s", model_dir)
  return True


def score_molecules(
  mols: list[Chem.Mol],
  uncertainty_threshold: float = 0.5,
) -> list[dict[str, Any]]:
  """Score molecules using the fine-tuned Uni-Mol model.

  Args:
    mols: List of RDKit Mol objects.
    uncertainty_threshold: Not used (Uni-Mol is deterministic)
      but kept for interface compatibility.

  Returns:
    List of dicts with ``smiles``, ``pka_mean``, ``pka_std``,
    and ``confident`` keys.
  """
  if not _ensure_unimol():
    return []

  import pandas as pd
  import tempfile

  smiles_list = [Chem.MolToSmiles(m) for m in mols if m]
  if not smiles_list:
    return []

  # Uni-Mol expects CSV input
  df = pd.DataFrame({"SMILES": smiles_list})
  with tempfile.NamedTemporaryFile(
    suffix=".csv", mode="w", delete=False,
  ) as f:
    df.to_csv(f, index=False)
    tmp_path = f.name

  try:
    preds = _predictor.predict(data=tmp_path)
    results = []
    for smi, pka in zip(smiles_list, preds.flatten()):
      results.append({
        "smiles": smi,
        "pka_mean": float(pka),
        "pka_std": 0.0,  # deterministic — no MC Dropout
        "confident": True,
      })
    return results
  except Exception as exc:
    logger.warning("Uni-Mol scoring failed: %s", exc)
    return []
  finally:
    os.unlink(tmp_path)
