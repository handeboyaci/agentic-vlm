"""Predictor agent: binding affinity scoring with uncertainty."""
from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem

from agent.base import BaseAgent
from agent.skills import predictor
from config.settings import PredictorConfig

logger = logging.getLogger(__name__)


class PredictorAgent(BaseAgent):
  """Scores molecules using a GNN predictor with MC Dropout."""

  def __init__(self, config: PredictorConfig | None = None):
    cfg = config or PredictorConfig()
    super().__init__(cfg)
    self._model = None

  def _ensure_model(self) -> None:
    if self._model is None:
      self._model = predictor.load_model(
        model_path=self.config.model_path,
        atom_feat_dim=self.config.atom_feat_dim,
        hidden_dim=self.config.hidden_dim,
        n_layers=self.config.n_layers,
        dropout=self.config.dropout,
      )

  def execute(
    self,
    mols: list[Chem.Mol],
    pdb_id: str | None = None,
  ) -> list[dict[str, Any]]:
    """Score each molecule and flag uncertainty.

    Args:
      mols: Molecules with 3D conformers.
      pdb_id: Optional PDB ID for ESM-2 cross-attention.

    Returns:
      List of dicts with ``smiles``, ``pka_mean``, ``pka_std``,
      and ``confident`` keys.
    """
    self._ensure_model()
    results: list[dict[str, Any]] = []
    for mol in mols:
      smiles = Chem.MolToSmiles(mol)
      score = predictor.score_molecule(
        mol,
        self._model,
        mc_samples=self.config.mc_samples,
        pdb_id=pdb_id,
      )
      if score is None:
        logger.warning("Could not score: %s", smiles)
        continue
      confident = (
        score["pka_std"] < self.config.uncertainty_threshold
      )
      results.append({
        "smiles": smiles,
        "pka_mean": score["pka_mean"],
        "pka_std": score["pka_std"],
        "confident": confident,
      })
    return results
