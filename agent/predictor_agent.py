"""Predictor agent: binding affinity scoring with multiple backends."""

from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem

from agent.base import BaseAgent
from agent.skills import predictor
from config.settings import PredictorConfig

logger = logging.getLogger(__name__)


class PredictorAgent(BaseAgent):
  """Scores molecules using GNN, Uni-Mol, or Vina."""

  BACKENDS = ("gnn", "unimol", "vina")

  def __init__(
    self,
    config: PredictorConfig | None = None,
    scoring: str = "gnn",
  ):
    cfg = config or PredictorConfig()
    super().__init__(cfg)
    if scoring not in self.BACKENDS:
      raise ValueError(
        f"Unknown scoring backend: {scoring!r}. Choose from {self.BACKENDS}",
      )
    self.scoring = scoring
    self._model = None

  def _ensure_model(self) -> None:
    if self.scoring != "gnn" or self._model is not None:
      return
    self._model = predictor.load_model(
      model_path=self.config.model_path,
      atom_feat_dim=self.config.atom_feat_dim,
      hidden_dim=self.config.hidden_dim,
      n_layers=self.config.n_layers,
      dropout=self.config.dropout,
    )

  def _score_gnn(
    self,
    mols: list[Chem.Mol],
    pdb_id: str | None = None,
  ) -> list[dict[str, Any]]:
    """Score via EGNN + MC Dropout."""
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
        logger.warning("GNN could not score: %s", smiles)
        continue
      confident = score["pka_std"] < self.config.uncertainty_threshold
      results.append(
        {
          "smiles": smiles,
          "pka_mean": score["pka_mean"],
          "pka_std": score["pka_std"],
          "confident": confident,
        }
      )
    return results

  def _score_unimol(
    self,
    mols: list[Chem.Mol],
    pdb_id: str | None = None,
  ) -> list[dict[str, Any]]:
    """Score via fine-tuned Uni-Mol."""
    from agent.skills import unimol_predictor

    return unimol_predictor.score_molecules(
      mols,
      uncertainty_threshold=self.config.uncertainty_threshold,
      pdb_id=pdb_id,
    )

  def _score_vina(
    self,
    mols: list[Chem.Mol],
    pdb_id: str | None = None,
  ) -> list[dict[str, Any]]:
    """Score via AutoDock Vina docking."""
    from agent.skills import vina_scorer

    return vina_scorer.score_molecules(mols, pdb_id=pdb_id)

  def execute(
    self,
    mols: list[Chem.Mol],
    pdb_id: str | None = None,
  ) -> list[dict[str, Any]]:
    """Score molecules using the configured backend.

    Args:
      mols: Molecules with 3D conformers.
      pdb_id: Optional PDB ID for protein context.

    Returns:
      List of dicts with ``smiles``, ``pka_mean``,
      ``pka_std``, and ``confident`` keys.
    """
    dispatch = {
      "gnn": self._score_gnn,
      "unimol": self._score_unimol,
      "vina": self._score_vina,
    }
    scorer = dispatch[self.scoring]
    logger.info("Scoring %d molecules with %s", len(mols), self.scoring)
    return scorer(mols, pdb_id=pdb_id)
