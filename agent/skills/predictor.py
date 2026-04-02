"""Predictor agent skills: GNN scoring with uncertainty."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from rdkit import Chem

from data.lp_pdbbind import atom_features
from models.gnn_predictor import GNNPredictor
from models.protein_encoder import precompute_esm2_embedding
from utils.pocket import extract_pocket_seq, fetch_pdb

logger = logging.getLogger(__name__)

_PROTEIN_EMB_CACHE: dict[str, torch.Tensor] = {}


def _mol_to_tensors(
  mol: Chem.Mol,
) -> tuple[torch.Tensor, torch.Tensor] | None:
  """Convert an RDKit Mol (with conformer) to feature + pos tensors."""
  if mol is None:
    return None
  try:
    feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)
    if mol.GetNumConformers() == 0:
      return None
    conf = mol.GetConformer()
    pos = torch.tensor(
      [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
      dtype=torch.float,
    )
    return x, pos
  except Exception as exc:
    logger.warning("Featurisation failed: %s", exc)
    return None


def load_model(
  model_path: str,
  atom_feat_dim: int = 43,
  hidden_dim: int = 128,
  n_layers: int = 4,
  dropout: float = 0.1,
  device: torch.device | None = None,
) -> GNNPredictor:
  """Load a GNNPredictor from *model_path*.

  Auto-detects whether the checkpoint was trained with a protein
  encoder by inspecting the state dict keys.
  """
  if device is None:
    device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu",
    )

  # Peek at checkpoint to detect protein encoder weights
  use_protein = False
  if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device)
    use_protein = any(k.startswith("protein_encoder.") for k in state)
    if use_protein:
      logger.info(
        "Detected protein encoder weights in %s",
        model_path,
      )
  else:
    state = None

  model = GNNPredictor(
    atom_feat_dim=atom_feat_dim,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    dropout=dropout,
    use_protein_encoder=use_protein,
  ).to(device)

  if state is not None:
    model.load_state_dict(state, strict=False)
    logger.info("Loaded model from %s", model_path)
  model.eval()
  return model


def score_molecule(
  mol: Chem.Mol,
  model: GNNPredictor,
  mc_samples: int = 30,
  device: torch.device | None = None,
  pdb_id: str | None = None,
) -> dict[str, Any] | None:
  """Score a single molecule; returns pKa mean/std or None."""
  if device is None:
    device = next(model.parameters()).device
  tensors = _mol_to_tensors(mol)
  if tensors is None:
    return None
  x, pos = tensors
  x, pos = x.to(device), pos.to(device)
  batch = torch.zeros(
    x.size(0),
    dtype=torch.long,
    device=device,
  )

  protein_embs = None
  if pdb_id:
    if pdb_id in _PROTEIN_EMB_CACHE:
      emb = _PROTEIN_EMB_CACHE[pdb_id]
      protein_embs = [emb.to(device)]
    else:
      try:
        pdb_path = fetch_pdb(pdb_id, cache_dir="data/raw/pdbs")
        if pdb_path:
          seq = extract_pocket_seq(pdb_path)
          if seq:
            emb = precompute_esm2_embedding(seq, cache_key=pdb_id)
            _PROTEIN_EMB_CACHE[pdb_id] = emb
            protein_embs = [emb.to(device)]
      except Exception:
        pass

  mean, std = model.predict_with_uncertainty(
    x,
    pos,
    batch,
    protein_embs=protein_embs,
    n_samples=mc_samples,
  )
  return {
    "pka_mean": float(mean.item()),
    "pka_std": float(std.item()),
  }
