"""ESM-2 protein encoder for ligand-protein cross-attention.

Provides:
  - ``ProteinEncoder``: cross-attention between ligand EGNN nodes
    and per-residue ESM-2 embeddings.
  - ``precompute_esm2_embedding()``: runs ESM-2 inference on a
    pocket amino-acid sequence, with disk caching.

Model: ``facebook/esm2_t33_650M_UR50D`` (650M params, 1280-dim
output).  Requires ``transformers`` to be installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Global cache (in-memory + disk) ────────────────────────────
_MODEL = None
_TOKENIZER = None
_EMB_CACHE: dict[str, torch.Tensor] = {}

ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 1280-dim
ESM_CACHE_DIR = "data/esm_cache"


def _load_esm2():
  """Lazily load the ESM-2 model + tokenizer."""
  global _MODEL, _TOKENIZER  # noqa: PLW0603
  if _MODEL is not None:
    return _MODEL, _TOKENIZER

  try:
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading ESM-2 model: %s", ESM2_MODEL_NAME)
    _TOKENIZER = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(ESM2_MODEL_NAME)
    _MODEL.eval()

    # Move to GPU if available (Colab A100)
    if torch.cuda.is_available():
      _MODEL = _MODEL.cuda()
      logger.info("ESM-2 loaded on CUDA")
    else:
      logger.info("ESM-2 loaded on CPU")

    return _MODEL, _TOKENIZER
  except ImportError:
    logger.warning(
      "transformers not installed — ESM-2 unavailable. Run: pip install transformers"
    )
    return None, None
  except Exception as exc:
    logger.warning("Failed to load ESM-2: %s", exc)
    return None, None


# ── Cross-attention module ──────────────────────────────────────


class ProteinEncoder(nn.Module):
  """Cross-attention between ligand nodes and ESM-2 embeddings.

  Given per-atom ligand features ``h_lig`` (N, ligand_dim) and
  per-residue ESM-2 embeddings ``protein_embs`` list[(L_i, 1280)],
  computes scaled dot-product cross-attention and adds the context
  to the ligand features as a residual connection.
  """

  def __init__(self, ligand_dim: int = 128, protein_dim: int = 1280):
    super().__init__()
    self.k_proj = nn.Linear(protein_dim, ligand_dim)
    self.v_proj = nn.Linear(protein_dim, ligand_dim)
    self.q_proj = nn.Linear(ligand_dim, ligand_dim)
    self.scale = ligand_dim**0.5

  def forward(self, h_lig, protein_embs, batch):
    """Apply cross-attention per graph in the batch.

    Args:
      h_lig: Ligand node features (N_total, ligand_dim).
      protein_embs: List of protein embeddings, one per graph.
      batch: Batch assignment vector (N_total,).

    Returns:
      Updated ligand features with protein context added.
    """
    h_out = []
    unique_batches = torch.unique(batch)

    for i, b_id in enumerate(unique_batches):
      idx = batch == b_id
      h_b = h_lig[idx]  # (n_atoms, ligand_dim)

      if i >= len(protein_embs):
        h_out.append(h_b)
        continue

      p_e = protein_embs[i]  # (n_residues, 1280)
      if p_e.device != h_b.device:
        p_e = p_e.to(h_b.device)

      q = self.q_proj(h_b)  # (n, D)
      k = self.k_proj(p_e)  # (m, D)
      v = self.v_proj(p_e)  # (m, D)

      scores = torch.matmul(q, k.transpose(0, 1)) / self.scale
      attn = F.softmax(scores, dim=-1)
      context = torch.matmul(attn, v)  # (n, D)

      h_out.append(h_b + context)

    return torch.cat(h_out, dim=0)


# ── ESM-2 embedding computation ────────────────────────────────


def precompute_esm2_embedding(
  sequence: str,
  cache_key: str | None = None,
  cache_dir: str = ESM_CACHE_DIR,
) -> torch.Tensor:
  """Compute per-residue ESM-2 embeddings for a protein sequence.

  Args:
    sequence: Amino acid sequence (e.g. "ACDEFG...").
    cache_key: Optional key for disk caching (e.g. PDB ID).
    cache_dir: Directory for cached .pt files.

  Returns:
    Tensor of shape (L, 1280) where L = len(sequence).
  """
  # 1. Check in-memory cache
  key = cache_key or sequence[:50]
  if key in _EMB_CACHE:
    return _EMB_CACHE[key]

  # 2. Check disk cache
  if cache_key:
    cache_path = Path(cache_dir) / f"{cache_key}.pt"
    if cache_path.exists():
      emb = torch.load(cache_path, map_location="cpu")
      _EMB_CACHE[key] = emb
      return emb

  # 3. Compute with real ESM-2
  model, tokenizer = _load_esm2()

  if model is not None and tokenizer is not None:
    device = next(model.parameters()).device
    inputs = tokenizer(
      sequence,
      return_tensors="pt",
      truncation=True,
      max_length=1024,
    ).to(device)

    with torch.no_grad():
      outputs = model(**inputs)
      # last_hidden_state: (1, L+2, 1280), strip BOS/EOS
      emb = outputs.last_hidden_state[0, 1:-1, :].cpu()

    _EMB_CACHE[key] = emb

    # Save to disk
    if cache_key:
      os.makedirs(cache_dir, exist_ok=True)
      torch.save(emb, Path(cache_dir) / f"{cache_key}.pt")
      logger.info(
        "Cached ESM-2 embedding for %s: shape %s",
        cache_key,
        emb.shape,
      )

    return emb

  # 4. Fallback: mock (random noise) with warning
  logger.warning(
    "Using MOCK ESM-2 embeddings (random noise). "
    "Install transformers for real protein context: "
    "pip install transformers"
  )
  emb = torch.randn(len(sequence), 1280)
  _EMB_CACHE[key] = emb
  return emb
