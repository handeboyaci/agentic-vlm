"""SOTA binding affinity predictor with E(n) equivariant GNN."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.egnn_layer import EGNNLayer
from models.multiscale_edges import MultiScaleEdgeBuilder
from models.attention_pool import AttentionPool


class GNNPredictor(nn.Module):
  """Binding affinity predictor.

  Combines an E(n) Equivariant GNN with multiscale edge features,
  attention pooling, and optional ESM-2 protein cross-attention.
  Uses MC Dropout for uncertainty estimation.
  """

  def __init__(
    self,
    atom_feat_dim: int = 43,
    hidden_dim: int = 128,
    n_layers: int = 4,
    edge_dim: int = 16,
    dropout: float = 0.1,
    use_protein_encoder: bool = False,
  ):
    super().__init__()
    self.use_protein_encoder = use_protein_encoder
    self.dropout_rate = dropout

    self.input_proj = nn.Sequential(
      nn.Linear(atom_feat_dim, hidden_dim),
      nn.SiLU(),
    )

    self.edge_builder = MultiScaleEdgeBuilder(edge_dim=edge_dim)
    self.egnn_layers = nn.ModuleList([
      EGNNLayer(
        node_dim=hidden_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
      )
      for _ in range(n_layers)
    ])

    self.protein_encoder = None
    if use_protein_encoder:
      from models.protein_encoder import ProteinEncoder

      self.protein_encoder = ProteinEncoder(
        ligand_dim=hidden_dim,
      )

    self.pool = AttentionPool(
      in_dim=hidden_dim,
      hidden_dim=hidden_dim,
    )

    self.head = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.SiLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim // 2, 1),
    )

  def forward(self, x, pos, batch, protein_embs=None):
    h = self.input_proj(x)
    edge_index, edge_attr = self.edge_builder(pos, batch)
    for layer in self.egnn_layers:
      h, pos = layer(h, pos, edge_index, edge_attr)
    if (
      self.protein_encoder is not None
      and protein_embs is not None
    ):
      h = self.protein_encoder(h, protein_embs, batch)
    graph_emb = self.pool(h, batch)
    return self.head(graph_emb)

  def predict_with_uncertainty(
    self,
    x,
    pos,
    batch,
    protein_embs=None,
    n_samples: int = 30,
  ):
    """MC Dropout uncertainty estimation."""
    self.train()  # enable dropout
    preds = []
    with torch.no_grad():
      for _ in range(n_samples):
        pred = self.forward(x, pos, batch, protein_embs)
        preds.append(pred)
    preds = torch.stack(preds, dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    self.eval()
    return mean, std
