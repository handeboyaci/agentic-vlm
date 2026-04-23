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
  attention pooling, and protein pocket awareness.
  """

  def __init__(
    self,
    atom_feat_dim: int = 44,
    hidden_dim: int = 128,
    n_layers: int = 4,
    edge_dim: int = 16,
    dropout: float = 0.1,
    use_protein_encoder: bool = False,
  ):
    super().__init__()
    self.use_protein_encoder = use_protein_encoder
    self.dropout_rate = dropout
    self.hidden_dim = hidden_dim

    # Projection for atom features (shared by ligand and protein)
    self.input_proj = nn.Sequential(
      nn.Linear(atom_feat_dim, hidden_dim),
      nn.SiLU(),
    )

    # Projection for ESM-2 protein embeddings (1280 -> hidden_dim)
    self.protein_proj = nn.Sequential(
      nn.Linear(1280, hidden_dim),
      nn.SiLU(),
      nn.LayerNorm(hidden_dim), # Normalize projected ESM-2 signals
    )

    self.edge_builder = MultiScaleEdgeBuilder(edge_dim=edge_dim)
    self.egnn_layers = nn.ModuleList(
      [
        EGNNLayer(
          node_dim=hidden_dim,
          edge_dim=edge_dim,
          hidden_dim=hidden_dim,
        )
        for _ in range(n_layers)
      ]
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

  def forward(self, x, pos, batch, ligand_mask=None, protein_res_idx=None, protein_embs=None):
    """
    Args:
        x: Atom features (N_total, 42)
        pos: Atom coordinates (N_total, 3)
        batch: Batch assignment (N_total)
        ligand_mask: Boolean mask for ligand atoms (N_total)
        protein_res_idx: Residue indices for protein atoms (N_total)
        protein_embs: List of ESM-2 embeddings [(L_i, 1280)]
    """
    # 0. Sanitize inputs
    x = torch.nan_to_num(x, nan=0.0)
    pos = torch.nan_to_num(pos, nan=0.0)
    # Clamp extreme positions (usually errors from RDKit conformer generation)
    pos = torch.clamp(pos, min=-1000.0, max=1000.0)
    
    # 1. Project base atom features
    h = self.input_proj(x)
    
    # 2. Inject ESM-2 embeddings into protein nodes
    if self.use_protein_encoder and protein_embs is not None and ligand_mask is not None and protein_res_idx is not None:
        # ligand_mask is True for ligand, False for protein
        prot_mask = ~ligand_mask
        if prot_mask.any():
            # Project all ESM-2 sequences in the batch
            # protein_embs is a list of [L_i, 1280]
            unique_batches = torch.unique(batch)
            for i, b_id in enumerate(unique_batches):
                # Mask for protein atoms in this specific graph
                graph_prot_mask = prot_mask & (batch == b_id)
                if not graph_prot_mask.any() or i >= len(protein_embs):
                    continue
                
                # Project all ESM-2 sequences in the batch
                p_raw = protein_embs[i].to(h.device)
                p_raw = torch.nan_to_num(p_raw, nan=0.0) # Sanitize protein embs
                
                # Get projected ESM-2 for this protein
                p_emb = self.protein_proj(p_raw) # (L_i, hidden)
                
                # Map residue embeddings to atoms
                indices = protein_res_idx[graph_prot_mask]
                
                # Safeguard: ESM-2 may truncate to 1024 tokens, ensure indices are in range
                max_idx = p_emb.shape[0] - 1
                indices = torch.clamp(indices, 0, max_idx)
                
                # Add the ESM-2 signal to the atom features
                h[graph_prot_mask] = h[graph_prot_mask] + p_emb[indices]

    # 3. Message Passing on the Unified 3D Graph
    edge_index, edge_attr = self.edge_builder(pos, batch)
    for layer in self.egnn_layers:
      h, pos = layer(h, pos, edge_index, edge_attr)
      
    # 4. Global Pooling (only over ligand atoms to focus the affinity prediction)
    num_graphs = int(batch.max().item()) + 1
    if ligand_mask is not None:
        # We only pool the ligand atoms for the final score
        # but the protein atoms helped shape the features during EGNN layers
        h_lig = h[ligand_mask]
        batch_lig = batch[ligand_mask]
        graph_emb = self.pool(h_lig, batch_lig, num_graphs=num_graphs)
    else:
        graph_emb = self.pool(h, batch, num_graphs=num_graphs)
        
    out = self.head(graph_emb)
    return torch.clamp(out, min=-20.0, max=20.0) # Clamp final pKa output

  def predict_with_uncertainty(
    self,
    x,
    pos,
    batch,
    ligand_mask=None,
    protein_res_idx=None,
    protein_embs=None,
    n_samples: int = 30,
  ):
    """MC Dropout uncertainty estimation."""
    self.train()  # enable dropout
    preds = []
    with torch.no_grad():
      for _ in range(n_samples):
        pred = self.forward(x, pos, batch, ligand_mask, protein_res_idx, protein_embs)
        preds.append(pred)
    preds = torch.stack(preds, dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    self.eval()
    return mean, std
