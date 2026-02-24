import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinEncoder(nn.Module):
  """Cross-attention between ligand nodes and ESM-2 protein embeddings."""
  def __init__(self, ligand_dim=128, protein_dim=1280):
    super().__init__()
    self.k_proj = nn.Linear(protein_dim, ligand_dim)
    self.v_proj = nn.Linear(protein_dim, ligand_dim)
    self.q_proj = nn.Linear(ligand_dim, ligand_dim)
    
  def forward(self, h_lig, protein_embs, batch):
    # protein_embs: list [ (L_i, 1280) ]
    # ligand_dim: (N_total, 128)
    
    h_out = []
    unique_batches = torch.unique(batch)
    
    for i, b_id in enumerate(unique_batches):
      idx = (batch == b_id)
      h_b = h_lig[idx] # (n_atoms, 128)
      p_e = protein_embs[i] # (n_residues, 1280)
      
      q = self.q_proj(h_b) # (n, 128)
      k = self.k_proj(p_e) # (m, 128)
      v = self.v_proj(p_e) # (m, 128)
      
      # Attention
      scores = torch.matmul(q, k.transpose(0, 1)) / (128**0.5)
      attn = F.softmax(scores, dim=-1)
      context = torch.matmul(attn, v) # (n, 128)
      
      h_out.append(h_b + context)
      
    return torch.cat(h_out, dim=0)

def precompute_esm2_embedding(sequence: str) -> torch.Tensor:
  """Mock ESM-2 embedding for demonstration if models not available."""
  # In production, this would use a real ESM-2 model
  # Returns a (len(sequence), 1280) tensor
  return torch.randn(len(sequence), 1280)
