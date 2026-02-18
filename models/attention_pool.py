import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
  """Attention-weighted graph pooling."""
  def __init__(self, in_dim, hidden_dim):
    super().__init__()
    # Adding gate to match state_dict
    self.gate = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, 1)
    )
    
  def forward(self, h, batch):
    weights = self.gate(h)
    max_batch = int(batch.max().item()) + 1
    exp_w = torch.exp(weights - weights.max())
    sum_exp = torch.zeros(max_batch, 1, device=h.device)
    sum_exp.index_add_(0, batch, exp_w)
    alpha = exp_w / (sum_exp[batch] + 1e-6)
    weighted_h = h * alpha
    graph_h = torch.zeros(max_batch, h.size(1), device=h.device)
    graph_h.index_add_(0, batch, weighted_h)
    return graph_h
