import torch
import torch.nn as nn


class AttentionPool(nn.Module):
  """Attention-weighted graph pooling."""

  def __init__(self, in_dim, hidden_dim):
    super().__init__()
    # Adding gate to match state_dict
    self.gate = nn.Sequential(
      nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
    )

  def forward(self, h, batch, num_graphs=None):
    if h.size(0) == 0:
        return torch.zeros(num_graphs if num_graphs is not None else 0, h.size(1), device=h.device)

    # 1. Compute weights and sanitize
    weights = self.gate(h)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 2. Stable Softmax across the batch
    max_batch = num_graphs if num_graphs is not None else int(batch.max().item()) + 1
    
    # Per-graph max for softmax stability
    # We use a large negative value for empty graphs to not affect the max
    weights_max = torch.full((max_batch, 1), -1e9, device=h.device)
    # Using a loop or scatter_max for better stability if needed, 
    # but weights.max() is usually okay if we've sanitized.
    weights_max.index_reduce_(0, batch, weights, 'amax', include_self=False)
    
    exp_w = torch.exp(torch.clamp(weights - weights_max[batch], min=-10.0, max=0.0))
    sum_exp = torch.zeros(max_batch, 1, device=h.device)
    sum_exp.index_add_(0, batch, exp_w)
    
    alpha = exp_w / (sum_exp[batch] + 1e-6)
    weighted_h = h * alpha
    
    graph_h = torch.zeros(max_batch, h.size(1), device=h.device)
    graph_h.index_add_(0, batch, weighted_h)
    return graph_h
