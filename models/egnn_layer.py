import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
  """E(n) Equivariant Graph Neural Network Layer with Stability Fixes."""

  def __init__(self, node_dim, edge_dim, hidden_dim):
    super().__init__()
    self.node_dim = node_dim
    self.edge_dim = edge_dim
    self.hidden_dim = hidden_dim

    # msg_mlp
    self.msg_mlp = nn.Sequential(
      nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
    )

    # node_mlp
    self.node_mlp = nn.Sequential(
      nn.Linear(node_dim + hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, node_dim),
    )

    # coord_mlp: Add Tanh for coordinate stability
    self.coord_mlp = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, 1, bias=False),
      nn.Tanh() # Prevents explosive coordinate updates
    )

    self.node_norm = nn.LayerNorm(node_dim)

  def forward(self, h, x, edge_index, edge_attr):
    row, col = edge_index
    
    # 1. Distance with epsilon
    diff = x[row] - x[col]
    dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
    dist = torch.sqrt(dist_sq + 1e-8)
    
    # 2. Messages - Use log1p for distance stability in large protein graphs
    edge_input = torch.cat([h[row], h[col], torch.log1p(dist_sq), edge_attr], dim=-1)
    m_ij = self.msg_mlp(edge_input)

    # 3. Message Aggregation with Normalization
    # We count neighbors to prevent sum-explosion in large protein graphs
    m_i = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
    m_i.index_add_(0, row, m_ij)
    
    # Normalize by number of neighbors for stability
    counts = torch.zeros(h.size(0), 1, device=h.device)
    counts.index_add_(0, row, torch.ones_like(dist_sq))
    m_i = m_i / (counts + 1e-8)

    # 4. Coordinate Update (Equivariant)
    # Stability fix: use unit radial vector to prevent explosive updates
    # and add a small scaling factor (0.1)
    radial = diff / (dist + 1e-8)
    trans = radial * self.coord_mlp(m_ij) * 0.1
    x_agg = torch.zeros_like(x)
    x_agg.index_add_(0, row, trans)
    x_agg = x_agg / (counts + 1e-8) # Normalize coordinate push
    x = x + x_agg

    # 5. Node Update
    h_input = torch.cat([h, m_i], dim=-1)
    h = h + self.node_mlp(h_input)
    h = self.node_norm(h)

    return h, x
