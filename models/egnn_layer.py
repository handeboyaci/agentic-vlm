import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
  """E(n) Equivariant Graph Neural Network Layer.

  Implementation matching the saved weights in gnn_predictor.pth.
  """

  def __init__(self, node_dim, edge_dim, hidden_dim):
    super().__init__()
    self.node_dim = node_dim
    self.edge_dim = edge_dim
    self.hidden_dim = hidden_dim

    # msg_mlp instead of edge_mlp
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

    # coord_mlp
    self.coord_mlp = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, 1, bias=False),
    )

    # Adding node_norm
    self.node_norm = nn.LayerNorm(node_dim)

  def forward(self, h, x, edge_index, edge_attr):
    row, col = edge_index
    dist_sq = torch.sum((x[row] - x[col]) ** 2, dim=-1, keepdim=True)
    # Add epsilon for numerical stability during backprop
    dist_sq = dist_sq + 1e-8
    edge_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
    m_ij = self.msg_mlp(edge_input)

    m_i = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
    m_i.index_add_(0, row, m_ij)

    trans = (x[row] - x[col]) * self.coord_mlp(m_ij)
    x_agg = torch.zeros_like(x)
    x_agg.index_add_(0, row, trans)
    x = x + x_agg

    h_input = torch.cat([h, m_i], dim=-1)
    h = h + self.node_mlp(h_input)
    h = self.node_norm(h)

    return h, x
