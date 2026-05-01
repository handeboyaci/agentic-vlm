import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph


class MultiScaleEdgeBuilder(nn.Module):
  """Builds multi-scale edges with RBF-encoded distance features."""

  def __init__(self, edge_dim=16, cutoff=8.0, num_rbf=16):
    super().__init__()
    self.cutoff = cutoff
    self.num_rbf = num_rbf
    self.edge_dim = edge_dim

    # RBF centers evenly spaced from 0 to cutoff
    centers = torch.linspace(0.0, cutoff, num_rbf)
    self.register_buffer("centers", centers)
    width = (cutoff / num_rbf) * 0.5
    self.register_buffer("width", torch.tensor(width))

  def forward(self, pos, batch):
    # Sparse radius graph: much more memory efficient than N x N
    edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
    row, col = edge_index[0], edge_index[1]

    # Compute distances for sparse edges
    diff = pos[row] - pos[col]
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-10)

    # RBF expansion: Gaussian basis functions
    # Shape: [num_edges, num_rbf]
    rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width**2))

    return edge_index, rbf
