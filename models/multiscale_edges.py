import torch
import torch.nn as nn


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
    num_nodes = pos.size(0)

    # Build pairwise indices efficiently with cutoff radius
    row_all = torch.arange(num_nodes, device=pos.device)
    col_all = torch.arange(num_nodes, device=pos.device)
    row = row_all.repeat_interleave(num_nodes)
    col = col_all.repeat(num_nodes)

    # Remove self-loops
    mask = row != col
    row, col = row[mask], col[mask]

    # Keep only edges within same graph
    batch_mask = batch[row] == batch[col]
    row, col = row[batch_mask], col[batch_mask]

    # Compute distances and apply cutoff
    diff = pos[row] - pos[col]
    dist = torch.norm(diff + 1e-10, dim=-1)
    cutoff_mask = dist < self.cutoff
    row, col = row[cutoff_mask], col[cutoff_mask]
    dist = dist[cutoff_mask]

    edge_index = torch.stack([row, col], dim=0)

    # RBF expansion: Gaussian basis functions
    # Shape: [num_edges, num_rbf]
    rbf = torch.exp(
      -((dist.unsqueeze(-1) - self.centers) ** 2)
      / (2 * self.width**2)
    )

    return edge_index, rbf
