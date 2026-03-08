"""GCN baseline model for binding affinity prediction."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNBaseline(torch.nn.Module):
  """Simple 3-layer GCN baseline."""

  def __init__(self, num_node_features: int, hidden_channels: int = 64):
    super().__init__()
    torch.manual_seed(12345)
    self.conv1 = GCNConv(num_node_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.conv3 = GCNConv(hidden_channels, hidden_channels)
    self.lin = torch.nn.Linear(hidden_channels, 1)

  def forward(self, x, edge_index, batch):
    x = self.conv1(x, edge_index).relu()
    x = self.conv2(x, edge_index).relu()
    x = self.conv3(x, edge_index)
    x = global_mean_pool(x, batch)
    x = F.dropout(x, p=0.5, training=self.training)
    return self.lin(x)
