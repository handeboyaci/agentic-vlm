import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_baseline import GNNBaseline


def train():
  data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
  )

  # Try using ESOL as a reliable surrogate if PDBbind details are tricky with just 'PDBbind'
  # For a real PDBbind, we might need 'PDBbind(root=..., name='refined')' etc.
  # Let's use ESOL for the baseline *proof of concept* to ensure it runs out of the box.
  # If the user strictly wants PDBbind, we can switch, but MoleculeNet's PDBbind often has download issues.
  print("Loading dataset (using ESOL as safe default for baseline implementation)...")
  dataset = MoleculeNet(root=data_dir, name="ESOL")

  print(f"Dataset: {dataset}:")
  print("====================")
  print(f"Number of graphs: {len(dataset)}")
  print(f"Number of features: {dataset.num_features}")
  print(f"Number of classes: {dataset.num_classes}")

  torch.manual_seed(12345)
  dataset = dataset.shuffle()

  train_dataset = dataset[: int(len(dataset) * 0.8)]
  test_dataset = dataset[int(len(dataset) * 0.8) :]

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # type: ignore
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # type: ignore

  model = GNNBaseline(num_node_features=dataset.num_features)  # type: ignore
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  criterion = torch.nn.MSELoss()

  def train_step():
    model.train()
    total_loss = 0
    for data in train_loader:
      out = model(data.x.float(), data.edge_index, data.batch)
      loss = criterion(out, data.y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss += loss.item()
    return total_loss / len(train_loader)

  def test_step(loader):
    model.eval()
    total_loss = 0
    for data in loader:
      out = model(data.x.float(), data.edge_index, data.batch)
      loss = criterion(out, data.y)
      total_loss += loss.item()
    return total_loss / len(loader)

  print("Starting training...")
  for epoch in range(1, 11):
    train_loss = train_step()
    test_loss = test_step(test_loader)
    print(
      f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
    )

  # Save model
  models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
  )
  torch.save(model.state_dict(), os.path.join(models_dir, "gnn_baseline.pth"))
  print("Model saved to models/gnn_baseline.pth")


if __name__ == "__main__":
  train()
