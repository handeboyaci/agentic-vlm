"""Evaluate the trained GNN predictor on the LP-PDBBind test set."""

import os
import torch
import numpy as np
import scipy.stats as stats
from torch_geometric.loader import DataLoader

from data.lp_pdbbind import LPPDBBind
from models.gnn_predictor import GNNPredictor


def evaluate_test_set():
  print("Loading test dataset (this will use the cached PyG graphs if available)...")
  test_dataset = LPPDBBind(root="data/pdbbind_deepchem", split="test")
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  print(f"Loaded {len(test_dataset)} test complexes.")

  device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
  )
  print(f"Evaluating on {device}")

  model_path = "models/gnn_predictor.pth"
  if not os.path.exists(model_path):
    print(f"Error: Could not find {model_path}.")
    print(
      "Please download the trained weights from Colab and place them in the models/ directory."
    )
    return

  # Initialize model with matching architecture
  model = GNNPredictor(
    atom_feat_dim=43,
    hidden_dim=128,
    n_layers=4,
    dropout=0.0,  # Turn off dropout for pure inference
  ).to(device)

  # Load weights
  state = torch.load(model_path, map_location=device, weights_only=True)
  model.load_state_dict(state, strict=False)
  model.eval()

  all_preds = []
  all_targets = []

  print("Running inference...")
  with torch.no_grad():
    for data in test_loader:
      data = data.to(device)
      # The model forward pass expects: x, pos, batch, protein_embs=None
      out = model(data.x.float(), data.pos.float(), data.batch)

      all_preds.extend(out.cpu().squeeze().tolist())
      all_targets.extend(data.y.cpu().squeeze().tolist())

  preds = np.array(all_preds)
  targets = np.array(all_targets)

  # Compute metrics
  mse = np.mean((preds - targets) ** 2)
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(preds - targets))

  # Pearson and Spearman correlation
  if len(preds) > 1:
    pearson_r, _ = stats.pearsonr(preds, targets)
    spearman_r, _ = stats.spearmanr(preds, targets)
  else:
    pearson_r, spearman_r = 0.0, 0.0

  print("\n" + "=" * 40)
  print("FINAL TEST SET METRICS (EGNN)")
  print("=" * 40)
  print(f"RMSE:          {rmse:.4f} pKa units")
  print(f"MAE:           {mae:.4f} pKa units")
  print(f"Pearson r:     {pearson_r:.4f}")
  print(f"Spearman rho:  {spearman_r:.4f}")
  print("=" * 40)


if __name__ == "__main__":
  evaluate_test_set()
