import torch
import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader
from data.lp_pdbbind import LPPDBBind
from models.gnn_predictor import GNNPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/pdbbind_deepchem"
    )

    # Use the real LP-PDBBind dataset with ESM-2 precomputation
    logger.info("Loading LP-PDBBind dataset (this may take time if ESM-2 is precomputing)...")
    dataset = LPPDBBind(
        root=data_dir, 
        split="train", 
        precompute_esm=True
    )

    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.9)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the Protein-Aware GNN
    model = GNNPredictor(
        atom_feat_dim=43,
        hidden_dim=128,
        n_layers=4,
        use_protein_encoder=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    def get_protein_embs(data):
        if hasattr(data, "protein_emb") and data.protein_emb is not None:
            # For simplicity in local training script, we handle single-graph or batch
            return [data.protein_emb.to(device)]
        return None

    def train_step():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            protein_embs = get_protein_embs(data)
            out = model(data.x.float(), data.pos.float(), data.batch, protein_embs=protein_embs)
            
            loss = criterion(out.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_dataset)

    logger.info("Starting training...")
    for epoch in range(1, 11):
        loss = train_step()
        logger.info(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    # Save model
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    torch.save(model.state_dict(), os.path.join(models_dir, "gnn_predictor_local.pth"))
    logger.info("Model saved to models/gnn_predictor_local.pth")

if __name__ == "__main__":
    train()
