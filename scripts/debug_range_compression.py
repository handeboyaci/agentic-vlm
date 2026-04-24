
import os
import torch
import numpy as np
import scipy.stats as stats
from torch_geometric.loader import DataLoader
from data.lp_pdbbind import LPPDBBind
from models.gnn_predictor import GNNPredictor

def _get_graph_inputs(data):
    ligand_mask = getattr(data, 'ligand_mask', None)
    protein_res_idx = getattr(data, 'protein_res_idx', None)
    if not hasattr(data, 'protein_emb') or data.protein_emb is None: return ligand_mask, protein_res_idx, None
    try:
        batch_ids = data.batch.unique()
        embs = [data.protein_emb[b_id].to(data.x.device) for b_id in batch_ids]
        return ligand_mask, protein_res_idx, embs
    except Exception: return ligand_mask, protein_res_idx, [data.protein_emb.to(data.x.device)]

def analyze_performance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use the specific root from the training run
    root = 'data/pdbbind_all_atom'
    print(f"Loading test dataset from {root}...")
    test_dataset = LPPDBBind(root=root, split='test', precompute_esm=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the best model
    model = GNNPredictor(
        atom_feat_dim=44, 
        hidden_dim=256, 
        n_layers=6,      
        dropout=0.0,
        use_protein_encoder=True,
    ).to(device)

    checkpoint_path = 'best_model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Checkpoint not found!")
        return

    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            l_mask, p_idx, p_embs = _get_graph_inputs(data)
            out = model(data.x.float(), data.pos.float(), data.batch, 
                        ligand_mask=l_mask, protein_res_idx=p_idx, protein_embs=p_embs)
            all_preds.extend(out.cpu().squeeze().tolist())
            all_targets.extend(data.y.cpu().squeeze().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # 1. Global Metrics
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    pearson_r, _ = stats.pearsonr(preds, targets)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson R: {pearson_r:.4f}")

    # 2. Range Analysis
    print(f"\nRANGE ANALYSIS:")
    bins = [(0, 4), (4, 6), (6, 8), (8, 10), (10, 15)]
    for low, high in bins:
        mask = (targets >= low) & (targets < high)
        if mask.any():
            r = np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))
            bias = np.mean(preds[mask] - targets[mask])
            print(f"Range [{low:2d}, {high:2d}]: RMSE={r:.4f}, Bias={bias:+.4f}, Count={mask.sum()}")

    # 3. Range Compression Check
    target_std = np.std(targets)
    pred_std = np.std(preds)
    print(f"\nRANGE COMPRESSION:")
    print(f"Target StdDev: {target_std:.4f}")
    print(f"Pred StdDev:   {pred_std:.4f}")
    print(f"Compression Ratio: {pred_std/target_std:.4f} (closer to 1.0 is better)")
    
    # 4. Save results for plotting if needed
    np.savez('debug_metrics.npz', preds=preds, targets=targets)
    print("\nSaved predictions and targets to debug_metrics.npz")

if __name__ == "__main__":
    analyze_performance()
