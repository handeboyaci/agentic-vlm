
import torch
from data.lp_pdbbind import LPPDBBind
from torch_geometric.loader import DataLoader
import numpy as np

def diagnose():
    root = 'data/pdbbind_all_atom'
    print(f"Loading test dataset from {root}...")
    try:
        dataset = LPPDBBind(root=root, split='test')
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    count = 0
    for data in loader:
        has_nan = False
        if torch.isnan(data.x).any():
            print(f"NaN in x (features) for PDB: {data.pdb_id}")
            has_nan = True
        if torch.isnan(data.pos).any():
            print(f"NaN in pos (coordinates) for PDB: {data.pdb_id}")
            has_nan = True
        if torch.isnan(data.y).any():
            print(f"NaN in y (target) for PDB: {data.pdb_id}")
            has_nan = True
        
        if has_nan:
            print("Sample x first row:", data.x[0])
            print("Sample pos first row:", data.pos[0])
            break
            
        count += 1
        if count > 100:
            print("Checked 100 samples, no NaNs found in raw data.")
            # Check for infinity or extreme values
            max_x = torch.max(torch.abs(data.x)).item()
            max_pos = torch.max(torch.abs(data.pos)).item()
            print(f"Max absolute x: {max_x:.4f}")
            print(f"Max absolute pos: {max_pos:.4f}")
            break

if __name__ == "__main__":
    diagnose()
