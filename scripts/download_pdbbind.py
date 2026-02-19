import os
import torch
from torch_geometric.datasets import MoleculeNet

def download_pdbbind():
    """
    Downloads the PDBbind dataset (part of MoleculeNet) to the data directory.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    print(f"Downloading PDBbind dataset to {data_dir}...")
    
    # MoleculeNet can download PDBbind. 
    # Note: PDBbind in MoleculeNet might be the refined set or core set.
    # We will use the default settings for now.
    try:
        dataset = MoleculeNet(root=data_dir, name='PDBbind', pre_transform=None, pre_filter=None)
        print(f"Successfully downloaded PDBbind dataset.")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        
    except Exception as e:
        print(f"Error downloading PDBbind: {e}")
        print("Attempting to use ESOL as a surrogate if PDBbind fails (PDBbind sometimes requires login or has changed URLs).")
        try:
            dataset = MoleculeNet(root=data_dir, name='ESOL')
            print(f"Successfully downloaded ESOL dataset as surrogate.")
            print(f"Number of graphs: {len(dataset)}")
        except Exception as e2:
            print(f"Critical error downloading datasets: {e2}")

if __name__ == "__main__":
    download_pdbbind()
