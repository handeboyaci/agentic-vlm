from __future__ import annotations
import os
import logging
import urllib.request
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

CSV_URL = (
  "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv"
)
ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]

def one_hot(val, choices):
  vec = [0] * (len(choices) + 1)
  if val in choices:
    vec[choices.index(val)] = 1
  else:
    vec[-1] = 1
  return vec

def atom_features(atom: Chem.Atom) -> list[float]:
  p_charge = 0.0
  if atom.HasProp("_GasteigerCharge"):
      try:
          val = atom.GetProp("_GasteigerCharge")
          # Handle NaN or empty strings
          if val and val.lower() != "nan":
              p_charge = float(val)
          if np.isnan(p_charge) or np.isinf(p_charge):
              p_charge = 0.0
      except:
          p_charge = 0.0

  return (
    one_hot(atom.GetSymbol(), ATOM_TYPES)
    + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    + one_hot(
      atom.GetHybridization(),
      [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
      ],
    )
    + one_hot(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
    + [
      atom.GetFormalCharge(),
      atom.GetNumRadicalElectrons(),
      int(atom.GetIsAromatic()),
      int(atom.IsInRing()),
      atom.GetMass() / 100.0,
      p_charge
    ]
    + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
  )  # 44-dim

def get_pocket_atoms(pdb_id: str, refined_dir: str = "data/refined-set") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pdb_path = os.path.join(refined_dir, pdb_id, f"{pdb_id}_pocket.pdb")
    if not os.path.exists(pdb_path):
        return None
    
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
    if mol is None: return None
    
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)
        mol = Chem.RemoveHs(mol)
    except:
        return None
        
    if mol.GetNumConformers() == 0: return None
    
    feats = [atom_features(a) for a in mol.GetAtoms()]
    conf = mol.GetConformer()
    pos = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    
    res_indices = []
    res_counter = -1
    last_res_info = None
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info:
            res_info = (info.GetResidueNumber(), info.GetResidueName(), info.GetChainId())
            if res_info != last_res_info:
                res_counter += 1
                last_res_info = res_info
        res_indices.append(max(0, res_counter))
        
    return (
        torch.tensor(feats, dtype=torch.float),
        torch.tensor(pos, dtype=torch.float),
        torch.tensor(res_indices, dtype=torch.long)
    )

def process_single_complex(pdb_id_str, smiles, value, seq):
    """Worker function for parallel processing."""
    pocket_data = get_pocket_atoms(pdb_id_str)
    seq_len = len(seq)
    
    # 1. Ligand Prep
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return []
    mol = Chem.AddHs(mol)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass
    l_feats = [atom_features(a) for a in mol.GetAtoms()]
    l_x = torch.tensor(l_feats, dtype=torch.float)
    
    # 2. Pocket Data
    p_x, p_pos, p_res_idx = torch.empty(0, 44), torch.empty(0, 3), torch.empty(0, dtype=torch.long)
    if pocket_data:
        p_x, p_pos, p_res_idx = pocket_data
        # CRITICAL FIX: Clamp indices to valid sequence length
        if seq_len > 0:
            p_res_idx = torch.clamp(p_res_idx, max=seq_len - 1)

    # 3. 3D Conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0: cid = AllChem.EmbedMolecule(mol, randomSeed=42)
    if cid < 0: return []
    
    AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    conf = mol.GetConformer(cid)
    l_pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float)
    
    # 4. Combine
    combined_x = torch.cat([l_x, p_x], dim=0)
    combined_pos = torch.cat([l_pos, p_pos], dim=0)
    ligand_mask = torch.zeros(combined_x.size(0), dtype=torch.bool)
    ligand_mask[:l_x.size(0)] = True
    
    data = Data(x=combined_x, pos=combined_pos, y=torch.tensor([value], dtype=torch.float),
                ligand_mask=ligand_mask, protein_res_idx=p_res_idx)
    data.protein_seq = seq
    data.pdb_id = pdb_id_str
    return [data]

class LPPDBBind(InMemoryDataset):
  def __init__(self, root, split="train", clean_level="CL1", n_conformers=1, max_samples=0, precompute_esm=False, transform=None, pre_transform=None):
    self.split = split
    self.clean_level = clean_level
    self.n_conformers = n_conformers if split == "train" else 1
    self.max_samples = max_samples
    self.precompute_esm = precompute_esm
    super().__init__(root, transform, pre_transform)
    self.load(self.processed_paths[0])

  @property
  def raw_file_names(self): return ["LP_PDBBind.csv"]
  @property
  def processed_file_names(self): return [f"lp_pdbbind_{self.split}.pt"]
  def download(self):
    dst = os.path.join(self.raw_dir, "LP_PDBBind.csv")
    if not os.path.exists(dst): urllib.request.urlretrieve(CSV_URL, dst)

  def process(self):
    print("DEBUG: Using patched data loader (NaN-Charge Fix Active)")
    df = pd.read_csv(os.path.join(self.raw_dir, "LP_PDBBind.csv"), index_col=0)
    if self.clean_level in df.columns: df = df[df[self.clean_level] & ~df["covalent"]]
    df_split = df[df["new_split"] == self.split].head(self.max_samples if self.max_samples > 0 else len(df))

    # Step 1: Parallel RDKit/PDB Processing
    print(f"Processing {len(df_split)} complexes for {self.split} split using all CPU cores...")
    results = Parallel(n_jobs=-1)(
        delayed(process_single_complex)(str(pdb_id), row["smiles"], float(row["value"]), str(row.get("seq", "")))
        for pdb_id, row in tqdm(df_split.iterrows(), total=len(df_split))
    )
    
    data_list = []
    for r in results: data_list.extend(r)

    # Step 2: Sequential ESM-2 Processing (ESM-2 must stay on GPU/main thread)
    if self.precompute_esm:
      try:
        from models.protein_encoder import precompute_esm2_embedding
        print(f"Precomputing ESM-2 embeddings for {len(data_list)} graphs...")
        for g in tqdm(data_list):
          if g.protein_seq and len(g.protein_seq) > 5:
            g.protein_emb = precompute_esm2_embedding(g.protein_seq, cache_key=g.pdb_id)
      except Exception as exc:
        logger.warning("ESM-2 precomputation failed: %s", exc)

    self.save(data_list, self.processed_paths[0])
