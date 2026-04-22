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
  # Extract partial charge (Gasteiger) if available
  p_charge = 0.0
  if atom.HasProp("_GasteigerCharge"):
      try:
          p_charge = float(atom.GetProp("_GasteigerCharge"))
      except:
          pass

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
      p_charge  # New feature for electrostatics
    ]
    + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
  )  # 44-dim


def get_pocket_atoms(pdb_id: str, refined_dir: str = "data/refined-set") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pdb_path = os.path.join(refined_dir, pdb_id, f"{pdb_id}_pocket.pdb")
    if not os.path.exists(pdb_path):
        return None
    
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=True)
    if mol is None: return None
    
    # Docking preparation: Add charges
    try:
        # Standard protein prep: add hydrogens then calculate charges
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)
        # To keep graph size small, we remove non-polar hydrogens 
        # but keep the charges we just calculated
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        logger.warning(f"Preparation failed for {pdb_id}: {e}")
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


def smiles_to_pyg(
  smiles: str,
  y: float,
  protein_seq: str = "",
  n_conformers: int = 1,
  pdb_id: str = "",
  pocket_data: tuple | None = None,
) -> list[Data]:
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return []
  mol = Chem.AddHs(mol)
  
  # Ligand Preparation: Add charges
  try:
      AllChem.ComputeGasteigerCharges(mol)
  except:
      pass

  # Ligand features
  l_feats = [atom_features(a) for a in mol.GetAtoms()]
  l_x = torch.tensor(l_feats, dtype=torch.float)
  
  # Pocket data
  p_x, p_pos, p_res_idx = torch.empty(0, 44), torch.empty(0, 3), torch.empty(0, dtype=torch.long)
  if pocket_data:
      p_x, p_pos, p_res_idx = pocket_data

  data_list = []
  for seed in range(n_conformers):
    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
      cid = AllChem.EmbedMolecule(mol, randomSeed=seed + 42)
    if cid < 0:
      continue
    
    AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    conf = mol.GetConformer(cid)
    l_pos = torch.tensor(
      [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
      dtype=torch.float,
    )
    
    combined_x = torch.cat([l_x, p_x], dim=0)
    combined_pos = torch.cat([l_pos, p_pos], dim=0)
    
    ligand_mask = torch.zeros(combined_x.size(0), dtype=torch.bool)
    ligand_mask[:l_x.size(0)] = True
    
    data = Data(
      x=combined_x, 
      pos=combined_pos, 
      y=torch.tensor([y], dtype=torch.float),
      ligand_mask=ligand_mask,
      protein_res_idx=p_res_idx
    )
    data.protein_seq = protein_seq
    data.pdb_id = pdb_id
    data_list.append(data)
    
  return data_list


class LPPDBBind(InMemoryDataset):
  def __init__(
    self,
    root: str,
    split: str = "train",
    clean_level: str = "CL1",
    n_conformers: int = 1,
    max_samples: int = 0,
    precompute_esm: bool = False,
    transform=None,
    pre_transform=None,
  ):
    self.split = split
    self.clean_level = clean_level
    self.n_conformers = n_conformers if split == "train" else 1
    self.max_samples = max_samples
    self.precompute_esm = precompute_esm
    super().__init__(root, transform, pre_transform)
    self.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return ["LP_PDBBind.csv"]

  @property
  def processed_file_names(self):
    return [f"lp_pdbbind_{self.split}.pt"]

  def download(self):
    dst = os.path.join(self.raw_dir, "LP_PDBBind.csv")
    if not os.path.exists(dst):
      urllib.request.urlretrieve(CSV_URL, dst)

  def process(self):
    df = pd.read_csv(
      os.path.join(self.raw_dir, "LP_PDBBind.csv"),
      index_col=0,
    )

    if self.clean_level in df.columns:
      df = df[df[self.clean_level] & ~df["covalent"]]

    df_split = df[df["new_split"] == self.split].head(
      self.max_samples if self.max_samples > 0 else len(df)
    )

    esm_fn = None
    if self.precompute_esm:
      try:
        from models.protein_encoder import precompute_esm2_embedding
        esm_fn = precompute_esm2_embedding
      except Exception as exc:
        logger.warning("ESM-2 precomputation unavailable: %s", exc)

    data_list = []
    for pdb_id, row in df_split.iterrows():
      pdb_id_str = str(pdb_id)
      seq = str(row.get("seq", ""))
      
      pocket_data = get_pocket_atoms(pdb_id_str)
      
      graphs = smiles_to_pyg(
        row["smiles"],
        float(row["value"]),
        seq,
        pdb_id=pdb_id_str,
        pocket_data=pocket_data
      )

      if esm_fn and seq and len(seq) > 5:
        try:
          emb = esm_fn(seq, cache_key=pdb_id_str)
          for g in graphs:
            g.protein_emb = emb
        except Exception:
          pass

      data_list.extend(graphs)

    self.save(data_list, self.processed_paths[0])
