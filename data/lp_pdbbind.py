from __future__ import annotations
import os
import logging
import urllib.request
import pandas as pd
import torch
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
    ]
    + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
  )  # 42-dim


def smiles_to_pyg(
  smiles: str,
  y: float,
  protein_seq: str = "",
  n_conformers: int = 1,
  pdb_id: str = "",
) -> list[Data]:
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return []
  mol = Chem.AddHs(mol)
  feats = [atom_features(a) for a in mol.GetAtoms()]
  x = torch.tensor(feats, dtype=torch.float)
  src, dst = [], []
  for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    src += [i, j]
    dst += [j, i]
  edge_index = torch.tensor([src, dst], dtype=torch.long)
  data_list = []
  for seed in range(n_conformers):
    params = AllChem.ETKDGv3()
    params.randomSeed = seed + 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
      cid = AllChem.EmbedMolecule(mol, randomSeed=seed + 42)
    if cid < 0:
      pos = torch.zeros(mol.GetNumAtoms(), 3)
    else:
      AllChem.MMFFOptimizeMolecule(mol, confId=cid)
      conf = mol.GetConformer(cid)
      pos = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=torch.float,
      )
    data = Data(
      x=x, edge_index=edge_index, pos=pos, y=torch.tensor([y], dtype=torch.float)
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
      os.path.join(self.raw_dir, "LP_PDBBind.csv"), index_col=0
    )

    # Apply LP-PDBBind recommended cleanup:
    # - CL1/CL2/CL3: remove complexes with known data quality issues
    # - covalent: remove covalent binders (different binding physics)
    cl_col = self.clean_level  # e.g. "CL1"
    if cl_col in df.columns:
      df = df[df[cl_col] & ~df["covalent"]]
      logger.info(
        "After %s + non-covalent filter: %d complexes", cl_col, len(df)
      )

    df_split = df[df["new_split"] == self.split].head(
      self.max_samples if self.max_samples > 0 else len(df)
    )
    data_list = []
    for _, row in df_split.iterrows():
      data_list.extend(
        smiles_to_pyg(row["smiles"], float(row["value"]), str(row.get("seq", "")))
      )
    self.save(data_list, self.processed_paths[0])
