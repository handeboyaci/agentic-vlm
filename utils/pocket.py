import logging
import os
import urllib.request
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"

# Residues to skip when identifying the ligand
SKIP_HETATM = {
  "HOH", "WAT", "H2O", "DOD",
  "NA", "CL", "MG", "ZN", "CA", "MN", "FE", "CU",
  "CO", "NI", "K", "BR", "I", "CD", "YB", "SM",
  "GOL", "EDO", "PEG", "DMS", "ACT", "FMT", "IMD",
  "SO4", "PO4", "CIT", "TRS", "BME", "MPD", "EPE",
  "MES", "HEP",
}

AA3TO1 = {
  "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
  "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
  "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
  "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
  "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def fetch_pdb(pdb_id: str, cache_dir: str) -> Optional[str]:
  pdb_id = pdb_id.lower().strip()
  os.makedirs(cache_dir, exist_ok=True)
  dst = os.path.join(cache_dir, f"{pdb_id}.pdb")
  if os.path.exists(dst):
    return dst
  url = RCSB_URL.format(pdb_id.upper())
  try:
    urllib.request.urlretrieve(url, dst)
    return dst
  except Exception as exc:
    logger.debug("PDB fetch failed for %s: %s", pdb_id, exc)
    return None

def extract_pocket_seq(pdb_path: str, cutoff: float = 8.0) -> Optional[str]:
  protein_atoms = []
  hetatm_by_res = {}
  try:
    with open(pdb_path) as f:
      for line in f:
        if line.startswith("ATOM"):
          atom_name = line[12:16].strip()
          if atom_name != "CA": continue
          resname = line[17:20].strip()
          chain = line[21]
          resseq = line[22:27].strip()
          x = float(line[30:38])
          y = float(line[38:46])
          z = float(line[46:54])
          protein_atoms.append((resname, resseq, chain, x, y, z))
        elif line.startswith("HETATM"):
          resname = line[17:20].strip()
          if resname in SKIP_HETATM: continue
          x = float(line[30:38])
          y = float(line[38:46])
          z = float(line[46:54])
          hetatm_by_res.setdefault(resname, []).append((x, y, z))
  except Exception as exc:
    logger.debug("PDB parse failed: %s", exc)
    return None

  if not protein_atoms or not hetatm_by_res: return None
  ligand_res = max(hetatm_by_res, key=lambda r: len(hetatm_by_res[r]))
  ligand_coords = np.array(hetatm_by_res[ligand_res])
  
  pocket_residues = []
  seen = set()
  for resname, resseq, chain, x, y, z in protein_atoms:
    key = (chain, resseq)
    if key in seen: continue
    ca_coord = np.array([x, y, z])
    dists = np.linalg.norm(ligand_coords - ca_coord, axis=1)
    if dists.min() <= cutoff:
      pocket_residues.append((int(resseq), AA3TO1.get(resname, "X")))
      seen.add(key)
  
  if not pocket_residues: return None
  pocket_residues.sort(key=lambda x: x[0])
  return "".join(r[1] for r in pocket_residues)

def extract_ligand_center(pdb_path: str) -> Optional[tuple[float, float, float]]:
  """Extract the geometric center (x,y,z) of the largest non-water HETATM."""
  hetatm_by_res = {}
  try:
    with open(pdb_path) as f:
      for line in f:
        if line.startswith("HETATM"):
          resname = line[17:20].strip()
          if resname in SKIP_HETATM: continue
          x = float(line[30:38])
          y = float(line[38:46])
          z = float(line[46:54])
          hetatm_by_res.setdefault(resname, []).append((x, y, z))
  except Exception as exc:
    logger.debug("PDB parse failed: %s", exc)
    return None

  if not hetatm_by_res: return None
  ligand_res = max(hetatm_by_res, key=lambda r: len(hetatm_by_res[r]))
  ligand_coords = np.array(hetatm_by_res[ligand_res])
  center = ligand_coords.mean(axis=0)
  return float(center[0]), float(center[1]), float(center[2])

