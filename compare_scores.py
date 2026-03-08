from rdkit import Chem
from agent.skills import predictor
from agent.skills import vina_scorer
from agent.skills import physicist
import warnings

warnings.filterwarnings("ignore")

smiles = [
  "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
  "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
]

print("Generating 3D conformers...")
mols = []
for s in smiles:
  m = Chem.MolFromSmiles(s)
  m = physicist.generate_conformer(m)
  mols.append(m)

pdb_id = "1M17"  # EGFR

# 1. EGNN
print(f"\n--- EGNN (Target: {pdb_id} Pocket Emb) ---")
egnn_model = predictor.load_model("gnn_predictor.pth")
for m, s in zip(mols, smiles):
  res = predictor.score_molecule(m, egnn_model, pdb_id=pdb_id)
  if res:
    print(f"{s[:15]:15s} | pKa: {res['pka_mean']:>5.2f} ± {res['pka_std']:.2f}")
  else:
    print(f"{s[:15]:15s} | Failed to score")

# 2. VINA
print(f"\n--- AutoDock Vina (Target: {pdb_id} Full 3D) ---")
vina_results = vina_scorer.score_molecules(mols, pdb_id=pdb_id)
for s, res in zip(smiles, vina_results):
  print(
    f"{s[:15]:15s} | pKa: {res['pka_mean']:>5.2f} (Vina Score: {res['vina_score']:>6.2f} kcal/mol)"
  )
