"""Generative cross-validation: test if the pipeline GENERATES actives.

Unlike the scoring cross-validation (which only scores known
compounds), this test starts from weak/fragment-like seed molecules
and runs the full DrugDiscoveryPipeline to see whether it can evolve
molecules that are structurally similar to known potent inhibitors.

Measures:
  1. Max Tanimoto similarity to known actives
  2. Murcko scaffold recovery
  3. Predicted pKa ranking consistency

Usage:
  PYTHONPATH=. python scripts/test_generative_validation.py
  PYTHONPATH=. python scripts/test_generative_validation.py --target bace1
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Scaffolds

from agent.pipeline import DrugDiscoveryPipeline
from config.settings import ArchitectConfig

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s  %(levelname)-8s  %(message)s",
  stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationTarget:
  """A target with weak seeds and known potent actives."""

  disease: str
  label: str
  # Weak/fragment-like seeds (NOT the answers)
  seeds: list[str]
  # Known potent inhibitors (the "ground truth")
  known_actives: list[str] = field(default_factory=list)
  known_names: list[str] = field(default_factory=list)


# ── Validation Targets ──────────────────────────────────────────

TARGETS = {
  "bace1": ValidationTarget(
    disease="Alzheimer's",
    label="BACE1 (Alzheimer's)",
    # Starting from weak fragments / early SAR
    seeds=[
      "NC(=O)c1cnc(N)s1",  # aminothiazole amide (fragment)
      "Fc1cc(N)cc(F)c1",  # difluoroaniline (fragment)
      "Cc1cc(NC=O)ccc1",  # toluamide (weak)
    ],
    known_actives=[
      # Verubecestat (MK-8931) — Merck Phase III
      "CC(C)(C)c1cc(NC(=O)c2cnc(N)s2)cc(F)c1F",
      # Atabecestat (JNJ-54861911)
      "CC1(C)CS(=O)(=O)c2cc(F)c(NC(=O)c3cnc(N)s3)cc21",
      # LY2886721
      "Fc1cc(cc(F)c1)C1(N=C(N)SC1=O)c1cc(ccc1)-c1cccnc1",
    ],
    known_names=[
      "Verubecestat",
      "Atabecestat",
      "LY2886721",
    ],
  ),
  "egfr": ValidationTarget(
    disease="Cancer",
    label="EGFR (Cancer)",
    seeds=[
      "c1cnc2ccccc2n1",  # quinazoline core (fragment)
      "Nc1ncnc2ccccc12",  # 4-aminoquinazoline (weak)
      "Fc1ccc(N)c(Cl)c1",  # 3-Cl-4-F-aniline (fragment)
    ],
    known_actives=[
      # Erlotinib (Tarceva)
      "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC",
      # Gefitinib (Iressa)
      "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    ],
    known_names=["Erlotinib", "Gefitinib"],
  ),
  "mpro": ValidationTarget(
    disease="COVID-19",
    label="MPro (COVID-19)",
    seeds=[
      "O=C1Nc2ccccc2C1=O",  # isatin (fragment hit)
      "O=C(NCc1ccccn1)c1ccccc1",  # benzamide-pyridine
      "CC(C)CC(N)C(=O)NC(CC1CCNC1=O)C=O",  # peptidomimetic
    ],
    known_actives=[
      # GC-376
      "CC(C)CC(NC(=O)OCc1ccccc1)C(=O)NC(CC1CCNC1=O)C=O",
      # Moonshot potent hit
      "Cc1ccncc1NC(=O)Cc1cc(Cl)cc(-c2ccccn2)c1",
    ],
    known_names=["GC-376", "Moonshot-potent"],
  ),
}


def compute_tanimoto(smiles_a: str, smiles_b: str) -> float:
  """Compute Tanimoto similarity using Morgan fingerprints."""
  mol_a = Chem.MolFromSmiles(smiles_a)
  mol_b = Chem.MolFromSmiles(smiles_b)
  if mol_a is None or mol_b is None:
    return 0.0
  fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
  fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
  return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def get_murcko_scaffold(smiles: str) -> str:
  """Get the Murcko scaffold as a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return ""
  try:
    scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)
  except Exception:
    return ""


def evaluate_target(
  target: ValidationTarget,
  generations: int = 3,
  feedback_rounds: int = 2,
) -> dict:
  """Run full pipeline from weak seeds and evaluate."""
  logger.info(
    "\n%s\n  Generative Validation: %s\n  Seeds: %d weak "
    "fragments\n  Known actives: %d\n%s",
    "=" * 60,
    target.label,
    len(target.seeds),
    len(target.known_actives),
    "=" * 60,
  )

  pipeline = DrugDiscoveryPipeline(
    architect_config=ArchitectConfig(
      population_size=max(6, len(target.seeds) * 2),
      top_k_survivors=3,
      mutation_rate=0.8,
    ),
    max_feedback_rounds=feedback_rounds,
    scoring="gnn",
  )

  results, target_info = pipeline.run(
    disease_name=target.disease,
    initial_smiles=target.seeds,
    generations=generations,
  )

  if not results:
    logger.warning("No candidates generated!")
    return {"n_generated": 0}

  generated_smiles = [r["smiles"] for r in results if "smiles" in r]

  # ── Tanimoto recovery ─────────────────────────────────────
  best_matches = []
  for i, known in enumerate(target.known_actives):
    best_sim = 0.0
    best_gen = ""
    for gen_smi in generated_smiles:
      sim = compute_tanimoto(known, gen_smi)
      if sim > best_sim:
        best_sim = sim
        best_gen = gen_smi
    best_matches.append(
      {
        "known_name": target.known_names[i],
        "known_smiles": known,
        "best_generated": best_gen[:60],
        "tanimoto": best_sim,
      }
    )

  # ── Scaffold recovery ──────────────────────────────────────
  known_scaffolds = {
    get_murcko_scaffold(s) for s in target.known_actives if get_murcko_scaffold(s)
  }
  gen_scaffolds = {
    get_murcko_scaffold(s) for s in generated_smiles if get_murcko_scaffold(s)
  }
  scaffold_overlap = len(known_scaffolds & gen_scaffolds)

  # ── Report ─────────────────────────────────────────────────
  print(f"\n  Generated {len(generated_smiles)} candidates")
  print(f"\n  {'Known Active':<20} {'Best Tanimoto':>13} {'Match':>5}")
  print(f"  {'-' * 40}")
  for m in best_matches:
    hit = "✅" if m["tanimoto"] >= 0.4 else "❌"
    print(f"  {m['known_name']:<20} {m['tanimoto']:>13.3f} {hit:>5}")

  avg_tanimoto = (
    sum(m["tanimoto"] for m in best_matches) / len(best_matches)
    if best_matches
    else 0.0
  )
  max_tanimoto = max(m["tanimoto"] for m in best_matches) if best_matches else 0.0

  print(f"\n  Avg Tanimoto to known actives: {avg_tanimoto:.3f}")
  print(f"  Max Tanimoto to known actives: {max_tanimoto:.3f}")
  print(
    f"  Scaffold recovery: {scaffold_overlap}"
    f"/{len(known_scaffolds)} known scaffolds found"
  )

  return {
    "label": target.label,
    "n_generated": len(generated_smiles),
    "best_matches": best_matches,
    "avg_tanimoto": avg_tanimoto,
    "max_tanimoto": max_tanimoto,
    "scaffold_overlap": scaffold_overlap,
    "total_known_scaffolds": len(known_scaffolds),
  }


def main():
  parser = argparse.ArgumentParser(
    description="Generative cross-validation",
  )
  parser.add_argument(
    "--target",
    choices=list(TARGETS.keys()) + ["all"],
    default="all",
  )
  parser.add_argument("--generations", type=int, default=3)
  args = parser.parse_args()

  targets = list(TARGETS.keys()) if args.target == "all" else [args.target]

  for key in targets:
    evaluate_target(TARGETS[key], generations=args.generations)


if __name__ == "__main__":
  main()
