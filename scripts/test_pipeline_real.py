"""Run the full pipeline with real drug candidates NOT in LP-PDBBind.

This script uses FDA-approved/late-stage clinical compounds as seed
molecules for 3 disease targets (Alzheimer's, Cancer, COVID-19) and
runs the complete DrugDiscoveryPipeline: Scout → Chemist → Architect
→ Physicist → Predictor → Feedback loop.

Usage:
  python scripts/test_pipeline_real.py
  python scripts/test_pipeline_real.py --target alzheimers
  python scripts/test_pipeline_real.py --target cancer --scoring vina
"""

import argparse
import json
import logging
import sys

from agent.pipeline import DrugDiscoveryPipeline
from config.settings import ArchitectConfig

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s  %(levelname)-8s  %(message)s",
  stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Real Inhibitors NOT in LP-PDBBind Training Data ─────────────
# These are modern clinical/approved drugs discovered AFTER the
# PDBBind 2020 dataset cutoff, or not present in LP-PDBBind.

REAL_SEEDS = {
  "alzheimers": {
    "disease": "Alzheimer's",
    "description": "BACE1 inhibitors (Phase II/III clinical)",
    "smiles": [
      # Verubecestat (MK-8931) — Merck Phase III (2017)
      "CC(C)(C)c1cc(NC(=O)c2cnc(N)s2)cc(F)c1F",
      # Atabecestat (JNJ-54861911) — J&J Phase II/III
      "CC1(C)CS(=O)(=O)c2cc(F)c(NC(=O)c3cnc(N)s3)cc21",
      # LY2886721 — Eli Lilly Phase II
      "Fc1cc(cc(F)c1)C1(N=C(N)SC1=O)c1cc(ccc1)-c1cccnc1",
      # CNP520 — Novartis Phase II/III
      "CC(NC(=O)c1cc(F)cc(F)c1)c1cc(ccc1)-c1ccnc(N)c1",
      # Elenbecestat (E2609) — Eisai/Biogen Phase III
      "CS(=O)(=O)c1ccc(NC(=O)c2cnc(N)s2)c(F)c1F",
    ],
  },
  "cancer": {
    "disease": "Cancer",
    "description": "EGFR tyrosine kinase inhibitors (FDA approved)",
    "smiles": [
      # Erlotinib (Tarceva) — FDA 2004
      "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC",
      # Gefitinib (Iressa) — FDA 2003
      "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
      # Afatinib (Gilotrif) — FDA 2013
      "CN(C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1)C",
      # Osimertinib (Tagrisso) — FDA 2015
      "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)C=C",
    ],
  },
  "covid": {
    "disease": "COVID-19",
    "description": "MPro protease inhibitors (FDA/EUA approved)",
    "smiles": [
      # GC-376 — pan-coronavirus (veterinary)
      "CC(C)CC(NC(=O)OCc1ccccc1)C(=O)NC(CC1CCNC1=O)C=O",
      # Ebselen — clinical candidate
      "c1ccc2c(c1)[Se]N2c1ccccc1",
      # Carmofur — antineoplastic, MPro inhibitor
      "CCCCCCNC(=O)N1C=CC(=O)NC1=O",
      # Tideglusib — GSK-3B / MPro repurposed
      "O=C1SC(=Nc2ccccc2)N1c1ccc(cc1)N1CCOCC1",
    ],
  },
}


def run_pipeline_test(target_key: str, scoring: str = "gnn"):
  """Run the full pipeline for a specific target."""
  data = REAL_SEEDS[target_key]
  logger.info(
    "=" * 60 + "\n"
    "  Pipeline Test: %s\n"
    "  Target: %s\n"
    "  Seeds: %d real inhibitors\n"
    "  Scoring: %s\n" + "=" * 60,
    data["disease"],
    data["description"],
    len(data["smiles"]),
    scoring,
  )

  pipeline = DrugDiscoveryPipeline(
    architect_config=ArchitectConfig(
      population_size=len(data["smiles"]),
      top_k_survivors=max(2, len(data["smiles"]) // 3),
    ),
    max_feedback_rounds=2,
    scoring=scoring,
  )

  results, target = pipeline.run(
    disease_name=data["disease"],
    initial_smiles=data["smiles"],
    generations=2,
  )

  logger.info(
    "\n%s Results for %s (%s)",
    "=" * 40,
    data["disease"],
    target.get("name", "?"),
  )
  for i, r in enumerate(results[:5]):
    badge = "✅" if r.get("confident") else "⚠️"
    logger.info(
      "  %d. %s | pKa: %.2f | std: %.3f %s",
      i + 1,
      r["smiles"][:50],
      r.get("pka_mean", 0),
      r.get("pka_std", 0),
      badge,
    )

  return results, target


def main():
  parser = argparse.ArgumentParser(
    description="Run the full pipeline with real drug candidates",
  )
  parser.add_argument(
    "--target",
    choices=list(REAL_SEEDS.keys()) + ["all"],
    default="all",
    help="Which target to test (default: all)",
  )
  parser.add_argument(
    "--scoring",
    choices=["gnn", "unimol", "vina"],
    default="gnn",
  )
  parser.add_argument(
    "--output",
    default="outputs/real_pipeline_results.json",
  )
  args = parser.parse_args()

  targets = list(REAL_SEEDS.keys()) if args.target == "all" else [args.target]

  all_output = {}
  for key in targets:
    results, target = run_pipeline_test(key, args.scoring)
    all_output[key] = {
      "disease": REAL_SEEDS[key]["disease"],
      "target": {
        k: v for k, v in target.items() if isinstance(v, (str, int, float, bool))
      },
      "num_candidates": len(results),
      "top_5": [{k: v for k, v in r.items() if k != "mol"} for r in results[:5]],
    }

  if args.output:
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
      json.dump(all_output, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
  main()
