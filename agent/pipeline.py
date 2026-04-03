"""End-to-end drug discovery pipeline orchestrator.

Unified orchestrator combining the deterministic pipeline with
LabManager's agentic features:
  - Post-evolution chemist filtering
  - Dead-population recovery from previously scored molecules
  - Structured completion summary logging
  - Auto-save to JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from rdkit import Chem

from agent.chemist_agent import ChemistAgent
from agent.architect_agent import ArchitectAgent
from agent.physicist_agent import PhysicistAgent
from agent.predictor_agent import PredictorAgent
from agent.scout_agent import ScoutAgent
from config.settings import (
  ArchitectConfig,
  ChemistConfig,
  PhysicistConfig,
  PredictorConfig,
  ScoutConfig,
)

logger = logging.getLogger(__name__)


class DrugDiscoveryPipeline:
  """Orchestrates the full drug discovery workflow.

  Runs: Scout → Chemist → (Architect → Chemist → Physicist →
  Predictor) × rounds, with dead-population recovery and
  structured summaries.
  """

  def __init__(
    self,
    chemist_config: Optional[ChemistConfig] = None,
    architect_config: Optional[ArchitectConfig] = None,
    physicist_config: Optional[PhysicistConfig] = None,
    predictor_config: Optional[PredictorConfig] = None,
    scout_config: Optional[ScoutConfig] = None,
    max_feedback_rounds: int = 3,
    scoring: str = "gnn",
    output_path: str = "",
  ) -> None:
    self.scout_agent = ScoutAgent(scout_config or ScoutConfig())
    self.chemist_agent = ChemistAgent(
      chemist_config or ChemistConfig(),
    )
    self.architect_agent = ArchitectAgent(
      architect_config or ArchitectConfig(),
    )
    self.physicist_agent = PhysicistAgent(
      physicist_config or PhysicistConfig(),
    )
    self.predictor_agent = PredictorAgent(
      predictor_config or PredictorConfig(),
      scoring=scoring,
    )
    self.max_feedback_rounds = max_feedback_rounds
    self.output_path = output_path

  def run(
    self,
    disease_name: str,
    initial_smiles: list | None = None,
    generations: int = 5,
  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the pipeline.

    Returns:
      Tuple of (sorted results list, target info dict).
    """
    logger.info("Pipeline starting for disease: %s", disease_name)
    stop_reason = "max rounds reached"

    # ── 1. Scout: target identification ───────────────────
    target, constraints = self.scout_agent.execute(disease_name)
    logger.info(
      "Target: %s (location=%s)",
      target.get("name"),
      target.get("location"),
    )

    # ── 2. Seed: fetch initial molecules ──────────────────
    if initial_smiles is None:
      from agent.skills.seed_molecules import fetch_seed_molecules

      target_name = target.get("name", disease_name)
      initial_smiles = fetch_seed_molecules(
        target_name=target_name,
        disease=disease_name,
      )
      logger.info(
        "Auto-fetched %d seed molecules from ChEMBL for '%s'",
        len(initial_smiles),
        target_name,
      )

    # ── 3. Chemist: initial filtering ─────────────────────
    mols = self.chemist_agent.execute(initial_smiles, constraints)
    if not mols:
      return ([], target)

    all_results: list[dict[str, Any]] = []
    scored_smiles: set[str] = set()
    population = mols
    prev_fitness: list[float] | None = None
    rounds_run = 0

    for feedback_round in range(1, self.max_feedback_rounds + 1):
      rounds_run = feedback_round
      logger.info(
        "── Feedback round %d/%d ──",
        feedback_round,
        self.max_feedback_rounds,
      )

      fitness = prev_fitness or [1.0] * len(population)

      for gen in range(generations):
        # ── 4. Architect: evolution ──────────────────────
        population = self.architect_agent.execute(population, fitness)

        # ── 5. Chemist: post-evolution filter ────────────
        pop_smiles = [Chem.MolToSmiles(m) for m in population]
        population = self.chemist_agent.execute(pop_smiles, constraints)

        # ── Dead-population recovery ─────────────────────
        if not population:
          if all_results:
            top_k = min(5, len(all_results))
            logger.warning(
              "Population died. Re-seeding from top-%d previously scored molecules.",
              top_k,
            )
            top_prev = sorted(
              all_results,
              key=lambda x: x.get("pka_mean", 0),
              reverse=True,
            )[:top_k]
            population = [
              Chem.MolFromSmiles(r["smiles"])
              for r in top_prev
              if Chem.MolFromSmiles(r["smiles"])
            ]
          if not population:
            stop_reason = "population died out"
            logger.warning("Population died with no recovery possible.")
            break

        # ── 6. Physicist: 3D conformations ───────────────
        phys_results = self.physicist_agent.execute(population)

        # ── 7. Predictor: binding affinity ───────────────
        scored_mols = [r["mol"] for r in phys_results if "mol" in r]
        if not scored_mols:
          break

        protein_id = target.get("pdb_id") or target.get("uniprot")
        predictions = self.predictor_agent.execute(scored_mols, pdb_id=protein_id)

        # Update fitness for next generation
        pred_map = {p["smiles"]: p for p in predictions}
        fitness = []
        for mol in population:
          smiles = Chem.MolToSmiles(mol)
          score = pred_map.get(smiles, {}).get("pka_mean", 0.0)
          fitness.append(score)

        # Merge new unique molecules into results
        for res in phys_results:
          smi = res["smiles"]
          if smi not in scored_smiles:
            p = pred_map.get(smi)
            if p:
              res.update(p)
              clean = {k: v for k, v in res.items() if k != "mol"}
              all_results.append(clean)
              scored_smiles.add(smi)

      # ── Feedback: split by confidence ──────────────────
      if not predictions:
        stop_reason = "no predictions"
        break

      uncertain = [p for p in predictions if not p["confident"]]
      if not uncertain:
        stop_reason = "all molecules confident"
        logger.info("All molecules confident; stopping.")
        break

      population, prev_fitness = [], []
      for u in uncertain:
        m = Chem.MolFromSmiles(u["smiles"])
        if m:
          population.append(m)
          prev_fitness.append(u.get("pka_mean", 0.0))

      if not population:
        stop_reason = "no uncertain molecules to evolve"
        break

    all_results.sort(key=lambda r: r.get("pka_mean", 0), reverse=True)

    # ── Structured completion summary ──────────────────────
    top_smi = all_results[0]["smiles"] if all_results else "N/A"
    top_pka = all_results[0].get("pka_mean", 0) if all_results else 0
    logger.info(
      "\n══════════════════════════════════════════\n"
      "  Pipeline Complete\n"
      "  Rounds run: %d\n"
      "  Molecules explored: %d\n"
      "  Stop reason: %s\n"
      "  Top candidate: %s (pKa=%.2f)\n"
      "══════════════════════════════════════════",
      rounds_run,
      len(all_results),
      stop_reason,
      top_smi,
      top_pka,
    )

    # ── Auto-save ──────────────────────────────────────────
    if self.output_path and all_results:
      output = {
        "disease": disease_name,
        "target": {
          k: v
          for k, v in target.items()
          if isinstance(v, (str, int, float, list, bool))
        },
        "candidates": all_results,
      }
      os.makedirs(
        os.path.dirname(self.output_path) or ".",
        exist_ok=True,
      )
      with open(self.output_path, "w") as f:
        json.dump(output, f, indent=2)
      logger.info("Results auto-saved to %s", self.output_path)

    return all_results, target


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
  )
  parser = argparse.ArgumentParser(
    description="Virtual Lab Manager Pipeline",
  )
  parser.add_argument("--disease", type=str, default="Alzheimer's")
  parser.add_argument("--generations", type=int, default=5)
  parser.add_argument("--pop_size", type=int, default=10)
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument(
    "--scoring",
    type=str,
    default="gnn",
    choices=["gnn", "unimol", "vina"],
  )
  parser.add_argument("--output", type=str, default=None)
  args = parser.parse_args()

  pipeline = DrugDiscoveryPipeline(
    architect_config=ArchitectConfig(
      population_size=args.pop_size,
      top_k_survivors=max(2, args.pop_size // 5),
    ),
    max_feedback_rounds=args.rounds,
    scoring=args.scoring,
    output_path=args.output or "",
  )
  results, target = pipeline.run(
    disease_name=args.disease,
    generations=args.generations,
  )

  print("\nTop Results for", args.disease)
  for i, r in enumerate(results[:5]):
    badge = "✅" if r.get("confident") else "⚠️"
    print(f"{i + 1}. {r['smiles']} | pKa: {r.get('pka_mean', 0):.2f} {badge}")
