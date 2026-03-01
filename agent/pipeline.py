"""End-to-end drug discovery pipeline orchestrator."""
from __future__ import annotations
import logging
import sys
import argparse
from typing import Any, Optional
from rdkit import Chem
from agent.chemist_agent import ChemistAgent
from agent.architect_agent import ArchitectAgent
from agent.physicist_agent import PhysicistAgent
from agent.predictor_agent import PredictorAgent
from agent.scout_agent import ScoutAgent
from config.settings import ArchitectConfig, ChemistConfig, PhysicistConfig, PredictorConfig, ScoutConfig

logger = logging.getLogger(__name__)

class DrugDiscoveryPipeline:
  """Orchestrates the full drug discovery workflow with uncertainty feedback."""
  def __init__(
    self,
    chemist_config: Optional[ChemistConfig] = None,
    architect_config: Optional[ArchitectConfig] = None,
    physicist_config: Optional[PhysicistConfig] = None,
    predictor_config: Optional[PredictorConfig] = None,
    scout_config: Optional[ScoutConfig] = None,
    max_feedback_rounds: int = 3,
  ) -> None:
    self.scout_agent = ScoutAgent(scout_config or ScoutConfig())
    self.chemist_agent = ChemistAgent(chemist_config or ChemistConfig())
    self.architect_agent = ArchitectAgent(architect_config or ArchitectConfig())
    self.physicist_agent = PhysicistAgent(physicist_config or PhysicistConfig())
    self.predictor_agent = PredictorAgent(predictor_config or PredictorConfig())
    self.max_feedback_rounds = max_feedback_rounds

  def run(self, disease_name: str, initial_smiles: list, generations: int = 5) -> list[dict[str, Any]]:
    logger.info("Pipeline starting for disease: %s", disease_name)
    
    # 1. Scout: target identification
    target, constraints = self.scout_agent.execute(disease_name)
    logger.info("Target: %s (location=%s)", target.get("name"), target.get("location"))
    
    # 2. Chemist: filtering
    mols = self.chemist_agent.execute(initial_smiles, constraints)
    if not mols:
      return [], target
    
    all_results = []
    population = mols
    prev_fitness = None
    
    for feedback_round in range(1, self.max_feedback_rounds + 1):
      logger.info("── Feedback round %d/%d ──", feedback_round, self.max_feedback_rounds)
      
      # 3a. Architect: evolution
      fitness = prev_fitness or [1.0] * len(population)
      for _ in range(generations):
        population = self.architect_agent.execute(population, fitness)
        fitness = [1.0] * len(population)
        
      # 3b. Physicist: 3D modeling
      phys_results = self.physicist_agent.execute(population)
      
      # 3c. Predictor: scoring
      scored_mols = [r["mol"] for r in phys_results if "mol" in r]
      if not scored_mols: break
      
      protein_id = target.get("pdb_id") or target.get("uniprot")
      predictions = self.predictor_agent.execute(scored_mols, pdb_id=protein_id)
      
      # Merge
      pred_map = {p["smiles"]: p for p in predictions}
      for res in phys_results:
        p = pred_map.get(res["smiles"])
        if p:
          res.update(p)
          all_results.append(res)
          
      # 3d. Feedback: split by confidence
      uncertain = [p for p in predictions if not p["confident"]]
      if not uncertain:
        logger.info("All molecules confident; stopping.")
        break
        
      population, prev_fitness = [], []
      for u in uncertain:
        m = Chem.MolFromSmiles(u["smiles"])
        if m:
          population.append(m)
          prev_fitness.append(u.get("pka_mean", 0.0))
      if not population: break

    all_results.sort(
      key=lambda r: r.get("pka_mean", 0), reverse=True,
    )
    return all_results, target

if __name__ == "__main__":
  import json

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
  )
  parser = argparse.ArgumentParser(
    description="Virtual Lab Manager Pipeline",
  )
  parser.add_argument(
    "--disease", type=str, default="Alzheimer's",
  )
  parser.add_argument("--generations", type=int, default=5)
  parser.add_argument("--pop_size", type=int, default=10)
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument(
    "--output", type=str, default=None,
    help="Path to save results as JSON",
  )
  args = parser.parse_args()

  SMALL_LIBRARY = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "COCc1cc(C(=O)Nc2ccc(cc2F)OCC3CC3)c(C)n1",
  ]

  pipeline = DrugDiscoveryPipeline(
    architect_config=ArchitectConfig(
      population_size=args.pop_size,
      top_k_survivors=max(2, args.pop_size // 5),
    ),
    max_feedback_rounds=args.rounds,
  )
  results, target = pipeline.run(
    disease_name=args.disease,
    initial_smiles=SMALL_LIBRARY,
    generations=args.generations,
  )

  print("\nTop Results for", args.disease)
  for i, r in enumerate(results[:5]):
    badge = "✅" if r.get("confident") else "⚠️"
    print(
      f"{i+1}. {r['smiles']}"
      f" | pKa: {r.get('pka_mean', 0):.2f} {badge}"
    )

  if args.output:
    # Strip non-serialisable mol objects
    serialisable = []
    for r in results:
      row = {k: v for k, v in r.items() if k != "mol"}
      serialisable.append(row)
    output = {
      "disease": args.disease,
      "target": {
        k: v for k, v in target.items()
        if isinstance(v, (str, int, float, list, bool))
      },
      "candidates": serialisable,
    }
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

