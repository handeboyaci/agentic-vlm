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
from config.settings import (
  ArchitectConfig,
  ChemistConfig,
  PhysicistConfig,
  PredictorConfig,
  ScoutConfig,
)

logger = logging.getLogger(__name__)


class DrugDiscoveryPipeline:
  """Orchestrates the full drug discovery workflow."""

  def __init__(
    self,
    chemist_config: Optional[ChemistConfig] = None,
    architect_config: Optional[ArchitectConfig] = None,
    physicist_config: Optional[PhysicistConfig] = None,
    predictor_config: Optional[PredictorConfig] = None,
    scout_config: Optional[ScoutConfig] = None,
    max_feedback_rounds: int = 3,
    scoring: str = "gnn",
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

  def run(
    self,
    disease_name: str,
    initial_smiles: list | None = None,
    generations: int = 5,
  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger.info("Pipeline starting for disease: %s", disease_name)

    # 1. Scout: target identification
    target, constraints = self.scout_agent.execute(disease_name)
    logger.info(
      "Target: %s (location=%s)",
      target.get("name"),
      target.get("location"),
    )

    # Auto-fetch seed molecules from ChEMBL using target protein
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

    # 2. Chemist: filtering
    mols = self.chemist_agent.execute(initial_smiles, constraints)
    if not mols:
      return ([], target)

    all_results = []
    scored_smiles = set()
    population = mols
    prev_fitness = None

    for feedback_round in range(1, self.max_feedback_rounds + 1):
      logger.info(
        "── Feedback round %d/%d ──", feedback_round, self.max_feedback_rounds
      )

      fitness = prev_fitness or [1.0] * len(population)

      for gen in range(generations):
        # 3a. Architect: evolution
        population = self.architect_agent.execute(population, fitness)

        # 3b. Physicist: 3D modeling
        phys_results = self.physicist_agent.execute(population)

        # 3c. Predictor: scoring
        scored_mols = [r["mol"] for r in phys_results if "mol" in r]
        if not scored_mols:
          break

        protein_id = target.get("pdb_id") or target.get("uniprot")
        predictions = self.predictor_agent.execute(scored_mols, pdb_id=protein_id)

        # Update fitness for the NEXT generation (elitism + parent selection)
        fitness = []
        pred_map = {p["smiles"]: p for p in predictions}

        for mol in population:
          smiles = Chem.MolToSmiles(mol)
          score = pred_map.get(smiles, {}).get("pka_mean", 0.0)
          fitness.append(score)

        # Merge new unique molecules into final output
        for res in phys_results:
          smi = res["smiles"]
          if smi not in scored_smiles:
            p = pred_map.get(smi)
            if p:
              res.update(p)
              all_results.append(res)
              scored_smiles.add(smi)

      # 3d. Feedback: split by confidence of the final generation
      if not predictions:
        break

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

      if not population:
        break

    all_results.sort(
      key=lambda r: r.get("pka_mean", 0),
      reverse=True,
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
    "--disease",
    type=str,
    default="Alzheimer's",
  )
  parser.add_argument("--generations", type=int, default=5)
  parser.add_argument("--pop_size", type=int, default=10)
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument(
    "--scoring",
    type=str,
    default="gnn",
    choices=["gnn", "unimol", "vina"],
    help="Scoring backend: gnn (default), unimol, or vina",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
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
    scoring=args.scoring,
  )
  results, target = pipeline.run(
    disease_name=args.disease,
    initial_smiles=SMALL_LIBRARY,
    generations=args.generations,
  )

  print("\nTop Results for", args.disease)
  for i, r in enumerate(results[:5]):
    badge = "✅" if r.get("confident") else "⚠️"
    print(f"{i + 1}. {r['smiles']} | pKa: {r.get('pka_mean', 0):.2f} {badge}")

  if args.output and results:
    # ── Cross-evaluate with ALL scoring backends ──
    from agent.predictor_agent import PredictorAgent
    from rdkit import Chem

    # Reconstruct Mol objects
    mols = []
    for r in results:
      mol = r.get("mol")
      if mol is None:
        from agent.skills import physicist

        sm_mol = Chem.MolFromSmiles(r["smiles"])
        mol = physicist.generate_conformer(sm_mol)
      mols.append(mol)

    protein_id = target.get("pdb_id") or target.get("uniprot")
    all_backends = ["gnn", "unimol", "vina"]
    score_maps: dict[str, dict[str, float]] = {}

    for backend in all_backends:
      if backend == args.scoring:
        # Already scored during generation
        score_maps[backend] = {r["smiles"]: r.get("pka_mean", 0.0) for r in results}
        continue
      print(f"\nCross-evaluating with {backend}...")
      try:
        pred = PredictorAgent(scoring=backend)
        preds = pred.execute(mols, pdb_id=protein_id)
        score_maps[backend] = {p["smiles"]: p.get("pka_mean", 0.0) for p in preds}
      except Exception as exc:
        logger.warning("Cross-eval with %s failed: %s", backend, exc)
        score_maps[backend] = {}

    # Build serialisable output with all scores
    serialisable = []
    for r in results:
      row = {k: v for k, v in r.items() if k != "mol"}
      smi = row["smiles"]
      for backend in all_backends:
        key = f"pka_{backend}"
        if backend == args.scoring:
          row[key] = row.pop("pka_mean", 0.0)
        elif smi in score_maps.get(backend, {}):
          row[key] = score_maps[backend][smi]
      serialisable.append(row)

    output = {
      "disease": args.disease,
      "target": {
        k: v for k, v in target.items() if isinstance(v, (str, int, float, list, bool))
      },
      "candidates": serialisable,
    }
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Generate 3-method comparison chart ──
    try:
      import matplotlib

      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
      import numpy as np

      labels = [f"Mol {i + 1}" for i in range(len(serialisable))]
      x = np.arange(len(labels))

      # Collect scores per backend (only backends that produced results)
      active_backends = []
      backend_scores = []
      colors = {"gnn": "#4A90D9", "unimol": "#E67E22", "vina": "#2ECC71"}

      for backend in all_backends:
        key = f"pka_{backend}"
        scores = [r.get(key) for r in serialisable]
        if any(s is not None for s in scores):
          active_backends.append(backend)
          backend_scores.append([s if s is not None else 0.0 for s in scores])

      n = len(active_backends)
      width = 0.8 / n if n else 0.35

      fig, ax = plt.subplots(figsize=(10, 6))
      for i, (backend, scores) in enumerate(zip(active_backends, backend_scores)):
        offset = (i - (n - 1) / 2) * width
        ax.bar(
          x + offset,
          scores,
          width,
          label=backend.upper(),
          color=colors.get(backend, "#888"),
        )

      ax.set_ylabel("Predicted pKa (Higher = Better)")
      target_name = target.get("name", "Unknown")
      pdb_label = target.get("pdb_id", "")
      title_suffix = f" — {target_name}" + (f" ({pdb_label})" if pdb_label else "")
      ax.set_title(f"Tri-Model Scoring for {args.disease}{title_suffix}")
      ax.set_xticks(x)
      ax.set_xticklabels(labels, rotation=45, ha="right")
      ax.legend()

      chart_path = (
        args.output.replace(".json", "_chart.png")
        if ".json" in args.output
        else "results_chart.png"
      )
      fig.tight_layout()
      plt.savefig(chart_path, dpi=150)
      plt.close(fig)
      print(f"Comparison chart saved to {chart_path}")
    except ImportError:
      logger.warning("matplotlib not installed, skipping chart generation.")
    except Exception as exc:
      logger.warning("Failed to generate chart: %s", exc)
