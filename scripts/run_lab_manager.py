"""CLI Entry Point for the LLM Lab Manager Orchestrator."""

import argparse
import json
import logging
import os
import sys

from agent.lab_manager import LabManager
from config.settings import ArchitectConfig, LabManagerConfig

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s  %(levelname)-8s  %(message)s",
  stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
  if "GOOGLE_API_KEY" not in os.environ:
    logger.error(
      "GOOGLE_API_KEY environment variable is missing. The Lab Manager needs Gemini to route requests."
    )
    sys.exit(1)

  parser = argparse.ArgumentParser(
    description="LLM-driven Virtual Lab Manager",
  )
  parser.add_argument(
    "prompt",
    type=str,
    help='User request (e.g. "Find a drug for Alzheimer\'s" or "Optimize these SMILES: CCO")',
  )
  parser.add_argument(
    "--generations", type=int, default=3, help="Generations per LLM evaluation round"
  )
  parser.add_argument(
    "--pop_size", type=int, default=10, help="Architect population size"
  )
  parser.add_argument(
    "--max_iterations", type=int, default=5, help="Maximum LLM feedback loops"
  )
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
    default="lab_manager_results.json",
    help="Path to save results as JSON",
  )
  args = parser.parse_args()

  manager = LabManager(
    config=LabManagerConfig(
      max_iterations=args.max_iterations,
      scoring=args.scoring,
      generations_per_round=args.generations,
    ),
    architect_config=ArchitectConfig(
      population_size=args.pop_size,
      top_k_survivors=max(2, args.pop_size // 5),
    ),
  )

  logger.info(f"Starting Lab Manager with prompt: '{args.prompt}'")
  results = manager.run(args.prompt)

  print(f"\nTop Results for request: '{args.prompt[:30]}...'")
  for i, r in enumerate(results[:5]):
    badge = "✅" if r.get("confident") else "⚠️"
    print(f"{i + 1}. {r['smiles']} | pKa: {r.get('pka_mean', 0):.2f} {badge}")

  if args.output and results:
    output = {
      "prompt": args.prompt,
      "candidates": results,
    }
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
  main()
