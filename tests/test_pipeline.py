"""Integration test: end-to-end DrugDiscoveryPipeline."""
import pytest

from agent.pipeline import DrugDiscoveryPipeline
from config.settings import ArchitectConfig


SMALL_LIBRARY = [
  "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
  "CCO",  # ethanol
  "c1ccccc1",  # benzene
  "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
]


def test_pipeline_runs_without_error() -> None:
  pipeline = DrugDiscoveryPipeline(
    architect_config=ArchitectConfig(
      population_size=5, top_k_survivors=2,
    ),
    max_feedback_rounds=1,
  )
  results, target = pipeline.run(
    disease_name="Alzheimer's",
    initial_smiles=SMALL_LIBRARY,
    generations=1,
  )
  assert isinstance(results, list)
  assert isinstance(target, dict)


def test_pipeline_result_structure() -> None:
  pipeline = DrugDiscoveryPipeline(max_feedback_rounds=1)
  results, target = pipeline.run(
    disease_name="Cancer",
    initial_smiles=SMALL_LIBRARY,
    generations=1,
  )
  for item in results:
    assert "smiles" in item
    assert "pka_mean" in item
    assert "confident" in item


def test_pipeline_empty_library() -> None:
  pipeline = DrugDiscoveryPipeline()
  results, _ = pipeline.run(
    disease_name="Diabetes",
    initial_smiles=[],
    generations=1,
  )
  assert results == []


def test_pipeline_invalid_smiles_only() -> None:
  pipeline = DrugDiscoveryPipeline()
  results, _ = pipeline.run(
    disease_name="Cancer",
    initial_smiles=["NotASmiles", "AlsoInvalid"],
    generations=1,
  )
  assert results == []
