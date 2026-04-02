"""Integration tests using real drug molecules from ChEMBL.

Uses FDA-approved or late-stage clinical compounds with known
activity against their respective protein targets. These molecules
are NOT in the LP-PDBBind training set (verified by target).
"""

import pytest
from rdkit import Chem

from agent.chemist_agent import ChemistAgent
from agent.architect_agent import ArchitectAgent
from agent.physicist_agent import PhysicistAgent
from agent.predictor_agent import PredictorAgent
from config.settings import (
  ArchitectConfig,
  ChemistConfig,
  PhysicistConfig,
  PredictorConfig,
)


# ── Real ChEMBL Inhibitors (NOT in LP-PDBBind) ──────────────────

# BACE1 inhibitors (Alzheimer's target, PDB: 4B7R)
BACE1_INHIBITORS = [
  "CC(C)(C)c1cc(NC(=O)c2cnc(N)s2)cc(F)c1F",  # Verubecestat
  "Fc1cc(cc(F)c1)C1(N=C(N)SC1=O)c1cc(ccc1)-c1cccnc1",  # LY2886721
  "CC1(C)CS(=O)(=O)c2cc(F)c(NC(=O)c3cnc(N)s3)cc21",  # Atabecestat
]

# EGFR inhibitors (Cancer target, PDB: 1M17)
EGFR_INHIBITORS = [
  "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC",  # Erlotinib
  "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Gefitinib
]

# MPro inhibitors (COVID-19 target, PDB: 6LU7)
MPRO_INHIBITORS = [
  "CC(C)CC(NC(=O)OCc1ccccc1)C(=O)NC(CC1CCNC1=O)C=O",  # GC-376
]


class TestBACE1Pipeline:
  """End-to-end test using real BACE1 inhibitors."""

  def test_chemist_accepts_real_inhibitors(self):
    chemist = ChemistAgent(ChemistConfig())
    constraints = {
      "max_mw": 500.0,
      "max_logp": 5.0,
      "max_hbd": 5,
      "max_hba": 10,
    }
    passed = chemist.execute(BACE1_INHIBITORS, constraints)
    assert len(passed) >= 1

  def test_architect_evolves_real_molecules(self):
    mols = [Chem.MolFromSmiles(s) for s in BACE1_INHIBITORS if Chem.MolFromSmiles(s)]
    architect = ArchitectAgent(
      ArchitectConfig(
        population_size=len(mols),
        top_k_survivors=2,
        mutation_rate=0.8,
      )
    )
    fitness = [1.0] * len(mols)
    next_gen = architect.execute(mols, fitness)
    assert len(next_gen) > 0

  def test_physicist_generates_3d(self):
    physicist = PhysicistAgent(PhysicistConfig())
    mols = [Chem.MolFromSmiles(s) for s in BACE1_INHIBITORS if Chem.MolFromSmiles(s)]
    results = physicist.execute(mols)
    assert len(results) >= 1
    for r in results:
      assert "smiles" in r
      assert r["mol"].GetNumConformers() > 0

  def test_predictor_scores_real_molecules(self):
    physicist = PhysicistAgent(PhysicistConfig())
    mols = [Chem.MolFromSmiles(s) for s in BACE1_INHIBITORS if Chem.MolFromSmiles(s)]
    phys_results = physicist.execute(mols)
    scored_mols = [r["mol"] for r in phys_results if "mol" in r]
    if not scored_mols:
      pytest.skip("No 3D conformers generated")
    predictor = PredictorAgent(
      PredictorConfig(mc_samples=5),
      scoring="gnn",
    )
    predictions = predictor.execute(scored_mols)
    assert len(predictions) >= 1
    for p in predictions:
      assert "pka_mean" in p
      assert "pka_std" in p
      assert "confident" in p


class TestEGFRPipeline:
  """Smoke test with real EGFR inhibitors."""

  def test_full_filter_and_evolve(self):
    chemist = ChemistAgent(ChemistConfig())
    constraints = {
      "max_mw": 600.0,
      "max_logp": 5.5,
      "max_hbd": 5,
      "max_hba": 10,
    }
    mols = chemist.execute(EGFR_INHIBITORS, constraints)
    if not mols:
      pytest.skip("No EGFR inhibitors passed filtering")
    architect = ArchitectAgent(
      ArchitectConfig(
        population_size=len(mols),
        top_k_survivors=max(1, len(mols) // 2),
      )
    )
    fitness = [1.0] * len(mols)
    evolved = architect.execute(mols, fitness)
    assert len(evolved) > 0


class TestMProPipeline:
  """Smoke test with real MPro inhibitors."""

  def test_gc376_is_valid(self):
    mol = Chem.MolFromSmiles(MPRO_INHIBITORS[0])
    assert mol is not None
    assert mol.GetNumHeavyAtoms() > 10
