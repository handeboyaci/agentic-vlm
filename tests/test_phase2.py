"""Unit tests for architect, physicist, and scout skills."""
import pytest
from rdkit import Chem

from agent.skills import architect, physicist, scout


# -------------------------------------------------------------------
# Architect
# -------------------------------------------------------------------


def test_mutate_molecule_returns_mol_or_none(aspirin: Chem.Mol):
  mutated = architect.mutate_molecule(aspirin)
  assert mutated is None or isinstance(mutated, Chem.Mol)


def test_mutate_molecule_none_input():
  assert architect.mutate_molecule(None) is None


def test_crossover_molecules(aspirin: Chem.Mol, caffeine: Chem.Mol):
  child = architect.crossover_molecules(aspirin, caffeine)
  assert child is None or isinstance(child, Chem.Mol)


def test_crossover_none_input(aspirin: Chem.Mol):
  assert architect.crossover_molecules(aspirin, None) is None
  assert architect.crossover_molecules(None, aspirin) is None


def test_evolve_generation(
  aspirin: Chem.Mol,
  caffeine: Chem.Mol,
  ethanol: Chem.Mol,
):
  population = [aspirin, caffeine, ethanol]
  fitness = [0.9, 0.7, 0.5]
  next_gen = architect.evolve_generation(
    population, fitness, top_k=2, mutation_rate=0.8,
  )
  assert isinstance(next_gen, list)
  assert len(next_gen) == len(population)


def test_evolve_generation_empty():
  result = architect.evolve_generation([], [], top_k=5)
  assert result == []


# -------------------------------------------------------------------
# Physicist
# -------------------------------------------------------------------


def test_generate_conformer(aspirin: Chem.Mol):
  mol_3d = physicist.generate_conformer(aspirin)
  assert mol_3d is not None
  assert mol_3d.GetNumConformers() > 0


def test_generate_conformer_none_input():
  assert physicist.generate_conformer(None) is None


def test_minimize_energy_returns_tuple(aspirin: Chem.Mol):
  mol_3d = physicist.generate_conformer(aspirin)
  assert mol_3d is not None
  result = physicist.minimize_energy(mol_3d)
  assert isinstance(result, tuple)
  assert len(result) == 2
  mol_min, energy = result
  assert isinstance(mol_min, Chem.Mol)
  assert isinstance(energy, float)


def test_minimize_energy_raises_on_none():
  with pytest.raises(ValueError):
    physicist.minimize_energy(None)


def test_calculate_3d_descriptors(aspirin: Chem.Mol):
  mol_3d = physicist.generate_conformer(aspirin)
  assert mol_3d is not None
  mol_min, _ = physicist.minimize_energy(mol_3d)
  descriptors = physicist.calculate_3d_descriptors(mol_min)
  assert "RadiusOfGyration" in descriptors
  assert "NPR1" in descriptors


# -------------------------------------------------------------------
# Scout
# -------------------------------------------------------------------


def _static_search(disease: str):
  """Force static DB lookup (bypass RAG)."""
  return {}


def test_identify_target_alzheimers():
  target = scout.identify_target(
    "Alzheimer's", search_func=_static_search,
  )
  assert target["name"] == "BACE1"
  assert target["location"] == "CNS"


def test_identify_target_unknown():
  target = scout.identify_target(
    "UnknownDisease", search_func=_static_search,
  )
  assert target["name"] == "Unknown"


def test_identify_target_injectable():
  mock = lambda _: {"name": "MockTarget", "location": "Systemic"}
  target = scout.identify_target("Anything", search_func=mock)
  assert target["name"] == "MockTarget"


def test_determine_constraints_cns():
  target = scout.identify_target(
    "Alzheimer's", search_func=_static_search,
  )
  constraints = scout.determine_constraints(target)
  assert constraints["max_mw"] == 400.0  # CNS rule


def test_determine_constraints_systemic():
  target = scout.identify_target(
    "Cancer", search_func=_static_search,
  )
  constraints = scout.determine_constraints(target)
  assert constraints["max_mw"] == 500.0
