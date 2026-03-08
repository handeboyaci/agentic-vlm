"""Shared pytest fixtures for VLM test suite."""

import pytest
from rdkit import Chem


@pytest.fixture
def aspirin() -> Chem.Mol:
  """Aspirin — MW 180, passes Lipinski."""
  return Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")


@pytest.fixture
def caffeine() -> Chem.Mol:
  """Caffeine — MW 194, passes Lipinski."""
  return Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")


@pytest.fixture
def ethanol() -> Chem.Mol:
  """Ethanol — tiny molecule, always passes Lipinski."""
  return Chem.MolFromSmiles("CCO")
