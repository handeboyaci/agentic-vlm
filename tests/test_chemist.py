"""Unit tests for chemist.py skills."""
import numpy as np
import pytest
from rdkit import Chem

from agent.skills import chemist


# -------------------------------------------------------------------
# validate_smiles
# -------------------------------------------------------------------


@pytest.mark.parametrize(
  "smiles,expected",
  [
    ("CCO", True),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", True),  # aspirin
    ("InvalidSMILES", False),
    ("", False),
  ],
)
def test_validate_smiles(smiles: str, expected: bool):
  assert chemist.validate_smiles(smiles) == expected


# -------------------------------------------------------------------
# apply_lipinski_rules
# -------------------------------------------------------------------


def test_lipinski_passes_aspirin(aspirin: Chem.Mol):
  assert chemist.apply_lipinski_rules(aspirin) is True


def test_lipinski_passes_caffeine(caffeine: Chem.Mol):
  assert chemist.apply_lipinski_rules(caffeine) is True


def test_lipinski_fails_large_molecule():
  cyclosporin_smiles = (
    "CC[C@@H]1NC(=O)[C@H]([C@H](CC)C)N(C)C(=O)"
    "[C@@H](CC(C)C)NC(=O)[C@H](CC(C)C)N(C)C(=O)"
    "[C@@H](CC(C)C)NC(=O)[C@H](C)N(C)C(=O)"
    "[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)N(C)C(=O)"
    "[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)N(C)C1=O"
  )
  mol = Chem.MolFromSmiles(cyclosporin_smiles)
  if mol is not None:
    assert chemist.apply_lipinski_rules(mol) is False


def test_lipinski_raises_on_none():
  with pytest.raises(ValueError):
    chemist.apply_lipinski_rules(None)


def test_lipinski_custom_constraints(aspirin: Chem.Mol):
  assert (
    chemist.apply_lipinski_rules(
      aspirin, constraints={"max_mw": 100},
    )
    is False
  )


# -------------------------------------------------------------------
# get_morgan_fingerprint
# -------------------------------------------------------------------


def test_morgan_fingerprint_shape(ethanol: Chem.Mol):
  fp = chemist.get_morgan_fingerprint(ethanol)
  assert isinstance(fp, np.ndarray)
  assert fp.shape == (2048,)


def test_morgan_fingerprint_custom_bits(aspirin: Chem.Mol):
  fp = chemist.get_morgan_fingerprint(aspirin, nBits=1024)
  assert fp.shape == (1024,)


def test_morgan_fingerprint_raises_on_none():
  with pytest.raises(ValueError):
    chemist.get_morgan_fingerprint(None)
