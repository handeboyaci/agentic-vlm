"""Chemist agent logic."""

from typing import Any

from rdkit import Chem

from agent.base import BaseAgent
from agent.skills import chemist


class ChemistAgent(BaseAgent):
  """Agent responsible for checking molecular properties and creating fingerprints."""

  def execute(
    self, smiles_list: list[str], constraints: dict[str, Any]
  ) -> list[Chem.Mol]:
    """Filters a list of SMILES strings based on constraints.

    Args:
      smiles_list: List of SMILES to evaluate.
      constraints: Dictionary of Lipinski constraints (max_mw, etc).

    Returns:
      A list of valid RDKit molecules that passed the filters.
    """
    passed: list[Chem.Mol] = []
    for smi in smiles_list:
      if not chemist.validate_smiles(smi):
        continue
      mol = Chem.MolFromSmiles(smi)
      if chemist.apply_lipinski_rules(mol, constraints=constraints):
        passed.append(mol)
    return passed
