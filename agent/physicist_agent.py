"""Physicist agent logic."""

from typing import Any

from rdkit import Chem

from agent.base import BaseAgent
from agent.skills import physicist


class PhysicistAgent(BaseAgent):
  """Agent responsible for 3D modeling and physical descriptors."""

  def execute(self, mols: list[Chem.Mol]) -> list[dict[str, Any]]:
    """Evaluates 3D geometries and minimises energy.

    Args:
      mols: List of molecules to evaluate.

    Returns:
      List of result dictionaries.
    """
    results: list[dict[str, Any]] = []
    for mol in mols:
      mol_3d = physicist.generate_conformer(mol, random_seed=self.config.random_seed)
      if mol_3d is None:
        continue

      mol_min, energy = physicist.minimize_energy(mol_3d)
      descriptors = physicist.calculate_3d_descriptors(mol_min)

      results.append(
        {
          "smiles": Chem.MolToSmiles(mol_min),
          "mol": mol_min,
          "energy": energy,
          "descriptors": descriptors,
          "passed_lipinski": True,
        }
      )
    return results
