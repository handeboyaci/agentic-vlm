"""Physicist agent skills: 3D conformer generation and energy minimisation."""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

logger = logging.getLogger(__name__)


def generate_conformer(
  mol: Optional[Chem.Mol],
  random_seed: int = 42,
) -> Optional[Chem.Mol]:
  """Generates a 3D conformer for a molecule.

  Args:
    mol: The input RDKit molecule.
    random_seed: Seed for reproducible embedding.

  Returns:
    The molecule with a 3D conformer, or None if generation fails.
  """
  if mol is None:
    logger.warning("generate_conformer received None molecule.")
    return None

  mol_h = Chem.AddHs(mol)
  res = AllChem.EmbedMolecule(mol_h, randomSeed=random_seed)

  if res == -1:
    logger.warning("3D embedding failed for molecule: %s", Chem.MolToSmiles(mol))
    return None

  logger.debug("Conformer generated successfully.")
  return mol_h


def minimize_energy(mol: Optional[Chem.Mol]) -> tuple[Optional[Chem.Mol], float]:
  """Minimises the energy of a molecule's conformer using MMFF94.

  Args:
    mol: The input RDKit molecule (must have a conformer).

  Returns:
    A tuple of ``(optimised_mol, final_energy_kcal_per_mol)``.
    Energy is ``float('inf')`` if minimisation fails.

  Raises:
    ValueError: If mol is None or has no conformer.
  """
  if mol is None or mol.GetNumConformers() == 0:
    raise ValueError("minimize_energy requires a molecule with at least one conformer.")

  try:
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
      logger.warning("MMFF parameterisation failed; returning inf energy.")
      return mol, float("inf")

    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
      logger.warning("Could not construct MMFF force field; returning inf energy.")
      return mol, float("inf")

    ff.Minimize()
    energy = ff.CalcEnergy()
    logger.debug("Energy minimisation complete: %.4f kcal/mol", energy)
    return mol, energy
  except Exception as exc:
    logger.warning("Energy minimisation raised an exception: %s", exc)
    return mol, float("inf")


def calculate_3d_descriptors(mol: Optional[Chem.Mol]) -> dict[str, float]:
  """Calculates 3D shape descriptors for a molecule.

  Args:
    mol: The input RDKit molecule (must have a 3D conformer).

  Returns:
    A dictionary of descriptor name → value.  Empty dict on failure.
  """
  if mol is None or mol.GetNumConformers() == 0:
    logger.warning("calculate_3d_descriptors: molecule has no conformer.")
    return {}

  descriptors: dict[str, float] = {}
  try:
    descriptors["NPR1"] = rdMolDescriptors.CalcNPR1(mol)
    descriptors["NPR2"] = rdMolDescriptors.CalcNPR2(mol)
    descriptors["PMI1"] = rdMolDescriptors.CalcPMI1(mol)
    descriptors["PMI2"] = rdMolDescriptors.CalcPMI2(mol)
    descriptors["PMI3"] = rdMolDescriptors.CalcPMI3(mol)
    descriptors["RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
    logger.debug("3D descriptors computed: %s", list(descriptors.keys()))
  except Exception as exc:
    logger.warning("3D descriptor calculation failed: %s", exc)

  return descriptors
