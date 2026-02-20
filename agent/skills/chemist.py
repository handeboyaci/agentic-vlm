"""Chemist agent skills: SMILES validation, Lipinski filtering, fingerprints."""
import logging
from typing import Any, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from config.settings import ChemistConfig

logger = logging.getLogger(__name__)

_DEFAULT_CFG = ChemistConfig()


def validate_smiles(smiles: str) -> bool:
  """Validates a SMILES string.

  Args:
    smiles: The SMILES string to validate.

  Returns:
    True if the SMILES string is valid, False otherwise.
  """
  if not smiles:
    logger.debug("Empty SMILES string received.")
    return False
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    logger.debug("Invalid SMILES: %s", smiles)
    return False
  return True


def apply_lipinski_rules(
  mol: Optional[Chem.Mol],
  constraints: Optional[dict[str, Any]] = None,
) -> bool:
  """Checks if a molecule passes Lipinski's Rule of Five.

  Args:
    mol: RDKit molecule object to evaluate.
    constraints: Optional dict of custom thresholds that override defaults.
      Recognised keys: ``max_mw``, ``max_logp``, ``max_hbd``, ``max_hba``.

  Returns:
    True if the molecule passes all rules, False otherwise.

  Raises:
    ValueError: If mol is None.

  Example:
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> apply_lipinski_rules(mol)
    True
  """
  if mol is None:
    logger.error("apply_lipinski_rules received None molecule.")
    raise ValueError("Cannot evaluate Lipinski rules for None molecule.")

  max_mw = float((constraints or {}).get("max_mw", _DEFAULT_CFG.lipinski_max_mw))
  max_logp = float((constraints or {}).get("max_logp", _DEFAULT_CFG.lipinski_max_logp))
  max_hbd = int((constraints or {}).get("max_hbd", _DEFAULT_CFG.lipinski_max_hbd))
  max_hba = int((constraints or {}).get("max_hba", _DEFAULT_CFG.lipinski_max_hba))

  mw = Descriptors.MolWt(mol)
  logp = Descriptors.MolLogP(mol)
  hbd = Descriptors.NumHDonors(mol)
  hba = Descriptors.NumHAcceptors(mol)

  logger.debug("Lipinski check — MW=%.2f, LogP=%.2f, HBD=%d, HBA=%d", mw, logp, hbd, hba)

  if mw > max_mw:
    logger.info("Failed Lipinski: MW %.2f > %.2f", mw, max_mw)
    return False
  if logp > max_logp:
    logger.info("Failed Lipinski: LogP %.2f > %.2f", logp, max_logp)
    return False
  if hbd > max_hbd:
    logger.info("Failed Lipinski: HBD %d > %d", hbd, max_hbd)
    return False
  if hba > max_hba:
    logger.info("Failed Lipinski: HBA %d > %d", hba, max_hba)
    return False

  return True


def get_morgan_fingerprint(
  mol: Optional[Chem.Mol],
  radius: int = _DEFAULT_CFG.fingerprint_radius,
  nBits: int = _DEFAULT_CFG.fingerprint_bits,
) -> np.ndarray:
  """Generates a Morgan fingerprint for a molecule.

  Args:
    mol: RDKit molecule object.
    radius: Radius of the circular fingerprint.
    nBits: Number of bits in the fingerprint vector.

  Returns:
    A numpy array of shape ``(nBits,)`` representing the fingerprint.

  Raises:
    ValueError: If mol is None.
  """
  if mol is None:
    logger.warning("get_morgan_fingerprint received None molecule.")
    raise ValueError("Cannot generate fingerprint for None molecule.")

  fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
  arr = np.zeros((nBits,), dtype=int)  # Fixed: was np.zeros((0,), dtype=int)
  Chem.DataStructs.ConvertToNumpyArray(fp, arr)
  logger.debug("Morgan fingerprint generated: radius=%d, nBits=%d", radius, nBits)
  return arr
