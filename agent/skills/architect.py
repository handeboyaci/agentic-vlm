"""Architect agent skills: genetic algorithm for molecule evolution."""

import logging
import random
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, DataStructs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reaction SMARTS for mutation
# ---------------------------------------------------------------------------
REACTION_SMARTS = [
  # Amide coupling (generic)
  "[C:1](=[O:2])[OH]>>[C:1](=[O:2])[N]",
  # Alkylation of amine
  "[N:1]>>[N:1]C",
  # Ether formation from alcohol
  "[O:1][H]>>[O:1]C",
  # Break a ring (simplified)
  "[r:1]>>[*:1]",
  # Add a hydroxyl group to an aromatic ring
  "[c:1]>>[c:1]O",
  # Add a halogen (chlorine) to an aromatic ring
  "[c:1]>>[c:1]Cl",
]


def mutate_molecule(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
  """Mutates a molecule by applying a random chemical reaction.

  Args:
    mol: The input RDKit molecule.

  Returns:
    A new RDKit molecule (mutated), or None if mutation fails.
  """
  if mol is None:
    logger.warning("mutate_molecule received None molecule.")
    return None

  rxn_smarts = random.choice(REACTION_SMARTS)
  rxn = AllChem.ReactionFromSmarts(rxn_smarts)
  ps = rxn.RunReactants((mol,))

  if not ps:
    logger.debug("Mutation produced no products for SMARTS: %s", rxn_smarts)
    return None

  try:
    products = random.choice(ps)
    if not products:
      return None
    product = products[0]
    Chem.SanitizeMol(product)
    logger.debug("Mutation successful: %s", Chem.MolToSmiles(product))
    return product
  except Exception as exc:
    logger.debug("Mutation sanitization failed: %s", exc)
    return None


def crossover_molecules(
  mol1: Optional[Chem.Mol],
  mol2: Optional[Chem.Mol],
) -> Optional[Chem.Mol]:
  """BRICS-based molecular crossover.

  Fragments both parent molecules using BRICS decomposition, then
  recombines a random fragment from each parent into a child molecule.
  Falls back to mutating mol1 if fragmentation yields no usable pieces.

  Args:
    mol1: First parent molecule.
    mol2: Second parent molecule.

  Returns:
    A child RDKit molecule, or None if crossover fails.
  """
  if mol1 is None or mol2 is None:
    logger.warning("crossover_molecules received None molecule(s).")
    return None

  frags1 = list(BRICS.BRICSDecompose(mol1))
  frags2 = list(BRICS.BRICSDecompose(mol2))

  if not frags1 or not frags2:
    logger.debug("BRICS decomposition yielded no fragments; falling back to mutation.")
    return mutate_molecule(mol1)

  # Properly use BRICSBuild to recombine fragments chemically
  try:
    # We combine one random fragment from mol1 with one from mol2
    frag1_mol = Chem.MolFromSmiles(random.choice(frags1))
    frag2_mol = Chem.MolFromSmiles(random.choice(frags2))

    # BRICSBuild yields a generator of possible recombined molecules
    builder = BRICS.BRICSBuild([frag1_mol, frag2_mol])

    # Try to get the first valid child
    for child in builder:
      child.UpdatePropertyCache(strict=False)
      Chem.GetSSSR(child)
      if child.GetNumAtoms() > 0:
        logger.debug("Crossover successful: %s", Chem.MolToSmiles(child))
        return child

  except Exception as exc:
    logger.debug("BRICS Build failed: %s", exc)

  # Fallback if no valid combination is found
  return mutate_molecule(mol1)


def evolve_generation(
  population: list[Chem.Mol],
  fitness_scores: list[float],
  top_k: int = 10,
  mutation_rate: float = 0.5,
) -> list[Chem.Mol]:
  """Evolves the population for one generation.

  Args:
    population: List of RDKit molecules.
    fitness_scores: Scores corresponding to each molecule (higher is better).
    top_k: Number of top molecules to keep as parents.
    mutation_rate: Probability of applying mutation vs. crossover.

  Returns:
    A new list of RDKit molecules (next generation).
  """
  if not population:
    logger.warning("evolve_generation called with empty population.")
    return []

  sorted_pairs = sorted(
    zip(population, fitness_scores), key=lambda x: x[1], reverse=True
  )
  parents = [p[0] for p in sorted_pairs[:top_k]]

  next_gen = list(parents)  # elitism: seed next generation with top survivors
  target_size = len(population)
  max_attempts = target_size * 10
  attempts = 0

  # Pre-compute fingerprints for diversity checking
  fp_list = [_mol_to_fp(m) for m in next_gen]

  while len(next_gen) < target_size and attempts < max_attempts:
    parent = random.choice(parents)
    if random.random() < mutation_rate:
      child = mutate_molecule(parent)
    else:
      other = random.choice(parents)
      child = crossover_molecules(parent, other)

    if child is not None:
      child_fp = _mol_to_fp(child)
      if child_fp is not None and _is_diverse(child_fp, fp_list):
        next_gen.append(child)
        fp_list.append(child_fp)
      elif child_fp is None:
        # Can't compute FP; accept anyway
        next_gen.append(child)
    attempts += 1

  if len(next_gen) < target_size:
    logger.warning(
      "evolve_generation: only produced %d/%d molecules after %d attempts.",
      len(next_gen),
      target_size,
      max_attempts,
    )

  return next_gen


def _mol_to_fp(mol: Chem.Mol):
  """Compute Morgan fingerprint for diversity checking."""
  try:
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
  except Exception:
    return None


def _is_diverse(fp, fp_list, threshold: float = 0.9) -> bool:
  """Return True if fp is sufficiently different from all in fp_list."""
  for existing in fp_list:
    if existing is None:
      continue
    sim = DataStructs.TanimotoSimilarity(fp, existing)
    if sim > threshold:
      return False
  return True
