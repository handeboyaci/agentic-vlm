"""Architect agent logic."""

from rdkit import Chem

from agent.base import BaseAgent
from agent.skills import architect


class ArchitectAgent(BaseAgent):
  """Agent responsible for evolving a population of molecules."""

  def execute(
    self,
    population: list[Chem.Mol],
    fitness_scores: list[float],
  ) -> list[Chem.Mol]:
    """Evolves the population for one generation.

    Args:
      population: Current list of molecules.
      fitness_scores: Matching list of fitness floats.

    Returns:
      The next generation of molecules.
    """
    next_gen = architect.evolve_generation(
      population,
      fitness_scores,
      top_k=self.config.top_k_survivors,
      mutation_rate=self.config.mutation_rate,
    )
    # Remove any None entries from failed mutations
    return [m for m in next_gen if m is not None]
