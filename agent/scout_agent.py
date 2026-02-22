"""Scout agent logic."""
from typing import Any

from agent.base import BaseAgent
from agent.skills import scout


class ScoutAgent(BaseAgent):
  """Agent responsible for target identification and constraint derivation."""

  def execute(self, disease_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finds the target for a disease and determines constraints.

    Args:
      disease_name: Name of the disease (e.g. "Alzheimer's").

    Returns:
      Tuple of (target_info_dict, constraints_dict).
    """
    target = scout.identify_target(disease_name)
    constraints = scout.determine_constraints(target)
    return target, constraints
