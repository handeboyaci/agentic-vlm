"""Abstract base class for all VLM agents."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
  """Enforces a consistent interface across all agent skill modules.

  All concrete agents must implement ``execute()``, which serves as the
  single, well-known entry point for the pipeline orchestrator.

  Args:
    config: An agent-specific config dataclass (e.g. ``ChemistConfig``).
  """

  def __init__(self, config: Any) -> None:
    self.config = config

  @abstractmethod
  def execute(self, *args: Any, **kwargs: Any) -> Any:
    """Main entry point for agent execution.

    Returns:
      Agent-specific result; type depends on the concrete implementation.
    """
