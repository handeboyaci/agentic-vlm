"""LLM-based Lab Manager for hybrid drug discovery orchestration."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.pipeline import DrugDiscoveryPipeline
from config.settings import (
  ArchitectConfig,
  ChemistConfig,
  PhysicistConfig,
  PredictorConfig,
  ScoutConfig,
  LabManagerConfig,
)

logger = logging.getLogger(__name__)


class EntryPhase(str, Enum):
  SCOUT = "SCOUT"
  SEED = "SEED"
  FILTER = "FILTER"
  SCORE = "SCORE"


class FeedbackAction(str, Enum):
  CONTINUE = "CONTINUE"
  STOP = "STOP"


class RoutingDecision(BaseModel):
  phase: EntryPhase = Field(
    description="The phase of the drug discovery pipeline to start at."
  )
  disease_name: Optional[str] = Field(description="The disease name, if mentioned.")
  target_name: Optional[str] = Field(
    description="The specific protein target, if mentioned."
  )
  smiles: Optional[list[str]] = Field(
    description="List of SMILES strings, if provided by the user."
  )


class FeedbackDecision(BaseModel):
  action: FeedbackAction = Field(
    description="Whether to CONTINUE evolving the population or STOP."
  )
  reasoning: str = Field(description="A brief explanation for the decision.")


class LabManager:
  """LLM-driven orchestrator targeting drug discovery.
  
  Wraps the deterministic ``DrugDiscoveryPipeline``, injecting
  LLM-powered routing and feedback decision making.
  """

  def __init__(
    self,
    config: Optional[LabManagerConfig] = None,
    chemist_config: Optional[ChemistConfig] = None,
    architect_config: Optional[ArchitectConfig] = None,
    physicist_config: Optional[PhysicistConfig] = None,
    predictor_config: Optional[PredictorConfig] = None,
    scout_config: Optional[ScoutConfig] = None,
  ) -> None:
    self.config = config or LabManagerConfig()
    
    # Initialize the underlying deterministic pipeline
    self.pipeline = DrugDiscoveryPipeline(
      chemist_config=chemist_config,
      architect_config=architect_config,
      physicist_config=physicist_config,
      predictor_config=predictor_config,
      scout_config=scout_config,
      max_feedback_rounds=self.config.max_iterations,
      scoring=self.config.scoring,
      output_path=self.config.output_path,
    )

    # Initialize Gemini model
    self.model = genai.GenerativeModel(
      model_name=self.config.model_name,
      generation_config=genai.GenerationConfig(
        temperature=self.config.temperature,
      ),
    )

  def _route_entry(self, prompt: str) -> RoutingDecision:
    """Ask the LLM where to start based on the user prompt."""
    logger.info("Asking LLM for entry routing...")

    system_instruction = """
    You are the Lab Manager of an automated drug discovery pipeline.
    Your job is to read the user's request and determine the STARTING phase.
    
    Phases:
    - SCOUT: The user only gave a disease name (e.g., "Find a drug for Alzheimer's").
    - SEED: The user specified a biological target (e.g., "I want to target AChE").
    - FILTER: The user provided specific SMILES strings to optimize.
    - SCORE: The user provided SMILES strings and only wants to score them, not optimize.
    
    Extract the disease_name, target_name, and any smiles if present.
    """

    response = self.model.generate_content(
      f"{system_instruction}\n\nUser Request: {prompt}",
      generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RoutingDecision,
        temperature=self.config.temperature,
      ),
    )
    try:
      data = json.loads(response.text)
      decision = RoutingDecision(**data)
      logger.info("LLM routed to phase: %s", decision.phase.value)
      return decision
    except Exception as e:
      logger.error("Failed to parse LLM routing response: %s", response.text)
      raise ValueError("LLM routing failed: %s" % e)

  def _get_feedback(
    self, predictions: list[dict[str, Any]], iteration: int
  ) -> bool:
    """Ask the LLM whether to continue evolving or stop.
    
    Returns:
      True to continue, False to stop.
    """
    logger.info("Asking LLM for feedback on iteration %d...", iteration)

    system_instruction = f"""
    You are evaluating round {iteration} of a drug discovery pipeline.
    Review the predicted pKa scores and confidence metrics.
    
    Rules:
    - If there are multiple highly confident predictions with good scores (e.g., pKa > 7.0), return STOP.
    - If many predictions are uncertain or scores are low, return CONTINUE to trigger another round of mutation.
    """

    # Prepare a concise summary of results for the LLM
    top_results = sorted(predictions, key=lambda x: x.get("pka_mean", 0), reverse=True)[
      :5
    ]
    summary = "\n".join(
      [
        f"SMILES: {r['smiles']}, pKa: {r.get('pka_mean', 0):.2f}, Confident: {r.get('confident', False)}"
        for r in top_results
      ]
    )

    response = self.model.generate_content(
      f"{system_instruction}\n\nTop Results:\n{summary}",
      generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=FeedbackDecision,
        temperature=self.config.temperature,
      ),
    )
    try:
      data = json.loads(response.text)
      decision = FeedbackDecision(**data)
      logger.info(
        "LLM feedback: %s (%s)",
        decision.action.value,
        decision.reasoning,
      )
      return decision.action == FeedbackAction.CONTINUE
    except Exception as e:
      logger.warning(
        "Failed to parse LLM feedback, defaulting to CONTINUE: %s",
        e,
      )
      return True

  def run(self, prompt: str) -> list[dict[str, Any]]:
    """Execute the pipeline driven by LLM decisions."""
    routing = self._route_entry(prompt)

    disease = routing.disease_name or "Unknown Disease"
    initial_smiles = routing.smiles
    
    # If the user explicitly requested SCORE mode, disable evolution
    if routing.phase == EntryPhase.SCORE:
      generations = 0
    else:
      generations = self.config.generations_per_round
      
    # Delegate to the deterministic pipeline, injecting the LLM feedback
    results, target_info = self.pipeline.run(
      disease_name=disease,
      initial_smiles=initial_smiles,
      generations=generations,
      feedback_fn=self._get_feedback,
    )
    
    return results
