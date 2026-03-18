"""LLM-based Lab Manager for hybrid drug discovery orchestration."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field
from rdkit import Chem

from agent.chemist_agent import ChemistAgent
from agent.architect_agent import ArchitectAgent
from agent.physicist_agent import PhysicistAgent
from agent.predictor_agent import PredictorAgent
from agent.scout_agent import ScoutAgent
from agent.skills.seed_molecules import fetch_seed_molecules
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
  """LLM-driven orchestrator for the drug discovery pipeline."""

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
    self.scout_agent = ScoutAgent(scout_config or ScoutConfig())
    self.chemist_agent = ChemistAgent(chemist_config or ChemistConfig())
    self.architect_agent = ArchitectAgent(architect_config or ArchitectConfig())
    self.physicist_agent = PhysicistAgent(physicist_config or PhysicistConfig())
    self.predictor_agent = PredictorAgent(
      predictor_config or PredictorConfig(),
      scoring=self.config.scoring,
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
      logger.info(f"LLM routed to phase: {decision.phase.value}")
      return decision
    except Exception as e:
      logger.error(f"Failed to parse LLM routing response: {response.text}")
      raise ValueError(f"LLM routing failed: {e}")

  def _get_feedback(
    self, predictions: list[dict[str, Any]], iteration: int
  ) -> FeedbackDecision:
    """Ask the LLM whether to continue evolving or stop."""
    logger.info(f"Asking LLM for feedback on iteration {iteration}...")

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
        f"LLM feedback decision: {decision.action.value} ({decision.reasoning})"
      )
      return decision
    except Exception as e:
      logger.warning(f"Failed to parse LLM feedback, defaulting to CONTINUE: {e}")
      return FeedbackDecision(
        action=FeedbackAction.CONTINUE, reasoning="Fallback due to parsing error."
      )

  def run(self, prompt: str) -> list[dict[str, Any]]:
    """Execute the pipeline driven by LLM decisions."""
    routing = self._route_entry(prompt)

    target = {}
    constraints = {}
    mols = []

    # ── Node 1: Scout (Target Identification) ──
    if routing.phase == EntryPhase.SCOUT:
      disease = routing.disease_name or "Unknown Disease"
      logger.info(f"Running SCOUT for disease: {disease}")
      target, constraints = self.scout_agent.execute(disease)
      routing.target_name = target.get("name")

    # ── Node 2: Seed (Fetch Initial Molecules) ──
    if routing.phase in (EntryPhase.SCOUT, EntryPhase.SEED):
      target_name = routing.target_name or "Unknown Target"
      disease = routing.disease_name or "Unknown Disease"
      logger.info(f"Running SEED for target: {target_name}")
      seed_smiles = fetch_seed_molecules(
        target_name=target_name,
        disease=disease,
      )
      routing.smiles = seed_smiles

    # ── Node 3: Chemist (Filter Molecules) ──
    if routing.phase in (EntryPhase.SCOUT, EntryPhase.SEED, EntryPhase.FILTER):
      smiles_list = routing.smiles or []
      if not smiles_list:
        logger.warning("No SMILES available to filter. Stopping.")
        return []

      logger.info(f"Running FILTER on {len(smiles_list)} molecules")
      # Use basic constraints if scout was skipped
      if not constraints:
        constraints = {
          "max_mw": self.chemist_agent.config.lipinski_max_mw,
          "max_logp": self.chemist_agent.config.lipinski_max_logp,
          "max_hbd": self.chemist_agent.config.lipinski_max_hbd,
          "max_hba": self.chemist_agent.config.lipinski_max_hba,
        }
      mols = self.chemist_agent.execute(smiles_list, constraints)
    else:
      # SCORE phase provided exactly
      smiles_list = routing.smiles or []
      mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]

    if not mols:
      logger.warning("No valid molecules survived filtering. Stopping.")
      return []

    # ── Iterative Feedback Loop ──
    all_results = []
    scored_smiles = set()
    population = mols
    prev_fitness = [1.0] * len(population)

    # Number of Architect mutations per LLM evaluation
    gens_per_round = self.config.generations_per_round

    for iteration in range(1, self.config.max_iterations + 1):
      logger.info(
        f"── Process Loop Iteration {iteration}/{self.config.max_iterations} ──"
      )

      # ── Node 4: Architect (Evolution) ──
      # We only evolve if we aren't in the very first iteration of a SCORE-only request
      if not (iteration == 1 and routing.phase == EntryPhase.SCORE) and iteration > 1:
        logger.info("Running EVOLVE (Architect)")
        for _ in range(gens_per_round):
          population = self.architect_agent.execute(population, prev_fitness)
          # Note: in a deep loop, Architect might need intermediate scoring,
          # but here we generate a block of generations then evaluate.

      if not population:
        logger.warning("Population died out.")
        break

      # ── Node 5: Physicist (3D Conformations) ──
      logger.info("Running 3D GEOMETRY (Physicist)")
      phys_results = self.physicist_agent.execute(population)
      scored_mols = [r["mol"] for r in phys_results if "mol" in r]

      if not scored_mols:
        break

      # ── Node 6: Predictor (Binding Affinity) ──
      logger.info("Running PREDICTOR (Scoring)")
      protein_id = target.get("pdb_id") or target.get("uniprot")
      predictions = self.predictor_agent.execute(scored_mols, pdb_id=protein_id)

      # Prepare fitness for next round and track unique results
      prev_fitness = []
      pred_map = {p["smiles"]: p for p in predictions}

      for mol in population:
        smiles = Chem.MolToSmiles(mol)
        score = pred_map.get(smiles, {}).get("pka_mean", 0.0)
        prev_fitness.append(score)

      for res in phys_results:
        smi = res["smiles"]
        if smi not in scored_smiles:
          p = pred_map.get(smi)
          if p:
            res.update(p)
            # Remove mol object for cleaner return, keeping smiles
            clean_res = {k: v for k, v in res.items() if k != "mol"}
            all_results.append(clean_res)
            scored_smiles.add(smi)

      # ── Decision Edge: LLM Feedback ──
      if routing.phase == EntryPhase.SCORE and self.config.max_iterations == 1:
        break  # Just score and exit

      feedback = self._get_feedback(predictions, iteration)
      if feedback.action == FeedbackAction.STOP:
        break

    all_results.sort(key=lambda x: x.get("pka_mean", 0), reverse=True)
    return all_results
