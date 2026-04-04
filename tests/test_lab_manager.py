"""Unit tests for the LLM Lab Manager."""

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from agent.lab_manager import (
  EntryPhase,
  FeedbackAction,
  LabManager,
)


@pytest.fixture
def mock_genai_model():
  """Mock the google.generativeai.GenerativeModel."""
  with patch("agent.lab_manager.genai.GenerativeModel") as mock_model_class:
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    yield mock_instance


@pytest.fixture
def lab_manager(mock_genai_model):
  """Instance of LabManager with mocked LLM and underlying agents."""
  manager = LabManager()

  # Mock the heavy agents so tests run quickly
  manager.pipeline.scout_agent.execute = MagicMock(return_value=({"name": "TestTarget"}, {}))
  manager.pipeline.chemist_agent.execute = MagicMock(
    return_value=[Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C")]
  )
  manager.pipeline.architect_agent.execute = MagicMock(
    return_value=[Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
  )
  manager.pipeline.physicist_agent.execute = MagicMock(
    return_value=[
      {"smiles": "CCO", "mol": Chem.MolFromSmiles("CCO")},
      {"smiles": "CC", "mol": Chem.MolFromSmiles("CC")},
    ]
  )
  manager.pipeline.predictor_agent.execute = MagicMock(
    return_value=[
      {"smiles": "CCO", "pka_mean": 8.5, "confident": True},
      {"smiles": "CC", "pka_mean": 4.1, "confident": False},
    ]
  )

  # For seed fetching
  with patch("agent.skills.seed_molecules.fetch_seed_molecules", return_value=["CCO", "C"]):
    yield manager


def test_routing_disease(lab_manager, mock_genai_model):
  """Test that 'Find a drug for X' correctly routes to SCOUT."""
  mock_response = MagicMock()
  mock_response.text = (
    '{"phase": "SCOUT", "disease_name": "Cancer", "target_name": null, "smiles": null}'
  )
  mock_genai_model.generate_content.return_value = mock_response

  decision = lab_manager._route_entry("Find a drug for Cancer")

  assert decision.phase == EntryPhase.SCOUT
  assert decision.disease_name == "Cancer"


def test_routing_target(lab_manager, mock_genai_model):
  """Test that 'Target AChE' routes to SEED."""
  mock_response = MagicMock()
  mock_response.text = (
    '{"phase": "SEED", "disease_name": null, "target_name": "AChE", "smiles": null}'
  )
  mock_genai_model.generate_content.return_value = mock_response

  decision = lab_manager._route_entry("I want to target AChE")

  assert decision.phase == EntryPhase.SEED
  assert decision.target_name == "AChE"


def test_routing_smiles(lab_manager, mock_genai_model):
  """Test that providing SMILES routes to FILTER."""
  mock_response = MagicMock()
  mock_response.text = '{"phase": "FILTER", "disease_name": null, "target_name": null, "smiles": ["CCO", "CCC"]}'
  mock_genai_model.generate_content.return_value = mock_response

  decision = lab_manager._route_entry("Optimize CCO and CCC")

  assert decision.phase == EntryPhase.FILTER
  assert decision.smiles == ["CCO", "CCC"]


def test_feedback_stop(lab_manager, mock_genai_model):
  """Test the feedback loop stopping when LLM is happy."""
  mock_response = MagicMock()
  mock_response.text = '{"action": "STOP", "reasoning": "Scores are high enough"}'
  mock_genai_model.generate_content.return_value = mock_response

  decision = lab_manager._get_feedback([{"smiles": "CCO", "pka_mean": 9.0}], 1)

  assert decision is False


def test_full_run_with_early_stop(lab_manager, mock_genai_model):
  """Test a full pipeline run that stops after 1 iteration."""
  # First LLM call: Routing
  route_resp = MagicMock()
  route_resp.text = '{"phase": "SCOUT", "disease_name": "Alzheimers", "target_name": null, "smiles": null}'

  # Second LLM call: Feedback says STOP
  feedback_resp = MagicMock()
  feedback_resp.text = '{"action": "STOP", "reasoning": "Good scores"}'

  mock_genai_model.generate_content.side_effect = [route_resp, feedback_resp]

  results = lab_manager.run("Find a drug for Alzheimers")

  assert len(results) == 2
  assert results[0]["smiles"] == "CCO"

  # Verify agents were called (the exact counts depend on generations=3)
  assert lab_manager.pipeline.scout_agent.execute.called
  assert lab_manager.pipeline.chemist_agent.execute.called
  assert lab_manager.pipeline.physicist_agent.execute.called
  assert lab_manager.pipeline.predictor_agent.execute.called
  assert lab_manager.pipeline.architect_agent.execute.called
