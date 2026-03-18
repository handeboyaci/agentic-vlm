from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ChemistConfig:
  lipinski_max_mw: float = 500.0
  lipinski_max_logp: float = 5.0
  lipinski_max_hbd: int = 5
  lipinski_max_hba: int = 10
  fingerprint_radius: int = 2
  fingerprint_bits: int = 2048


@dataclass
class ArchitectConfig:
  population_size: int = 20
  mutation_rate: float = 0.1
  top_k_survivors: int = 5
  random_seed: int = 42


@dataclass
class PhysicistConfig:
  random_seed: int = 42
  minimization_iter: int = 100


@dataclass
class PredictorConfig:
  model_path: str = "models/gnn_predictor.pth"
  atom_feat_dim: int = 43
  hidden_dim: int = 128
  n_layers: int = 4
  dropout: float = 0.1
  uncertainty_threshold: float = 0.5
  mc_samples: int = 30


@dataclass
class ScoutConfig:
  default_location: str = "Systemic"
  cns_max_mw: float = 500.0
  cns_max_logp: float = 3.0
  cns_max_hbd: int = 3
  cns_max_psa: float = 90.0


@dataclass
class LabManagerConfig:
  """Configuration for the LLM-based Lab Manager orchestrator."""

  model_name: str = "gemini-2.0-flash"
  max_iterations: int = 5
  temperature: float = 0.1
  scoring: str = "gnn"
  generations_per_round: int = 3
