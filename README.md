# Virtual Lab Manager (VLM)

**Agentic drug discovery pipeline** that orchestrates five specialised AI agents
to go from a disease name to prioritised lead compounds — fully automated.

## Architecture

```
Disease Query
     │
     ▼
┌──────────┐   RAG (PubMed + ChEMBL)
│  Scout   │──────────────────────────► Target + Constraints
└──────────┘   Gemini / OpenAI / Fallback
     │
     ▼
┌──────────┐
│ Chemist  │──► Lipinski / BBB filtering
└──────────┘
     │
     ▼
┌──────────┐
│ Architect│──► Genetic algorithm (mutation + BRICS crossover)
└──────────┘
     │
     ▼
┌──────────┐
│ Physicist│──► 3D conformer generation + MMFF energy minimisation
└──────────┘
     │
     ▼
┌──────────┐
│ Predictor│──► EGNN binding affinity + MC Dropout uncertainty
└──────────┘
     │
     ▼
  Uncertain? ──yes──► Re-evolve via Architect (feedback loop)
     │
    no
     │
     ▼
  Ranked Lead Candidates
```

## Key Features

| Feature | Detail |
|---|---|
| **RAG Scout** | Retrieves literature from PubMed + ChEMBL, synthesises target with Gemini API, graceful 3-tier fallback |
| **E(n) Equivariant GNN** | SE(3)-equivariant message passing for 3D molecular graphs |
| **Uni-Mol v2 Fine-Tuning** | Foundation model (84M params) fine-tuned on LP-PDBBind for binding affinity |
| **AutoDock Vina** | Physics-based docking (optional, requires conda) |
| **ESM-2 Cross-Attention** | Protein pocket sequence embeddings via cross-attention with ligand nodes |
| **MC Dropout Uncertainty** | Confidence-scored predictions drive the feedback loop |
| **Genetic Algorithm** | BRICS-based crossover + reaction SMARTS mutation |
| **Prompt Engineering** | JSON schema enforcement for structured LLM outputs |

## Scoring Backends

The pipeline supports three interchangeable scoring backends:

```bash
# EGNN + MC Dropout (default)
python agent/pipeline.py --disease "Alzheimer's" --scoring gnn

# Uni-Mol v2 (requires fine-tuned weights — see notebooks/)
python agent/pipeline.py --disease "Cancer" --scoring unimol

# AutoDock Vina (requires conda install vina meeko)
python agent/pipeline.py --disease "COVID-19" --scoring vina
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build the RAG knowledge base
python scripts/build_rag_index.py --diseases "Alzheimer's,Cancer,COVID-19"

# Run the pipeline
export GOOGLE_API_KEY="your-key-here"
python agent/pipeline.py --disease "Alzheimer's" --generations 5 --rounds 2

# Save results as JSON
python agent/pipeline.py --disease "Cancer" --output results.json
```

## Project Structure

```
├── agent/                 # Multi-agent orchestration
│   ├── base.py            # Abstract base agent
│   ├── scout_agent.py     # Target identification
│   ├── chemist_agent.py   # Molecular filtering
│   ├── architect_agent.py # Genetic algorithm evolution
│   ├── physicist_agent.py # 3D conformer generation
│   ├── predictor_agent.py # Binding affinity scoring (GNN/Uni-Mol/Vina)
│   ├── pipeline.py        # End-to-end orchestrator
│   └── skills/            # Agent skill implementations
├── models/                # Neural network architectures
│   ├── egnn_layer.py      # E(n) Equivariant GNN layer
│   ├── multiscale_edges.py# RBF edge features
│   ├── attention_pool.py  # Gated attention pooling
│   ├── gnn_predictor.py   # Full predictor model
│   └── protein_encoder.py # ESM-2 cross-attention
├── notebooks/             # Colab notebooks
│   └── finetune_unimol.ipynb  # Uni-Mol fine-tuning on LP-PDBBind
├── rag/                   # Retrieval-Augmented Generation
│   ├── knowledge_base.py  # PubMed + ChEMBL fetching
│   ├── vector_store.py    # ChromaDB indexing
│   └── llm_synthesizer.py # LLM synthesis + fallback
├── data/                  # Dataset loaders
│   └── lp_pdbbind.py      # LP-PDBBind loader
├── utils/
│   └── pocket.py          # PDB pocket extraction
├── tests/                 # pytest suite
├── config/
│   └── settings.py        # Dataclass configs
└── scripts/
    ├── build_rag_index.py # RAG indexing CLI
    └── train_predictor.py # GNN training script
```

## Tech Stack

- **ML**: PyTorch, PyTorch Geometric, ESM-2, Uni-Mol
- **Chemistry**: RDKit, BRICS decomposition, MMFF94, AutoDock Vina
- **RAG**: ChromaDB, sentence-transformers, Gemini / OpenAI
- **Data**: PubMed E-utilities, ChEMBL REST API, LP-PDBBind

## Testing

```bash
pytest tests/ -v
```

