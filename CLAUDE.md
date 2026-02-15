# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OG-LANS** (Ontology-Graph Loss-Aware Adaptive Negative Sampling) is an academic research project aiming to improve Direct Preference Optimization (DPO) for structured Event Extraction (EE). It addresses the specific challenge of "Reasoning-Extraction Inconsistency" in Large Language Models by introducing a dynamic, loss-aware curriculum learning framework.

- **Goal**: Master's Thesis Chapter / ACL 2026 Paper
- **Core Problem**: Static negative sampling in DPO limits model discrimination capability; "False Negatives" introduce noise; Hallucinations persist despite strong reasoning.
- **Solution**: A dynamic loop where model competence (estimated from IPO loss) dictates the hardness of negative samples (via Ontology Graph distance), filtered by NLI-based consistency checks.

## Project Structure

```
OG-LANS/
├── main.py                 # Training entry point
├── evaluate.py             # Local model evaluation (Strict/Relaxed F1, Hallucination, CoT Faithfulness)
├── evaluate_api.py         # API-based evaluation (DeepSeek/OpenAI)
├── configs/
│   └── config.yaml         # Centralized configuration
├── src/
│   └── oglans/             # Core Python package
│       ├── __init__.py
│       ├── config.py       # ConfigManager singleton
│       ├── data/           # Data adapters and prompt builders
│       ├── trainer/        # CGADPOTrainer, LANSCallback
│       └── utils/          # DS-CNS, SCV, reproducibility, logging
├── scripts/                # Shell scripts and utilities
├── tests/                  # Unit tests (pytest)
├── logs/                   # Training outputs, TensorBoard logs
├── data/                   # Raw and processed datasets
├── pyproject.toml          # Package definition (PEP 621)
├── pytest.ini              # Test configuration
└── backup/                 # Archived non-essential files
```

## Core Methodology & Specifications

### 1. OG-CNS (Ontology-Graph Driven Negative Sampling)
The system generates synthetic negative samples $y_w$ based on the semantic distance in the event ontology graph.
- **Ontology Distance $D(t_i, t_j)$**: Shortest path length between event types/arguments in the ontology graph.
  - $D=1$: Hard Negatives (Siblings/Parent-Child).
  - $D \gg 1$: Easy Negatives (Distant concepts).
- **Multi-Granularity Perturbations**:
  - **Event-level**: Replace `event_type` with neighbors in the graph.
  - **Argument-level**: Role mismatch or entity swap.
  - **Value-level**: Numerical/Entity perturbation (e.g., modifying "100 million" to "10 million").

### 2. OG-LANS (Loss-Aware Adaptive Scheduler)
Dynamically adjusts the difficulty of negative samples based on real-time model competence.
- **Competence Estimation**:
  $$C(t) = \mathrm{EMA}\left(\sigma\left(\alpha - \mathcal{L}_{\text{IPO}}^{(t)}\right)\right)$$
  - $\alpha$: Loss baseline (`lans_alpha` in config, default: 0.5)
  - Uses Exponential Moving Average (EMA) to smooth the signal.
- **Pacing Function (Curriculum)**:
  $$\lambda(C) = D_{\max} - (D_{\max} - D_{\min}) \cdot C$$
  - Determines the *minimum distance* allowed for negative sampling.
  - Low Competence $\to$ High $\lambda$ $\to$ Only Easy Negatives allowed.
  - High Competence $\to$ Low $\lambda$ $\to$ Hard Negatives allowed.
- **CGA (Contrastive Gradient Amplification)**:
  - Weight hard negatives more when model is weak.
  - $w_{grad} = 1 + \beta_{\text{CGA}} \cdot (1 - C)$

### 3. SCV (Semantic Consistency Verification)
Prevents "False Negatives" (valid extractions labeled as wrong) using NLI.
- **Pipeline**:
  1. Convert Negative JSON candidate $\to$ Natural Language Hypothesis $H$.
  2. Input: Premise (Document) + Hypothesis $H$ to NLI Model (Erlangshen-MegatronBert-1.3B).
  3. **Logic**:
     - **Entailment > Threshold**: Reject (False Negative - it's actually true).
     - **Contradiction**: Keep (High Quality Negative).
     - **Neutral**: Keep (Standard Negative).

### 4. CoT & Prompt Engineering
Uses a "Three-Step Reasoning" Chain-of-Thought format.
```markdown
<thought>
Step 1: Schema Analysis (Pattern Recognition)
- Trigger detection...
Step 2: Entity Scanning (Argument Extraction)
- Extracting arguments for identified events...
Step 3: Constraint Checking (Self-Correction)
- Verifying boundaries and schema constraints...
</thought>
```json
[...]
```

## Architecture & Environment

- **Language**: Python 3.10+
- **Framework**: Unsloth + Transformers + PyTorch + TRL
- **Base Model**: Qwen/Qwen3-4B-Instruct-2507
- **Hardware**: Optimized for single RTX 4090 (24GB) using 4-bit QLoRA + Flash Attention 2.

## Common Commands

### Installation

```bash
# Install as editable package (recommended for development)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Training

```bash
# Full training with OG-LANS enabled
python main.py --data_dir ./data/raw/DuEE-Fin --exp_name full_run

# Debug training (fast validation, 10 steps)
bash scripts/run_debug.sh

# With CLI overrides
python main.py --training.max_steps 100 --algorithms.lans.enabled false
```

### Evaluation

```bash
# Local model evaluation (Strict F1, Hallucination Rate, CoT Faithfulness)
python evaluate.py --checkpoint logs/DuEE-Fin/checkpoints/full_run

# API-based evaluation (DeepSeek)
python evaluate_api.py --split dev --use_fewshot
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_lans.py -v
```

### Utilities

```bash
# Rebuild Ontology Graph from Schema
python scripts/build_graph.py
```

## Implementation Details (Source Map)

| Module | Path | Description |
|--------|------|-------------|
| **DS-CNS Sampler** | `src/oglans/utils/ds_cns.py` | OG-CNS sampling and LANS scheduling |
| **SCV Verifier** | `src/oglans/utils/scv.py` | NLI-based false negative filtering |
| **CGA Trainer** | `src/oglans/trainer/unsloth_trainer.py` | Custom `CGADPOTrainer` with gradient amplification |
| **Prompt Builder** | `src/oglans/data/prompt_builder.py` | Chinese CoT template construction |
| **Config Manager** | `src/oglans/config.py` | Singleton configuration loader |
| **Reproducibility** | `src/oglans/utils/reproducibility.py` | Global seed control for deterministic training |

## Configuration Keys (`config.yaml`)

### Core Algorithm Switches
- `algorithms.lans.enabled`: Toggle Dynamic Scheduling (LANS).
- `algorithms.scv.enabled`: Toggle NLI Filtering (SCV).
- `algorithms.lans.use_cga`: Toggle Contrastive Gradient Amplification.
- `algorithms.lans.use_ema`: Toggle EMA smoothing for competence.

### Key Hyperparameters
- `algorithms.lans.lans_alpha`: The $\alpha$ parameter for competence estimation (default: 0.5).
- `algorithms.lans.cga_beta`: CGA strength parameter (default: 0.1).
- `algorithms.lans.strategies.easy_ratio`: Distance threshold for EASY strategy (default: 0.7).
- `algorithms.lans.strategies.hard_ratio`: Distance threshold for HARD strategy (default: 0.4).

### Ablation Experiment Tags
- `experiment.ablation_tag`: Controls which components are disabled.
  - `full`: Complete OG-LANS + SCV + CGA
  - `A1_no_lans`: Static curriculum (no LANS)
  - `A2_no_scv`: No SCV filtering
  - `A3_random_neg`: Random negative sampling
  - `A4_no_ema`: No EMA smoothing
  - `A5_no_cga`: No gradient amplification
  - `A6_no_ontology`: No ontology graph distance
  - `A7_single_granularity`: Single-level perturbation only

## Import Convention

All imports should use the `oglans` package namespace:

```python
# Correct
from oglans.utils.ds_cns import DSCNSampler, LANSScheduler
from oglans.data.adapter import DuEEFinAdapter
from oglans.config import ConfigManager

# Incorrect (legacy)
from src.utils.ds_cns import ...
```
