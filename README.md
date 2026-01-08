# Counterfactual DPO Training Pipeline

A pipeline for generating diverse counterfactuals from classification datasets, evaluating them, and training language models with Direct Preference Optimization (DPO) to produce better counterfactuals.

## Overview

This pipeline implements a four-stage approach:

1. **Generate** diverse counterfactuals using high-temperature sampling
2. **Evaluate** counterfactuals on correctness, confidence, and semantic similarity
3. **Construct** DPO preference pairs (good vs bad examples)
4. **Train** the model with DPO to produce better counterfactuals

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │
│  (any type) │    │     CFs     │    │     CFs     │    │  DPO Pairs  │    │  with DPO   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │
                         ▼                  ▼                  ▼                  ▼
                   ~40 CFs/entry      correctness,       chosen vs         improved CF
                   high temp          confidence,        rejected          model
                                      similarity
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All settings are centralized in `config.py`:

```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Easily swap models
    ACTIVE_DATASETS = ["snli_premise", "snli_hypothesis"]  # Add more datasets
    COUNTERFACTUALS_PER_ENTRY = 40
    
    # High-temperature sampling for diversity
    TEMPERATURE = 1.2
    TOP_P = 0.95
    TOP_K = 100
    # ... see config.py for all options
```

## Usage

### 1. Generate Counterfactuals

```bash
python generate_counterfactuals.py \
    --datasets snli_premise snli_hypothesis \
    --cfs_per_entry 40 \
    --resume  # Resume from previous progress
```

Generates diverse counterfactuals using high-temperature sampling. Target labels are distributed evenly across the generated counterfactuals.

Output: `results/counterfactuals/{dataset}_progress.jsonl`

### 2. Evaluate Counterfactuals

```bash
python evaluate_counterfactuals.py \
    --datasets snli_premise snli_hypothesis
```

Evaluates each counterfactual on:
- **Correctness**: Did the label change succeed? (verified via LLM)
- **Confidence**: How confident is the model in the new label?
- **Semantic Similarity**: How minimal were the edits?

Output: `results/counterfactuals/{dataset}_evaluated.jsonl`

### 3. Construct DPO Pairs

```bash
python construct_dpo_pairs.py \
    --datasets snli_premise snli_hypothesis \
    --chosen_count 2 \
    --rejected_count 2
```

Selects contrastive pairs:
- **Chosen**: Correct + confident + minimal edits
- **Rejected**: Incorrect + overconfident + excessive edits

Output: `results/dpo_pairs/dpo_training.jsonl`

### 4. Train with DPO

```bash
python train_dpo_lora.py \
    --dataset_path ./results/dpo_pairs/dpo_training.jsonl \
    --output_dir ./results/dpo_model \
    --num_train_epochs 1 \
    --bf16 \
    --gradient_checkpointing
```

## Project Structure

```
dpo/
├── config.py                    # Central configuration
├── datasets.py                  # Dataset registry (extensible)
├── prompts.py                   # Prompt templates
├── utils.py                     # Shared utilities
├── generate_counterfactuals.py  # Stage 1: Generate CFs
├── evaluate_counterfactuals.py  # Stage 2: Evaluate CFs
├── construct_dpo_pairs.py       # Stage 3: Build DPO pairs
├── train_dpo_lora.py            # Stage 4: DPO training
├── requirements.txt
├── data/                        # Downloaded datasets (gitignored)
└── results/                     # Output files (gitignored)
    ├── counterfactuals/
    └── dpo_pairs/
```

## Datasets

### Currently Supported

- **snli_premise**: Edit the premise to change the NLI relationship
- **snli_hypothesis**: Edit the hypothesis to change the NLI relationship

### Adding a New Dataset

The pipeline is designed to be extensible to any classification task:

1. Add a new class in `datasets.py` extending `BaseDataset`:

```python
@register_dataset
class MyNewDataset(BaseDataset):
    name = "my_dataset"
    labels = ["label1", "label2", "label3"]
    edit_target = "text"  # Field name to edit
    
    def load(self, data_dir: str, split: str = "train") -> list[dict]:
        # Load and return entries as list of dicts
        ...
    
    def format_for_prompt(self, entry: dict) -> dict:
        # Format entry for prompt generation
        ...
```

2. Add prompt templates in `prompts.py` (register in `PROMPT_REGISTRY`)
3. Add dataset name to `Config.ACTIVE_DATASETS`

## DPO Pair Selection Logic

The pipeline selects **contrastive** pairs to maximize learning signal:

| | Chosen (Good) | Rejected (Bad) |
|---|---|---|
| Label change | ✓ Correct | ✗ Failed |
| Confidence | High | High (overconfident) |
| Similarity | High (minimal edits) | Low (excessive changes) |

**Chosen score**: `confidence × similarity` (higher = better)  
**Rejected score**: `confidence × (1 - similarity)` (higher = worse)

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (optional)

## License

MIT
