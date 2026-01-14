# Counterfactual DPO Training Pipeline

A pipeline for generating diverse counterfactuals from classification datasets, evaluating them, training language models with Direct Preference Optimization (DPO), and measuring improvements.

## Overview

This pipeline implements a five-stage approach:

1. **Generate** diverse counterfactuals using high-temperature sampling
2. **Evaluate** counterfactuals on correctness, confidence, and semantic similarity
3. **Construct** DPO preference pairs (good vs bad examples)
4. **Train** the model with DPO to produce better counterfactuals
5. **Compare** base model vs DPO-tuned model (edit distance, LFR, perplexity)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │ ─▶ │   Compare   │
│  (any type) │    │     CFs     │    │     CFs     │    │  DPO Pairs  │    │  with DPO   │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │                  │
                         ▼                  ▼                  ▼                  ▼                  ▼
                   ~40 CFs/entry      correctness,       chosen vs         improved CF       Edit Dist,
                   high temp          confidence,        rejected          model (LoRA)      LFR, PPL
                                      similarity
```

## Supported Datasets

- **boolq**: Edit passage to flip yes/no answer
- **snli_premise**: Edit premise to change NLI relationship
- **snli_hypothesis**: Edit hypothesis to change NLI relationship

## Quick Start

### SLURM (HPC)

```bash
# Test run (5 samples, ~30 min)
sbatch run_test.sh

# All datasets (10 samples each, ~1 hour)
sbatch run_all_datasets.sh

# Production run (50 samples, ~3 hours)
sbatch run_50samples.sh
```

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python generate_counterfactuals.py --datasets boolq --max_entries 5
python evaluate_counterfactuals.py --datasets boolq
python construct_dpo_pairs.py --datasets boolq
python train_dpo_lora.py --max_steps 50 --use_4bit
python evaluate_models.py --num_samples 10
```

## Pipeline Stages

### Stage 1: Generate Counterfactuals

```bash
python generate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --max_entries 50 \
    --cfs_per_entry 40
```

Generates diverse counterfactuals using high-temperature sampling. Target labels are distributed evenly across the generated counterfactuals.

**Output:** `results/counterfactuals/{dataset}_progress.jsonl`

### Stage 2: Evaluate Counterfactuals

```bash
python evaluate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis
```

Evaluates each counterfactual on:
- **Correctness**: Did the label change succeed? (verified via LLM)
- **Confidence**: How confident is the model in the new label?
- **Semantic Similarity**: How minimal were the edits?

**Output:** `results/counterfactuals/{dataset}_evaluated.jsonl`

### Stage 3: Construct DPO Pairs

```bash
python construct_dpo_pairs.py \
    --datasets boolq snli_premise snli_hypothesis \
    --chosen_count 2 \
    --rejected_count 2
```

Selects contrastive pairs:
- **Chosen**: Correct + confident + minimal edits
- **Rejected**: Incorrect + overconfident + excessive edits

**Output:** `results/dpo_pairs/dpo_training.jsonl`

### Stage 4: Train with DPO

```bash
python train_dpo_lora.py \
    --dataset_path ./results/dpo_pairs/dpo_training.jsonl \
    --output_dir ./results/dpo_model \
    --max_steps 100 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing
```

Fine-tunes the model using DPO with QLoRA (4-bit quantization).

**Output:** `results/dpo_model/` (LoRA adapter)

### Stage 5: Evaluate Models

```bash
python evaluate_models.py \
    --dpo_model_path ./results/dpo_model \
    --datasets boolq snli_premise snli_hypothesis \
    --split validation \
    --num_samples 20 \
    --cfs_per_entry 5
```

Compares base model vs DPO-tuned model on held-out validation data:

| Metric | Description |
|--------|-------------|
| **Levenshtein Distance** | Character-level edit distance (lower = more minimal edits) |
| **Label Flip Rate (LFR)** | % of CFs that successfully flip the label (higher = better) |
| **Perplexity (PPL)** | Fluency of generated text (lower = more natural) |

**Note:** For SNLI datasets, the same entry indices are used for both `snli_premise` and `snli_hypothesis` to ensure fair comparison on identical NLI pairs.

**Output:** 
- `results/evaluation/eval_report.md` - Human-readable comparison
- `results/evaluation/eval_summary.json` - Structured metrics

## Configuration

All settings are centralized in `config.py`:

```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    ACTIVE_DATASETS = ["boolq", "snli_premise", "snli_hypothesis"]
    COUNTERFACTUALS_PER_ENTRY = 40
    
    # High-temperature sampling for diversity
    TEMPERATURE = 1.2
    TOP_P = 0.95
    TOP_K = 100
```

## Project Structure

```
cfg-dpo/
├── config.py                    # Central configuration
├── dataset_registry.py          # Dataset registry (extensible)
├── prompts.py                   # Prompt templates
├── utils.py                     # Shared utilities
├── generate_counterfactuals.py  # Stage 1: Generate CFs
├── evaluate_counterfactuals.py  # Stage 2: Evaluate CFs
├── construct_dpo_pairs.py       # Stage 3: Build DPO pairs
├── train_dpo_lora.py            # Stage 4: DPO training
├── evaluate_models.py           # Stage 5: Model comparison
├── requirements.txt
├── run_test.sh                  # SLURM: Quick test
├── run_all_datasets.sh          # SLURM: All datasets
├── run_50samples.sh             # SLURM: Production run
├── data/                        # Downloaded datasets (gitignored)
└── results/                     # Output files (gitignored)
    ├── counterfactuals/
    ├── dpo_pairs/
    ├── dpo_model_*/
    └── evaluation/
```

## DPO Pair Selection Logic

The pipeline selects **contrastive** pairs to maximize learning signal:

| | Chosen (Good) | Rejected (Bad) |
|---|---|---|
| Label change | Correct | Failed |
| Confidence | High | High (overconfident) |
| Similarity | High (minimal edits) | Low (excessive changes) |

**Chosen score**: `confidence × similarity` (higher = better)  
**Rejected score**: `confidence × (1 - similarity)` (higher = worse)

## Adding a New Dataset

1. Add a new class in `dataset_registry.py` extending `BaseDataset`:

```python
@register_dataset
class MyNewDataset(BaseDataset):
    name = "my_dataset"
    labels = ["label1", "label2", "label3"]
    edit_target = "text"
    
    def load(self, data_dir, split="train"): ...
    def format_for_prompt(self, entry): ...
    def get_original_text(self, entry): ...
    def get_verification_inputs(self, entry, edited_text): ...
    def parse_label_from_response(self, response): ...
    def build_result_entry(self, entry, counterfactuals): ...
```

2. Add prompt templates in `prompts.py` (register in `PROMPT_REGISTRY`)

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (optional, set in `~/.hf_token.env`)

## License

MIT
