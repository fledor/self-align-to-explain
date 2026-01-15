# Counterfactual DPO Training Pipeline

A pipeline for generating diverse counterfactuals from classification datasets, evaluating them, training language models with Direct Preference Optimization (DPO), and measuring improvements.

## Overview

This pipeline implements a five-stage approach:

1. **Generate** diverse counterfactuals using high-temperature sampling
2. **Evaluate** counterfactuals on correctness, confidence, and semantic similarity
3. **Construct** DPO preference pairs using unified ranking
4. **Train** the model with DPO to produce better counterfactuals
5. **Compare** base model vs DPO-tuned model (edit distance, LFR, perplexity)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │ ─▶ │   Compare   │
│  (any type) │    │     CFs     │    │     CFs     │    │  DPO Pairs  │    │  with DPO   │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │                  │
                         ▼                  ▼                  ▼                  ▼                  ▼
                   ~40 CFs/entry      correctness,       unified rank        improved CF       Edit Dist,
                   high temp          confidence,        → chosen/reject     model (LoRA)      LFR, PPL
                                      similarity
```

## Currently Implemented Datasets

The pipeline is extensible to any classification task. The following datasets are currently implemented:

- **boolq**: Edit passage to flip yes/no answer
- **snli_premise**: Edit premise to change NLI relationship
- **snli_hypothesis**: Edit hypothesis to change NLI relationship

## Pipeline Stages

### Stage 1: Generate Counterfactuals

```bash
python generate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --max_entries 100 \
    --cfs_per_entry 40 \
    --output_dir ./results/counterfactuals_100e40c
```

Generates diverse counterfactuals using high-temperature sampling. Target labels are distributed evenly across the generated counterfactuals.

**Output:** `results/counterfactuals_{suffix}/{dataset}_progress.jsonl`

### Stage 2: Evaluate Counterfactuals

```bash
python evaluate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --input_dir ./results/counterfactuals_100e40c \
    --output_dir ./results/counterfactuals_100e40c
```

Evaluates each counterfactual on:
- **Correctness**: Did the label change succeed? (verified via LLM)
- **Confidence**: How confident is the model in the new label?
- **Semantic Similarity**: How minimal were the edits? (sentence embeddings)

**Output:** `results/counterfactuals_{suffix}/{dataset}_evaluated.jsonl`

### Stage 3: Construct DPO Pairs

```bash
python construct_dpo_pairs.py \
    --datasets boolq \
    --input_dir ./results/counterfactuals_100e40c \
    --output_dir ./results/dpo_pairs_boolq_100e40c \
    --max_pairs 2
```

Constructs preference pairs using **1-to-1 pairing** (see Design Choices below).

**Output:**
- `dpo_pairs.jsonl` - Minimal format for debugging/analysis
- `dpo_training.jsonl` - Full format for TRL DPOTrainer

### Stage 4: Train with DPO

```bash
python train_dpo_lora.py \
    --dataset_path ./results/dpo_pairs_boolq_100e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_100e40c \
    --max_steps 200 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing
```

Fine-tunes the model using DPO with QLoRA (4-bit quantization).

**Output:** `results/dpo_model_{dataset}_{suffix}/` (LoRA adapter)

### Stage 5: Evaluate Models

```bash
python evaluate_models.py \
    --dpo_model_path ./results/dpo_model_boolq_100e40c \
    --datasets boolq \
    --split validation \
    --num_samples 20 \
    --cfs_per_entry 5 \
    --output_dir ./results/evaluation_boolq_100e40c \
    --resume  # Optional: resume from previous progress
```

Compares base model vs DPO-tuned model on held-out validation data:

| Metric | Description |
|--------|-------------|
| **Levenshtein Distance** | Character-level edit distance (lower = more minimal edits) |
| **Label Flip Rate (LFR)** | % of CFs where the model's prediction changed (higher = better) |
| **Perplexity (PPL)** | Fluency of generated text (lower = more natural) |

**Important**: LFR is computed by comparing each CF's label to the BASE model's prediction on the **original input** (not ground truth). This measures "did the edit change the model's mind?" rather than "did we hit the target label?"

**Fair comparison**: The BASE model is used as the judge for ALL counterfactuals (both base-generated and DPO-generated).

**Notes:** 
- For SNLI datasets, the same entry indices are used for both `snli_premise` and `snli_hypothesis` to ensure fair comparison on identical NLI pairs.
- Supports deduplication, progressive saves (`--resume`), and resume capability.

**Output:** 
- `results/evaluation_{dataset}_{suffix}/original_predictions.json` - Base model predictions on original inputs
- `results/evaluation_{dataset}_{suffix}/eval_report.md` - Human-readable comparison
- `results/evaluation_{dataset}_{suffix}/eval_summary.json` - Structured metrics

---

## Output Directory Naming

Output directories include a **suffix** based on the run configuration to prevent overwriting results:

```
SUFFIX = {entries}e{cfs}c

Example: 100e40c = 100 entries, 40 counterfactuals each
```

This produces directory structures like:
```
results/
├── counterfactuals_100e40c/
│   ├── boolq_progress.jsonl
│   └── boolq_evaluated.jsonl
├── dpo_pairs_boolq_100e40c/
├── dpo_model_boolq_100e40c/
└── evaluation_boolq_100e40c/
```

Run scripts define these at the top:
```bash
ENTRIES=100
CFS=40
SUFFIX="${ENTRIES}e${CFS}c"
```

---

## Design Choices

### DPO Pair Selection: 1-to-1 Pairing with Hard Weighting

We use **1-to-1 pairing** to create strong contrasts between chosen and rejected examples:

```
Unified Score = correctness_bonus + (confidence × similarity)

where correctness_bonus = 100 if label flip succeeded, 0 otherwise
```

**Selection rules:**
1. **Chosen pool**: Only CFs that successfully flipped the label (`is_correct=True`)
2. **Rejected pool**: All valid CFs, sorted by unified score (worst first)

**Pairing:**
- Pair 1: Best chosen ↔ Worst rejected
- Pair 2: 2nd-best chosen ↔ 2nd-worst rejected

```
Before (N×N = 4 pairs):          After (1-to-1 = 2 pairs):
┌─────────┐    ┌──────────┐     ┌─────────┐    ┌──────────┐
│ best    │───▶│ worst    │     │ best    │───▶│ worst    │
│         │───▶│ 2nd-worst│     └─────────┘    └──────────┘
├─────────┤    ├──────────┤     ┌─────────┐    ┌──────────┐
│ 2nd-best│───▶│ worst    │     │ 2nd-best│───▶│ 2nd-worst│
│         │───▶│ 2nd-worst│     └─────────┘    └──────────┘
└─────────┘    └──────────┘     (max 2 pairs, stronger contrasts)
```

**Why this approach:**
- Chosen examples always flip the label (primary objective)
- Each pair has maximum contrast between best and worst
- Avoids weak pairs like second-best↔second-worst in N×N
- Hard weighting ensures correct CFs rank above incorrect CFs

### Diverse Generation with Deduplication

We generate diverse counterfactuals using high temperature and seed variation:

```python
TEMPERATURE = 1.2
TOP_P = 0.99
TOP_K = 100
```

**Diversity mechanisms:**
1. **Unique random seed per generation** - Each CF generation uses a fresh `torch.manual_seed()` to maximize output variety
2. **Post-generation deduplication** - Duplicate `edited_text` values are removed, keeping only unique CFs
3. **Failed parse filtering** - CFs that fail to parse are removed. The model is prompted to wrap its edited text in `<edit>...</edit>` tags; responses without valid tags are discarded.

This means the actual CF count per entry may be less than requested if duplicates or parse failures occur. The tradeoff is that all saved CFs are unique and usable for DPO training.

### Semantic Similarity

We use `sentence-transformers/all-MiniLM-L6-v2` to compute semantic similarity between original and edited text. Higher similarity = more minimal edits.

### Streamlined Data Storage

We store only essential data to minimize file sizes:

**Counterfactuals** (after evaluation):
```json
{
  "edited_text": "The actual edited content",
  "target_label": "entailment",
  "parse_success": true,
  "is_correct": true,
  "predicted_label": "entailment",
  "confidence": 0.95,
  "semantic_similarity": 0.87
}
```

**DPO Pairs** (`dpo_pairs.jsonl` - minimal, for debugging):
```json
{
  "entry_idx": 0,
  "dataset_name": "boolq",
  "original_label": "true",
  "chosen_text": "Edited text without tags",
  "rejected_text": "Edited text without tags",
  "chosen_target_label": "false",
  "chosen_unified_score": 100.85,
  "chosen_is_correct": true
}
```

**DPO Training** (`dpo_training.jsonl` - for TRL DPOTrainer):
```json
{
  "prompt": "<|im_start|>system...",
  "chosen": "Edited text without tags",
  "rejected": "Edited text without tags"
}
```

Note: Chosen/rejected responses are stored as plain text (edit tags removed) since the model should learn to produce the content directly.

---

## Configuration

All settings are centralized in `config.py`:

```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    ACTIVE_DATASETS = ["boolq", "snli_premise", "snli_hypothesis"]
    COUNTERFACTUALS_PER_ENTRY = 40
    
    # High-temperature sampling for diversity
    TEMPERATURE = 1.2
    TOP_P = 0.99  # High for maximum diversity
    TOP_K = 100
```

## Project Structure

```
cfg-dpo/
├── config.py                      # Central configuration
├── dataset_registry.py            # Dataset registry (extensible)
├── prompts.py                     # Prompt templates
├── utils.py                       # Shared utilities
│
├── generate_counterfactuals.py    # Stage 1: Generate CFs
├── evaluate_counterfactuals.py    # Stage 2: Evaluate CFs
├── construct_dpo_pairs.py         # Stage 3: Build DPO pairs
├── train_dpo_lora.py              # Stage 4: DPO training
├── evaluate_models.py             # Stage 5: Model comparison
│
├── requirements.txt
├── README.md
│
├── data/                          # Downloaded datasets (tracked in git)
│   ├── boolq/
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   └── snli/
│       ├── train.jsonl
│       └── validation.jsonl
│
├── model_cache/                   # Cached model weights (gitignored)
├── logs/                          # SLURM job logs (gitignored)
│
└── results/                       # Output files (gitignored)
    ├── counterfactuals_{suffix}/
    ├── dpo_pairs_{dataset}_{suffix}/
    ├── dpo_model_{dataset}_{suffix}/
    └── evaluation_{dataset}_{suffix}/
```

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

## Intermediate Results

Results are saved after each stage, allowing pipeline resumption:

| Stage | Output File | Description |
|-------|-------------|-------------|
| 1 | `counterfactuals_{suffix}/{dataset}_progress.jsonl` | Raw generated CFs |
| 2 | `counterfactuals_{suffix}/{dataset}_evaluated.jsonl` | CFs with evaluation scores |
| 3 | `dpo_pairs_{dataset}_{suffix}/dpo_pairs.jsonl` | Minimal pairs for debugging |
| 3 | `dpo_pairs_{dataset}_{suffix}/dpo_training.jsonl` | Full format for training |
| 4 | `dpo_model_{dataset}_{suffix}/` | LoRA adapter weights |
| 5 | `evaluation_{dataset}_{suffix}/original_predictions.json` | Base model predictions on originals |
| 5 | `evaluation_{dataset}_{suffix}/base_cfs_progress.jsonl` | Base model CF generation progress |
| 5 | `evaluation_{dataset}_{suffix}/dpo_cfs_progress.jsonl` | DPO model CF generation progress |
| 5 | `evaluation_{dataset}_{suffix}/eval_report.md` | Human-readable comparison |
| 5 | `evaluation_{dataset}_{suffix}/eval_summary.json` | Structured metrics |

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (optional, set in `~/.hf_token.env`)

## License

MIT
