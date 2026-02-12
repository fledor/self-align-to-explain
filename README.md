# Self-Align to Explain

**Thesis Title:** *Self-Align to Explain: Comparing Post-Training Methods for Counterfactual Generation*

**Abstract:** This thesis compares post-training self-alignment methods for counterfactual example generation. Using self-generated training data, we evaluate SFT, DPO, GRPO, and GDPO on their ability to produce minimal edits that flip classifier predictions.

---

A research pipeline for generating diverse counterfactuals from classification datasets, training language models with various post-training methods, and comparing their effectiveness.

## Research Goal

This project investigates how different training paradigms affect a language model's ability to generate **minimal, label-flipping counterfactuals** for text classification tasks. We compare:

| Method | Type | Description |
|--------|------|-------------|
| **DPO** | Offline | Direct Preference Optimization on pre-generated preference pairs |
| **SFT** | Offline | Supervised Fine-Tuning on successful counterfactuals only |
| **GRPO** | Online | Group Relative Policy Optimization with generation during training |
| **GDPO** | Online | Generalized DPO with iterative generation and training cycles |

**Key distinction:** Offline methods train on pre-generated data (Stages 1-2), while online methods interleave generation and training.

---

## Overview

The pipeline implements a multi-stage approach for **offline methods** (DPO, SFT):

1. **Generate** diverse counterfactuals using high-temperature sampling
2. **Evaluate** counterfactuals on label flip rate, confidence, and semantic similarity
3. **Construct** training data (preference pairs for DPO, successful CFs for SFT)
4. **Train** the model with chosen method (DPO, SFT, or future: GRPO, GDPO)
5. **Compare** base model vs fine-tuned model (edit distance, LFR, perplexity)

```
OFFLINE METHODS (DPO, SFT):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │ ─▶ │   Compare   │
│  (any type) │    │     CFs     │    │     CFs     │    │ Train Data  │    │ DPO/SFT/... │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │                  │
                         ▼                  ▼                  ▼                  ▼                  ▼
                   ~40 CFs/entry      label_flipped,     pairs (DPO) or      improved CF       Edit Dist,
                   high temp          confidence,        chosen only (SFT)   model (LoRA)      LFR, PPL
                                      similarity

ONLINE METHODS (GRPO, GDPO - planned):
┌─────────────┐    ┌──────────────────────────────────────────────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Train (generate + evaluate + learn interleaved)    │ ─▶ │   Compare   │
└─────────────┘    └──────────────────────────────────────────────────────┘    └─────────────┘
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
    --output_dir ./results/counterfactuals_100e40c \
    --resume  # Optional: resume from previous progress
```

Generates diverse counterfactuals using high-temperature sampling. Target labels are distributed evenly across the generated counterfactuals.

**Parallel execution** (for large-scale runs):
```bash
python generate_counterfactuals.py \
    --datasets boolq \
    --max_entries 2000 \
    --cfs_per_entry 40 \
    --start_idx 0 --end_idx 100 \  # Process entries 0-99
    --shard_id 0 \                 # Shard identifier
    --output_dir ./results/counterfactuals_2000e40c
```

This creates `{dataset}_progress_shard0.jsonl`. Use `utils.merge_shard_files()` to combine after all shards complete.

**Output:** `results/counterfactuals_{suffix}/{dataset}_progress.jsonl` (or `_shard{N}.jsonl` if sharding)

### Stage 2: Evaluate Counterfactuals

```bash
python evaluate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --input_dir ./results/counterfactuals_100e40c \
    --output_dir ./results/counterfactuals_100e40c \
    --resume  # Optional: resume from previous progress
```

First computes the model's prediction on each **original (unedited) input**, then evaluates each counterfactual on:
- **Label Flipped**: Did the model's prediction CHANGE from the original? (not: did it match the target)
- **Confidence**: How confident is the model in the new label?
- **Semantic Similarity**: How minimal were the edits? (sentence embeddings)

**Important**: `label_flipped` compares against the model's OWN prediction on the original input (stored in `original_predictions.json`), not the ground truth label. This matches the evaluation approach in Stage 5.

**Output:** 
- `results/counterfactuals_{suffix}/{dataset}_original_predictions.json` - Model predictions on original inputs
- `results/counterfactuals_{suffix}/{dataset}_evaluated.jsonl` - CFs with evaluation scores

### Stage 3: Construct Training Data

```bash
python construct_dpo_pairs.py \
    --datasets boolq \
    --input_dir ./results/counterfactuals_100e40c \
    --output_dir ./results/dpo_pairs_boolq_100e40c \
    --max_pairs 2
```

Constructs preference pairs using **1-to-1 pairing** (see Design Choices below). The output can be used for both DPO and SFT training.

**Output:**
- `dpo_pairs.jsonl` - Minimal format for debugging/analysis
- `dpo_training.jsonl` - Full format for training (DPO uses both columns, SFT uses "chosen" only)

### Stage 4: Train Model

Choose a training method:

**Option A: DPO (Direct Preference Optimization)**
```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_1pair/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_1pair_b4_2ep \
    --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing \
    --use_wandb --wandb_run_name "dpo-1pair-b4-2ep-boolq"
```

**Option B: SFT (Supervised Fine-Tuning on chosen CFs only)**
```bash
python train_sft.py \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_1pair/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq_1pair_b4_1ep \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing \
    --use_wandb --wandb_run_name "sft-1pair-b4-1ep-boolq"
```

> **Note**: Prefer `--num_train_epochs` over `--max_steps`. Fixed step counts confound batch size with training duration (see RESULTS.md).

Both methods use QLoRA (4-bit quantization) and produce LoRA adapters.

**Output:** `results/{dpo,sft}_model_{dataset}_{suffix}/` (LoRA adapter)

### Stage 5: Evaluate Models

```bash
python evaluate_models.py \
    --dpo_model_path ./results/dpo_model_boolq_100e40c \  # or sft_model_*
    --datasets boolq \
    --split validation \
    --num_samples 50 \
    --cfs_per_entry 5 \
    --output_dir ./results/evaluation_boolq_100e40c \
    --base_eval_dir ./results/evaluation_boolq_prev \    # Optional: reuse base CFs for fair A/B
    --resume
```

Compares base model vs fine-tuned model (DPO, SFT, or any LoRA adapter) on held-out validation data:

| Metric | Description |
|--------|-------------|
| **Levenshtein Distance** | Character-level edit distance (lower = more minimal edits) |
| **Label Flip Rate (LFR)** | % of CFs where the model's prediction changed (higher = better) |
| **Perplexity (PPL)** | Fluency of generated text (lower = more natural) |

**Key details:**
- LFR compares against the BASE model's prediction on the **original input** (not ground truth)
- BASE model is the judge for ALL counterfactuals (ensures fair comparison)
- Use `--base_eval_dir` to reuse base model CFs when comparing different training configs
- For SNLI, same entry indices are used for both premise/hypothesis variants

**Output:** `results/evaluation_{dataset}_{suffix}/` with `eval_report.md`, `eval_summary.json`

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

### Training Data Construction: 1-to-1 Pairing with Hard Weighting

We use **1-to-1 pairing** to create strong contrasts between chosen and rejected examples:

```
Unified Score = flip_bonus + (confidence × similarity)

where flip_bonus = 100 if model's prediction changed from original, 0 otherwise
```

**Selection rules:**
1. **Chosen pool**: Only CFs where the model's prediction changed from original (`label_flipped=True`)
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

**Method-specific usage:**
- **DPO**: Uses both chosen and rejected columns for preference learning
- **SFT**: Uses only the "chosen" column (ignores rejected)

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
  "label_flipped": true,
  "predicted_label": "contradiction",
  "confidence": 0.95,
  "semantic_similarity": 0.87
}
```

Note: `label_flipped` indicates whether the model's prediction changed from the original (stored in entry's `original_prediction` field), not whether it matched the target label.

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
  "chosen_label_flipped": true,
  "pair_rank": 1
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

## Current Results

See `RESULTS.md` for full experimental results and `OVERVIEW.md` for a detailed analysis of all runs.

### Best results per method (LFR improvement over base)

| Method | BoolQ | SNLI-Premise | SNLI-Hypothesis |
|--------|:-----:|:------------:|:---------------:|
| Base | 44.9% | 51.5% | 46.3% |
| **DPO** (2-pair b4, 2ep) | **52.5%** (+9.3%) | **71.5%** (+17.4%) | — |
| **DPO** (1-pair b4, 2ep) | 50.7% (+8.1%) | 65.7% (+12.2%) | **54.3%** (+7.5%) |
| SFT (best, ~0.3-0.5ep) | 49.2% (+4.1%) | 54.5% (+2.7%) | 45.3% (-0.8%) |

**Key findings:**
- **DPO outperforms SFT** on all datasets, winning every head-to-head comparison at equal training duration
- **2 epochs** is optimal for DPO; SFT overfits quickly and peaks at ~0.3-0.5 epochs
- **Smaller batch (b4)** with more gradient updates outperforms larger batch (b16) at equal epochs
- **2-pair ≥ 1-pair** when trained long enough (the original "1-pair is better" finding was a training duration artifact — see RESULTS.md)
- SFT can hurt performance (negative LFR on SNLI), while DPO consistently improves

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
    TOP_P = 0.99
    TOP_K = 100
```

**Training configuration:**
- LoRA: rank=32, alpha=16, dropout=0.05
- Learning rate: 5e-6
- Training duration: `num_train_epochs` (preferred) or `max_steps` (legacy; confounds batch size with training duration)
- Batch size: 1-4 (with gradient accumulation for effective batch 4-16)

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
├── construct_dpo_pairs.py         # Stage 3: Build training data
│
│   # Training scripts (offline methods)
├── train_dpo.py                   # Stage 4: DPO training
├── train_sft.py                   # Stage 4: SFT training (chosen only)
│
│   # Training scripts (online methods - TODO)
│   # train_grpo.py                # Stage 4: GRPO (generation during training)
│   # train_gdpo.py                # Stage 4: GDPO (iterative gen→train cycles)
│
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
    ├── dpo_pairs_{dataset}_{suffix}_1pair/
    ├── dpo_model_{dataset}_{suffix}/
    ├── dpo_model_{dataset}_{suffix}_1pair/
    ├── sft_model_{dataset}_{suffix}/
    ├── sft_model_{dataset}_{suffix}_1pair/
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
| 2 | `counterfactuals_{suffix}/{dataset}_original_predictions.json` | Model's predictions on original inputs |
| 2 | `counterfactuals_{suffix}/{dataset}_evaluated.jsonl` | CFs with evaluation scores |
| 3 | `dpo_pairs_{dataset}_{suffix}/dpo_training.jsonl` | Training data (DPO uses both, SFT uses "chosen") |
| 4 | `{dpo,sft}_model_{dataset}_{suffix}/` | LoRA adapter weights |
| 5 | `evaluation_{dataset}_{suffix}/eval_report.md` | Human-readable comparison |
| 5 | `evaluation_{dataset}_{suffix}/eval_summary.json` | Structured metrics |

## Parallel Execution (Large-Scale Runs)

For large datasets (e.g., 2000+ entries), Stage 1 can be parallelized using SLURM array jobs:

```bash
# Step 1: Submit parallel Stage 1 jobs (20 shards × 100 entries each = 2000 total)
sbatch run_parallel_stage1.sh

# Step 2: After ALL Stage 1 jobs complete, merge and run Stages 2-5
sbatch run_stages_2to5.sh
```

**Configuration** (edit at top of scripts):
```bash
TOTAL_ENTRIES=2000
ENTRIES_PER_SHARD=400  # Creates 5 shards (#SBATCH --array=0-4)
CFS=40
DATASETS="boolq snli_premise snli_hypothesis"
```

**How it works:**
1. `run_parallel_stage1.sh`: Each array task processes a shard (e.g., entries 0-99, 100-199, etc.)
2. `run_stages_2to5.sh`: Merges shard files, then runs Stages 2-5 sequentially per dataset

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (optional, set in `~/.hf_token.env`)
- `WANDB_API_KEY`: Weights & Biases API key for logging (required for experiment tracking)

## Experiment Tracking with Weights & Biases

All training scripts are configured to log metrics to [Weights & Biases](https://wandb.ai) for experiment tracking and visualization.

**Configuration:**
- **Entity:** `cfg-dpo`
- **Project:** `Self-Align to Explain`
- **Dashboard:** https://wandb.ai/cfg-dpo/Self-Align%20to%20Explain

**Setup:**
```bash
# Option 1: Set environment variable
export WANDB_API_KEY="your-api-key"

# Option 2: Interactive login
wandb login
```

**Logged metrics:**
- Training loss curves (DPO loss, SFT loss)
- Learning rate schedule
- Gradient norms
- Training/eval rewards (DPO)

**Usage:**
Training scripts automatically use W&B with `--use_wandb`. Run names follow the pattern: `{method}-{pairs}-{batch}-{dataset}` (e.g., `dpo-1pair-b16-boolq`).

```bash
# Already enabled in all run scripts
python train_dpo.py ... --use_wandb --wandb_run_name "dpo-1pair-b4-boolq"
python train_sft.py ... --use_wandb --wandb_run_name "sft-2pair-b16-snli_premise"
```

---

## Future Work

### Benchmark Evaluation

After completing the comparison of post-training methods (SFT, DPO, GRPO, GDPO), we plan to evaluate all trained models on standard benchmarks to check for capability degradation (catastrophic forgetting):

**Planned benchmarks:**
- **MMLU** - Multitask Language Understanding
- **HellaSwag** - Commonsense reasoning
- **ARC** - AI2 Reasoning Challenge
- **TruthfulQA** - Truthfulness evaluation

**Goal:** Ensure that fine-tuning for counterfactual generation does not significantly degrade the model's general capabilities.

### Online Methods

Implement and compare online learning methods:
- **GRPO** - Group Relative Policy Optimization (generation during training)
- **GDPO** - Generalized DPO (iterative generation + training cycles)

## License

MIT
