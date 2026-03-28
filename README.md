# Self-Align to Explain
### Comparing Post-Training Methods for Counterfactual Generation

This thesis compares post-training self-alignment methods — SFT, DPO, GRPO, and GDPO — for counterfactual example generation. We evaluate each method's ability to produce minimal, fluent edits that flip classifier predictions, measuring label flip rate, edit distance, and perplexity.

---

## Research Goal

This project investigates how different post-training methods affect a language model's ability to generate **minimal, label-flipping counterfactuals** for text classification tasks. All methods fine-tune Qwen/Qwen2.5-7B-Instruct with QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three datasets: **BoolQ** (edit passage to flip yes/no answer), **SNLI-Premise** and **SNLI-Hypothesis** (edit premise/hypothesis to change NLI relationship).


| Method   | Type      | Description                                                                         |
| -------- | --------- | ----------------------------------------------------------------------------------- |
| **SFT**  | Offline   | Supervised Fine-Tuning on successful counterfactuals only                           |
| **DPO**  | Offline   | Direct Preference Optimization on pre-generated preference pairs                    |
| **GRPO** | Online RL | Group Relative Policy Optimization with reward-driven generation during training    |
| **GDPO** | Online RL | Group reward-Decoupled normalization Policy Optimization (per-reward normalization) |


---

## Results

Best ΔLFR (label flip rate improvement over base model) per method, using fair-base N=200 evaluation:


| Method              | BoolQ      | SNLI-Premise | SNLI-Hypothesis |
| ------------------- | ---------- | ------------ | --------------- |
| **DPO**             | **+10.5%** | +17.3%       | **+21.0%**      |
| SFT                 | +1.8%      | -1.7%        | +1.7%           |
| GRPO (g16 mv2)      | +1.7%      | **+27.5%**   | +15.1%          |
| GDPO v6 (best ckpt) | +1.1%      | +18.2%       | **+18.7%**      |


**Key findings:**

- **GRPO mv2 g16 is the best overall method**: +27.5% on SNLI-P, +15.1% on SNLI-H, +1.7% on BoolQ
- **GDPO v6 with early stopping surpasses GRPO on SNLI-H**: +18.7% vs +15.1%
- **DPO leads on BoolQ** (+10.5%); all online RL methods struggle on BoolQ (long passages, format collapse)
- **Learning rate is the dominant factor for GDPO**: lr=1e-6 gives near-zero ΔLFR; lr=5e-6 unlocks +15.7-18.7%
- **Conditioned rewards + KL penalty > more groups**: GDPO v6 (8 groups) +15.7% > paper-match (32 groups) +7.6%
- **Early stopping is dataset-dependent**: GDPO peaks at ~0.25ep on SNLI-P, ~0.75ep on SNLI-H

See `RESULTS.md` for complete results and `OVERVIEW.md` for detailed analysis.

---

## Pipeline Overview

**DPO / SFT** use pre-generated training data:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │ ─▶ │   Compare   │
│  (any type) │    │     CFs     │    │     CFs     │    │ Train Data  │    │  DPO / SFT  │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │                  │
                         ▼                  ▼                  ▼                  ▼                  ▼
                   ~40 CFs/entry      label_flipped,     pairs (DPO) or      improved CF       Edit Dist,
                   high temp          confidence,        chosen only (SFT)   model (LoRA)      LFR, PPL
                                      similarity
```

**GRPO / GDPO** skip data pre-generation — generation + reward + learning happen within each training step:

```
┌─────────────┐    ┌──────────────────────────────────────────────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Train (generate + reward + learn per step)         │ ─▶ │   Compare   │
└─────────────┘    └──────────────────────────────────────────────────────┘    └─────────────┘
```

---

## Usage

### Stage 1-3: Generate Training Data (DPO/SFT only)

```bash
# Generate counterfactuals
python generate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --max_entries 2000 --cfs_per_entry 40 \
    --output_dir ./results/counterfactuals_2000e40c --resume

# Evaluate counterfactuals (label flip, confidence, similarity)
python evaluate_counterfactuals.py \
    --datasets boolq --input_dir ./results/counterfactuals_2000e40c \
    --output_dir ./results/counterfactuals_2000e40c --resume

# Construct preference pairs
python construct_dpo_pairs.py \
    --datasets boolq --input_dir ./results/counterfactuals_2000e40c \
    --output_dir ./results/dpo_pairs_boolq --max_pairs 2
```

Supports parallel execution via `--start_idx`, `--end_idx`, `--shard_id` for large-scale runs.

### Stage 4: Train

**DPO:**

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_1pair_b4_2ep \
    --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing \
    --use_wandb --wandb_run_name "dpo-boolq-1p-b4-2ep"
```

**SFT** (uses "chosen" column only; overfits quickly, ~0.2-0.5 epochs recommended):

```bash
python train_sft.py \
    --dataset_path ./results/dpo_pairs_boolq/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq \
    --num_train_epochs 0.5 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing
```

**GRPO** (online RL — generates counterfactuals during training):

```bash
python train_grpo.py \
    --dataset_name boolq --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16 \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --num_generations 16 --max_completion_length 512 --temperature 1.2 \
    --multi_reward \
    --use_4bit --bf16 --gradient_checkpointing
```

Three reward configurations: **single v2** (default, composite `flip + confidence * sim`), **multi-reward v2** (`--multi_reward`, four decomposed rewards with gated confidence and format reward), and multi-reward v1 (historical, collapsed). Additional flags for fair comparison with GDPO: `--conditioned_rewards`, `--epsilon_high 0.28`, `--beta 0.0005`.

**GDPO** (per-reward normalization — extends GRPO for multi-reward settings):

```bash
python train_gdpo.py \
    --dataset_name snli_premise --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_premise_v6 \
    --num_train_epochs 1 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --num_generations 16 --generation_batch_size 128 --max_completion_length 512 \
    --learning_rate 5e-6 --beta 0.0005 --epsilon_high 0.28 --conditioned_rewards \
    --use_4bit --bf16 --gradient_checkpointing
```

Best config (v6): `generation_batch_size=128` (8 groups for meaningful batch normalization), `--conditioned_rewards` (similarity gated on flip), `--beta 0.0005` (KL penalty), `--epsilon_high 0.28` (DAPO asymmetric clipping). See `OVERVIEW.md` section 5 for the full GDPO analysis.

### Stage 5: Evaluate

```bash
python evaluate_models.py \
    --model_path ./results/dpo_model_boolq_1pair_b4_2ep \
    --datasets boolq --split validation \
    --num_samples 200 --cfs_per_entry 10 \
    --output_dir ./results/dpo_model_boolq_1pair_b4_2ep/eval_boolq_200 \
    --base_eval_dir ./results/base_eval_boolq_200 --resume
```

Compares base model vs fine-tuned model on **label flip rate** (LFR), **normalized edit distance** (NED), and **perplexity** (PPL). The base model judges all counterfactuals. Use `--base_eval_dir` to reuse base model CFs for fair A/B comparison across configs.

---

## Design Choices

### Preference Pair Construction

We use **1-to-1 pairing** with hard weighting to create maximum-contrast preference pairs:

```
Unified Score = flip_bonus + (confidence × similarity)
where flip_bonus = 100 if label flipped, 0 otherwise
```

Chosen pool: only label-flipping CFs (sorted best-first). Rejected pool: all valid CFs (sorted worst-first). Pairs are formed 1-to-1: best chosen ↔ worst rejected, 2nd-best ↔ 2nd-worst. This ensures each pair has maximum contrast. DPO uses both columns; SFT uses only chosen.

### Generation Diversity

High-temperature sampling (T=1.2, top_p=0.99, top_k=100) with unique random seeds per generation and post-generation deduplication. CFs must have valid `<edit>...</edit>` tags; invalid responses are discarded.

### Evaluation

- **Fair-base comparison**: base model CFs generated once, reused across all model evaluations via `--base_eval_dir`
- **LFR**: measured against the base model's own prediction on the original input (not ground truth)
- **Hardware caveat**: bf16 inference varies 2-5pp across GPU types; run comparisons on the same node

---

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
├── train_dpo.py                   # Stage 4: DPO training
├── train_sft.py                   # Stage 4: SFT training
├── train_grpo.py                  # Stage 4: GRPO (+ reward functions)
├── train_gdpo.py                  # Stage 4: GDPO (per-reward normalization)
├── gdpo_trainer.py                # GDPOTrainer subclass
│
├── evaluate_models.py             # Stage 5: Model comparison
│
├── RESULTS.md                     # Full evaluation results
├── OVERVIEW.md                    # Detailed method analysis
├── results_charts.html            # Interactive results visualization
│
├── data/                          # Downloaded datasets (tracked)
├── results/                       # Output files (gitignored)
├── logs/                          # SLURM logs (gitignored)
└── model_cache/                   # Cached weights (gitignored)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Environment variables:**

- `HF_TOKEN`: HuggingFace token (set in `~/.hf_token.env`)
- `WANDB_API_KEY`: Weights & Biases key (set in `~/.wandb_token.env`)

All training scripts support `--use_wandb` with `--wandb_run_name`. Dashboard: [https://wandb.ai/cfg-dpo/Self-Align%20to%20Explain](https://wandb.ai/cfg-dpo/Self-Align%20to%20Explain)

---

## Future Work

- **Fair comparison analysis** — GRPO-fair runs (GRPO with GDPO's stabilization tricks) pending to isolate per-reward normalization effect
- **BoolQ** — All online RL methods struggle; DPO remains best (+10.5%). Longer sequences, SFT warm-start, and NED penalties tried without meaningful improvement
- **Benchmark evaluation** — Check for capability degradation on MMLU, HellaSwag, ARC
- **Optimal early stopping** — Checkpoint sweeps needed across all methods and datasets

## License

MIT