# Self-Align to Explain
### Comparing Post-Training Methods for Counterfactual Generation

This thesis compares post-training self-alignment methods — SFT, DPO, SimPO, GRPO, and GDPO — for counterfactual example generation. We evaluate each method's ability to produce minimal, fluent edits that flip classifier predictions, measuring label flip rate, edit distance, and perplexity.

---

## Research Goal

This project investigates how different post-training methods affect a language model's ability to generate **minimal, label-flipping counterfactuals** for text classification tasks. All methods fine-tune **Qwen/Qwen2.5-7B-Instruct** (primary) with QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three datasets: **BoolQ** (edit passage to flip yes/no answer), **SNLI-Premise** and **SNLI-Hypothesis** (edit premise/hypothesis to change NLI relationship). Scale-ups to **Qwen2.5-14B-Instruct**, **Qwen2.5-3B-Instruct**, and **Llama-3.1-8B-Instruct** (cross-architecture) use the same recipe.


| Method    | Type      | Description                                                                         |
| --------- | --------- | ----------------------------------------------------------------------------------- |
| **SFT**   | Offline   | Supervised Fine-Tuning on successful counterfactuals only                           |
| **DPO**   | Offline   | Direct Preference Optimization on pre-generated preference pairs                    |
| **SimPO** | Offline   | Simple Preference Optimization — reference-free CPO variant with length normalization and margin γ |
| **GRPO**  | Online RL | Group Relative Policy Optimization with reward-driven generation during training    |
| **GDPO**  | Online RL | Group reward-Decoupled normalization Policy Optimization (per-reward normalization) |


---

## Results

Best ΔLFR (label flip rate improvement over base model) per method, using fair-base N=200 evaluation on **Qwen2.5-7B-Instruct**. Each cell is the best run with **parse rate ≥ 15% of base** (parse-collapse artifacts excluded). NED is reported as the median over non-trivial edits (NED>0) and PPL as the median:


| Method              | BoolQ       | SNLI-Premise | SNLI-Hypothesis |
| ------------------- | ----------- | ------------ | --------------- |
| **DPO**             | +22.1%      | +24.5%       | +23.2%          |
| SimPO               | **+28.9%**  | **+31.5%**   | **+31.1%**      |
| SFT                 | +9.4%       | +0.4%        | +1.8%           |
| GRPO (best variant) | +21.5%      | +27.4%       | +20.7%          |
| GDPO v6 (best ckpt) | −0.2%       | +16.5%       | +17.7%          |

On the canonical fair base, **SimPO leads 7B BoolQ** (+28.9%, 56% parse), ahead of DPO (+22.1%). GRPO with **lr=1e-5** jumps to +21.5% (single) / +19.2% (multi), up from +7.7% / +9.8% at the default lr — the same lr lift seen at 3B.

Scale-up highlights (fair-base N=200, see `RESULTS.md`):

| Model    | Best SNLI-P          | Best SNLI-H          | Best BoolQ           |
| -------- | -------------------- | -------------------- | -------------------- |
| 7B       | SimPO +31.5%         | SimPO +31.1%         | SimPO +28.9%         |
| 14B      | GRPO multi +30.1%    | GRPO multi +15.1%    | SimPO +7.4%          |
| 3B       | GRPO single +27.1%   | GDPO +15.1%          | GRPO single +11.0%   |
| Llama 8B | GRPO single +31.4%   | GRPO multi +8.6%     | GDPO +16.3%          |


**Key findings:**

- **SimPO leads all three 7B tasks** on the canonical fair base (+28.9% BoolQ, +31.5% SNLI-P, +31.1% SNLI-H), ahead of GRPO on SNLI and DPO on BoolQ
- **GRPO multi is the strongest online method at 14B** (SNLI-P +30.1%, SNLI-H +15.1%) and competitive at 7B SNLI-P (+27.4%)
- **Llama-8B GRPO single yields the highest single ΔLFR on SNLI-P** (+31.4%); GDPO leads Llama BoolQ (+16.3%)
- **Llama SNLI-H is the one cell where online RL decisively beats offline**: GRPO multi (+8.6%) clears base comfortably; offline methods are positive but modest (DPO +4.0%, SimPO +3.0%, SFT −1.7%)
- **GRPO lr=1e-5 is the key hyperparameter on BoolQ at every scale**: lifts 7B BoolQ GRPO single +7.7%→+21.5% and multi +9.8%→+19.2%, mirroring the same lift at 3B
- **GRPO reward design is scale-dependent**: multi ≥ single at 7B/14B SNLI, but single wins at 3B and on Llama SNLI-P
- **3B cannot learn BoolQ CFs from offline pairs**: DPO ≈ +0% and the stronger 2pair recipe is +1.7%; only online RL (GRPO single +11.0%) works there
- **Parse rate matters as much as LFR**: high-ΔLFR runs frequently come from parse collapse. Charts gate on parse ≥ 15% of base
- **Metric hygiene**: PPL/NED are reported as medians; NED excludes trivial NED=0 "non-edits"
- **No general-capability degradation from CF fine-tuning**: across all 72 best adapters, MMLU and ANLI are unchanged vs base — see `BENCHMARKS.md`
- **Fair comparison protocol**: within each model×dataset, base counterfactual verdicts are frozen once and reused across all methods (identical base LFR for every ΔLFR in a cell)

See `RESULTS.md` for complete results, `OVERVIEW.md` for detailed analysis, and `BENCHMARKS.md` for MMLU/ANLI.

---

## Pipeline Overview

**DPO / SimPO / SFT / KTO** use pre-generated training data:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Generate   │ ─▶ │  Evaluate   │ ─▶ │  Construct  │ ─▶ │    Train    │ ─▶ │   Compare   │
│  (any type) │    │     CFs     │    │     CFs     │    │ Train Data  │    │DPO/SFT/SimPO│    │   Models    │
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

### Stage 1-3: Generate Training Data (DPO/SimPO/SFT/KTO only)

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

**SimPO** (reference-free; requires `beta` and `simpo_gamma`):

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq/dpo_training.jsonl \
    --output_dir ./results/simpo_model_boolq \
    --loss_type simpo --beta 3.0 --simpo_gamma 0.5 \
    --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing
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

**Orchestrated pipelines** (recommended for full runs):

```bash
# 7B / 14B: full pipeline (generate → build pairs → train → eval)
MODEL=Qwen/Qwen2.5-14B-Instruct TAG=qwen25_14b ./submit_model_pipeline.sh

# 3B: equivalent pipeline
./submit_3b_pipeline.sh

# Llama 3.1 8B: cross-architecture pipeline
./submit_llama8b_pipeline.sh

# Post-hoc fair eval for any adapter
ADAPTER_DIR=results/my_model DATASET=boolq ./run_eval_14b_fair.sh
ADAPTER_DIR=results/my_model DATASET=boolq ./run_eval_3b_fair.sh
ADAPTER_DIR=results/my_model DATASET=boolq ./run_eval_llama8b_fair.sh
```

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

Chosen pool: only label-flipping CFs (sorted best-first). Rejected pool: all valid CFs (sorted worst-first). Pairs are formed 1-to-1: best chosen ↔ worst rejected, 2nd-best ↔ 2nd-worst. This ensures each pair has maximum contrast. DPO/SimPO use both columns; SFT and KTO use only the chosen column.

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
├── train_dpo.py                   # Stage 4: DPO / SimPO training (also supports KTO via --loss_type kto)
├── train_sft.py                   # Stage 4: SFT training
├── train_grpo.py                  # Stage 4: GRPO (+ reward functions)
├── train_gdpo.py                  # Stage 4: GDPO (per-reward normalization)
├── gdpo_trainer.py                # GDPOTrainer subclass
│
├── evaluate_models.py             # Stage 5: Model comparison
│
├── submit_model_pipeline.sh       # Orchestrate full pipeline (7B/14B)
├── submit_3b_pipeline.sh          # Orchestrate full pipeline (3B)
├── run_dpo_sweep.sh               # DPO / SimPO / KTO Slurm job
├── run_grpo.sh                    # GRPO Slurm job
├── run_gdpo.sh                    # GDPO Slurm job
├── run_sft_train_only.sh          # SFT training Slurm job (no inline eval)
├── run_eval_14b_fair.sh           # Fair eval for any 14B adapter
├── run_eval_3b_fair.sh            # Fair eval for any 3B adapter
├── submit_llama8b_pipeline.sh     # Orchestrate full pipeline (Llama 8B)
├── run_eval_llama8b_fair.sh       # Fair eval for any Llama 8B adapter
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

- **Consistent `<edit>`-tag training targets (targeted retrain)** — offline methods (DPO/SimPO/SFT) currently train on the bare edited text with the `<edit>…</edit>` wrapper stripped, while the eval parser and online GRPO/GDPO rewards require the tags. The **dual-parse diagnostic** (`PARSE_TAG_ISSUE.md §9`) quantified the impact: the **featured matrix is unaffected** (0/12 pick changes), so no global retrain is needed — but the gate wrongly excluded genuinely strong **Llama offline** runs. Planned fix is a **targeted retrain with `<edit>`-wrapped targets** of just **Llama SNLI-H DPO, Llama SNLI-H SimPO, and Llama SNLI-P DPO** (the cells that re-qualify at +6.8 to +9.9% under honest compliance), re-evaluated with the unchanged strict parser
- **Benchmark evaluation** — ✅ done: MMLU + ANLI across all 76 models (4 base + 72 adapters); no capability degradation (ΔMMLU mean +0.0pp, ΔANLI +0.1pp). See `BENCHMARKS.md`
- **Parse-rate-aware reporting** — the "best non-degraded run" selection (parse ≥ 15% of base) is documented in `OVERVIEW.md § Parse-gate exclusions`; could be formalized into a combined LFR×parse quality score

## License

MIT
