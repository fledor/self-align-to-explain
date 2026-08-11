# Self-Align to Explain
### Comparing Post-Training Methods for Counterfactual Generation

Code and results for the thesis *Self-Align to Explain*. It compares five post-training
methods (**SFT, DPO, SimPO, GRPO, GDPO**) for fine-tuning an instruction-tuned LLM to
generate **minimal, label-flipping counterfactuals** for text classification.

The task: given an input the model classifies as label *A*, produce a minimal edit that
makes the same model predict a different label *B*. Each model judges its own
counterfactuals, so a flip is measured against the base model's prediction, not the
ground truth. Metrics are label flip rate (LFR), normalized edit distance (NED), and
perplexity (PPL). Tasks: **BoolQ** (edit the passage to flip a yes/no answer),
**SNLI-Premise** and **SNLI-Hypothesis** (edit one side to change the NLI relation).
Base models: **Qwen2.5-3B/7B/14B-Instruct** and **Llama-3.1-8B-Instruct**. All methods
train QLoRA adapters (4-bit NF4, LoRA r=32, α=16). Learning rate and a few method
hyperparameters are tuned per model×dataset cell; the winning configuration for each
cell is in [`BEST_CONFIGS.md`](reports/BEST_CONFIGS.md).

| Method    | Type      | Description                                                                          |
| --------- | --------- | ------------------------------------------------------------------------------------ |
| **SFT**   | Offline   | Supervised fine-tuning on successful counterfactuals only                            |
| **DPO**   | Offline   | Direct Preference Optimization on pre-generated preference pairs                     |
| **SimPO** | Offline   | Reference-free preference optimization with length normalization and margin γ        |
| **GRPO**  | Online RL | Group Relative Policy Optimization with reward-driven generation during training     |
| **GDPO**  | Online RL | GRPO variant with per-reward (decoupled) group normalization for multi-reward setups |

---

## Results

ΔLFR (label-flip-rate improvement over the same model's base, in percentage points) for
the best parse-gated run per cell; fair frozen-base evaluation, N=200 prompts × 10
counterfactuals. GRPO shows whichever reward construction (composite or decomposed) wins
the cell. Bold = best method in the cell. Full metrics per run: `reports/frozen_metrics.json`
and `reports/results_charts.html`.

| Model | Method | BoolQ | SNLI-Premise | SNLI-Hypothesis |
| --- | --- | --- | --- | --- |
| 3B | SFT | +1.2% | +4.4% | +3.1% |
|  | DPO | +1.7% | +11.0% | +7.3% |
|  | SimPO | +7.5% | +25.8% | +6.7% |
|  | GRPO | **+11.0%** | **+27.1%** | +8.2% |
|  | GDPO | +1.3% | +17.7% | **+15.1%** |
| 7B | SFT | +9.4% | +0.4% | +1.8% |
|  | DPO | +22.1% | +24.5% | +23.2% |
|  | SimPO | **+28.9%** | **+31.5%** | **+31.1%** |
|  | GRPO | +21.5% | +27.4% | +22.1% |
|  | GDPO | −0.2% | +16.5% | +17.7% |
| 14B | SFT | +0.5% | +0.0% | +3.8% |
|  | DPO | +2.7% | +15.5% | +6.7% |
|  | SimPO | **+7.4%** | +22.3% | +13.7% |
|  | GRPO | +2.5% | **+30.1%** | **+15.1%** |
|  | GDPO | +5.7% | +26.5% | +8.7% |
| Llama-8B | SFT | −0.9% | +1.9% | −1.7% |
|  | DPO | +11.5% | +2.4% | +4.0% |
|  | SimPO | +9.0% | +19.0% | +3.0% |
|  | GRPO | +12.5% | **+31.4%** | **+8.6%** |
|  | GDPO | **+16.3%** | +11.2% | +6.0% |

**Key findings:**

- **There is no universal winner.** The best method changes with model and task, and
  per-cell 95% bootstrap confidence intervals (median half-width ±3.7 pp) overlap the
  runner-up in 11 of 12 cells, so individual cell wins should be read as ties.
- **GRPO is the most consistent method.** It has the best average rank across the 12
  cells, is statistically indistinguishable from the cell's best method everywhere, and
  is the only method whose confidence interval clears zero in all 12 cells.
- **Offline dominance is a 7B phenomenon.** SimPO sweeps all three 7B tasks, but away
  from 7B the online methods win 8 of 9 cells. SFT is consistently weak and is the only
  method the rank-based tests separate from the rest.
- **Tuning often outweighs the objective.** Learning rate swings GRPO on BoolQ from
  ~+8–10% to +21.5% at 7B (and similarly at 3B), only a higher rate lifts DPO above base
  at 14B and on Llama, the SimPO margin is decisive at 3B, and online runs peak at
  0.25–0.75 epochs and are early-stopped by checkpoint sweep. The winning configuration
  per cell is recorded in [`BEST_CONFIGS.md`](reports/BEST_CONFIGS.md).
- **Metric hygiene matters.** High-ΔLFR runs can be parse-collapse artifacts, so featured
  runs must retain a parse rate ≥ 15% of base; NED is the median over genuine edits
  (NED > 0) and PPL the median; base verdicts are frozen so every method in a cell is
  compared against the identical base LFR.
- **Counterfactual fine-tuning costs no general capability.** Across all 72 released
  adapters, MMLU stays within ±0.4 pp and ANLI within +1.2/−0.7 pp of base — see
  [`BENCHMARKS.md`](reports/BENCHMARKS.md).

Per-run metrics for all featured runs are in `reports/frozen_metrics.json`/`.tsv` and
the interactive `reports/results_charts.html`; qualitative examples of the edits each
method produces are in `reports/qualitative/`.

---

## Pipeline Overview

**SFT / DPO / SimPO** use pre-generated training data:

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
┌─────────────┐    ┌──────────────────────────────────────────────┐    ┌─────────────┐
│   Dataset   │ ─▶ │  Train (generate + reward + learn per step)  │ ─▶ │   Compare   │
└─────────────┘    └──────────────────────────────────────────────┘    └─────────────┘
```

---

## Usage

The exact command behind every featured run is in [`BEST_CONFIGS.md`](reports/BEST_CONFIGS.md);
the commands below show the shape of each stage.

**Stages 1–3 — generate training data** (offline methods only):

```bash
python generate_counterfactuals.py \
    --datasets boolq snli_premise snli_hypothesis \
    --max_entries 2000 --cfs_per_entry 40 \
    --output_dir ./results/counterfactuals_2000e40c --resume

python evaluate_counterfactuals.py \
    --datasets boolq --input_dir ./results/counterfactuals_2000e40c \
    --output_dir ./results/counterfactuals_2000e40c --resume

python construct_dpo_pairs.py \
    --datasets boolq --input_dir ./results/counterfactuals_2000e40c \
    --output_dir ./results/dpo_pairs_boolq --max_pairs 2
```

Stages 1–2 support sharded parallel execution (`--start_idx`, `--end_idx`, `--shard_id`).
Data is generated per base model, since each model learns from its own counterfactuals.

**Stage 4 — train:**

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq \
    --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --use_4bit --bf16 --gradient_checkpointing
```

- **SimPO**: same script with `--loss_type simpo --beta 3.0 --simpo_gamma 0.5` (reference-free).
- **SFT**: `train_sft.py` on the same data (uses the *chosen* column only; overfits
  quickly — the featured runs stop after ~200 steps).
- **GRPO**: `train_grpo.py --dataset_name boolq --num_generations 16` — online, no
  pre-generated data; `--multi_reward` selects the decomposed reward (flip, similarity,
  flip-gated confidence, format) instead of the composite one.
- **GDPO**: `train_gdpo.py` — decomposed rewards with per-reward group normalization;
  featured configuration uses `--generation_batch_size 128 --conditioned_rewards
  --beta 0.0005 --epsilon_high 0.28`.

**Stage 5 — evaluate:**

```bash
python evaluate_models.py \
    --model_path ./results/dpo_model_boolq \
    --datasets boolq --split validation \
    --num_samples 200 --cfs_per_entry 10 \
    --output_dir ./results/dpo_model_boolq/eval_boolq_200 \
    --base_eval_dir ./results/base_eval_boolq_200 --resume
```

Compares base vs fine-tuned model on LFR, NED, and PPL. `--base_eval_dir` reuses the
base model's counterfactuals and frozen verdicts, so every adapter of a model×dataset
cell is scored against the identical base.

---

## Design Choices

### Preference pairs (offline methods)

Counterfactuals are ranked by a unified score with validity strictly first:

```
score = flip_bonus + (confidence × similarity)     flip_bonus = 100 if label flipped else 0
```

The *chosen* pool holds only label-flipping CFs (best-first); the *rejected* pool holds
all well-formed CFs (worst-first). Pairs are formed by matching ranks from opposite ends
— best chosen with worst rejected — which maximizes contrast, and the rejected member
must target the same label as the chosen one. Up to two pairs per entry are built; one
vs two pairs is a tuned setting. DPO/SimPO use both columns, SFT the chosen column only.

### Rewards (online methods)

The **composite** reward collapses the same signals into one scalar,
`r = 1[flip] + confidence × similarity`. The **decomposed** reward keeps four signals
separate — flip, similarity, flip-gated confidence, and a format reward for well-formed
`<edit>` tags — combined with equal weights. GDPO normalizes each reward within the
group before combining (its only difference from GRPO), and its featured configuration
adds a small KL penalty (β=0.0005), DAPO asymmetric clipping (ε_high=0.28), and a
generation batch of eight groups so per-reward statistics are meaningful.

### Generation and evaluation protocol

- Generation (training data and eval): temperature 1.2, top-p 0.99, top-k 100, unique
  seed per sample, post-generation deduplication; edits must be wrapped in
  `<edit>...</edit>` tags.
- **Frozen fair base**: base CFs are generated once per model×dataset and their verdicts
  frozen, so base LFR is identical for every method in a cell.
- **LFR** is measured against the base model's own prediction on the original input, not
  the ground-truth label.
- **Parse gate**: featured runs need a parse rate ≥ 15% of base (collapse screen).
- **Medians**: NED over non-trivial edits (NED > 0); PPL as median.
- Hardware caveat: bf16 inference varies 2–5 pp across GPU types; comparisons should run
  on the same node.

---

## Released Adapters

The best adapter for each of the 72 model×dataset×method cells is released on Hugging
Face: [`fledor/self-align-to-explain-adapters`](https://huggingface.co/fledor/self-align-to-explain-adapters).
Each subfolder (`<base>_<dataset>_<method>`, e.g. `qwen7b_snli_premise_simpo`) contains
the LoRA adapter, tokenizer, training config, and evaluation metrics; adapters load on
top of their stock base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "fledor/self-align-to-explain-adapters",
                                  subfolder="qwen7b_snli_premise_simpo")
```

---

## Project Structure

```
self-align-to-explain/
├── config.py                      # Central configuration
├── dataset_registry.py            # Dataset registry (extensible)
├── prompts.py                     # Prompt templates (verbatim, as used in the thesis)
├── utils.py                       # Shared utilities
│
├── generate_counterfactuals.py    # Stage 1: Generate CFs
├── evaluate_counterfactuals.py    # Stage 2: Evaluate CFs
├── construct_dpo_pairs.py         # Stage 3: Build training data
├── train_dpo.py                   # Stage 4: DPO / SimPO training
├── train_sft.py                   # Stage 4: SFT training
├── train_grpo.py                  # Stage 4: GRPO (+ reward functions)
├── train_gdpo.py                  # Stage 4: GDPO (per-reward normalization)
├── gdpo_trainer.py                # GDPOTrainer subclass
├── evaluate_models.py             # Stage 5: Model comparison (frozen fair base)
│
├── reports/
│   ├── BEST_CONFIGS.md            # Exact winning configuration per cell (72 runs)
│   ├── frozen_metrics.json / .tsv # Frozen fair-eval metrics behind all tables
│   ├── BENCHMARKS.md              # MMLU / ANLI capability check
│   ├── results_charts.html        # Interactive results visualization
│   └── qualitative/               # Qualitative counterfactual examples (7B)
│
├── data/                          # Datasets (tracked)
└── results/                       # Outputs (gitignored)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Environment variables:** `HF_TOKEN` (HuggingFace token); optionally `WANDB_API_KEY`
if training with `--use_wandb`.

## License

MIT — see [`LICENSE`](LICENSE).
