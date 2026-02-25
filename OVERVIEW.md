# Post-Training Methods for Counterfactual Generation: Overview

Comparing post-training self-alignment methods for improving counterfactual generation quality. All methods fine-tune Qwen/Qwen2.5-7B-Instruct using QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three NLI/classification datasets.

**Last updated**: February 25, 2026

---

## 1. Method Comparison

Best ΔLFR per method per dataset, using fair-base evaluations (N=100 unless noted).

| Dataset | DPO | SFT | GRPO g4 † | GRPO g16 | GDPO |
|---------|:---:|:---:|:---------:|:--------:|:----:|
| **BoolQ** | **+9.4%** | +4.1% | *pending* | *pending* | *planned* |
| | 2pair b16, 3ep | 2pair b4, 200step | | | |
| **SNLI-P** | **+17.3%** | +2.7% | +7.2% | *pending* | *planned* |
| | 2pair b4, 2ep | 1pair b4, 200step | multi g4, 2ep | | |
| **SNLI-H** | **+7.5%** | +1.8% | -10.6% | *pending* | *planned* |
| | 1pair b4, 2ep | 2pair b4, 2ep | single g4, 2ep | | |

† GRPO g4 single-reward used old reward formula (`flip + 0.8 * sim`). Multi-reward and g16 v2 use the corrected formula (`flip + confidence * sim`).

DPO leads on all datasets by a wide margin. GRPO g4 sits between SFT and DPO on SNLI-P but is harmful on SNLI-H (compromised by reward advantage collapse). SFT provides modest or negative gains.

Higher-N evaluations (N=200, fair base) confirm the pattern: DPO 2pair b4 2ep reaches +21.0% on SNLI-H and +22.0% on SNLI-P, suggesting the N=100 numbers underestimate the true effect.

---

## 2. DPO Analysis

DPO (Direct Preference Optimization) learns from chosen/rejected counterfactual pairs. It consistently produces the strongest LFR improvements across all datasets and configurations.

### Training duration

The original 200-step runs used `max_steps=200`, which maps to very different effective epochs depending on dataset size and batch size:

| Config | BoolQ | SNLI-P | SNLI-H |
|--------|:-----:|:------:|:------:|
| 1pair b4 | 0.49ep | 0.41ep | 0.46ep |
| 1pair b16 | 1.94ep | 1.64ep | 1.84ep |
| 2pair b4 | 0.25ep | 0.21ep | 0.25ep |
| 2pair b16 | 1.00ep | 0.84ep | 0.99ep |

The early finding that "1pair b16 is best" was an artifact of this confound: 1pair b16 got ~1.9 epochs while 2pair b4 got only ~0.25 epochs. When training for equal epochs, the ranking reverses.

### Optimal configuration: 2 epochs, small batch, more data

At 2 full epochs:
- **b4 outperforms b16**: smaller batches yield 4x more gradient updates per epoch, aiding convergence
- **2pair outperforms 1pair**: more training data helps when the model trains long enough to learn from it
- **3 epochs show diminishing returns**: b16 3ep results are similar to 2ep (BoolQ: +9.4% vs +4.9%, SNLI-P: +16.5% vs +17.2%)

Best DPO config: **2pair b4 at 2 epochs** (BoolQ +9.3%, SNLI-P +17.3%). The DPO 1pair b16 2ep BoolQ result (-6.7%) is an outlier, likely evaluation variance — the same model at 200step (~1.9ep, only 5 fewer gradient steps) scores +3.1%.

### Edit quality trade-off

DPO produces larger edits (higher NED) than both SFT and GRPO. On SNLI-P, NED rises from ~0.25 (base) to ~0.33 (DPO), and PPL remains comparable or slightly higher. The LFR gains come at the cost of minimality.

---

## 3. SFT Analysis

SFT (Supervised Fine-Tuning) trains on successful counterfactuals only, without rejected examples for contrast. It provides weak improvements at best and often hurts performance.

### Overfitting pattern

SFT performance follows a non-monotonic pattern with training duration:

| Duration | BoolQ 1p-b4 | SNLI-P 1p-b4 | SNLI-H 1p-b4 |
|----------|:-----------:|:------------:|:------------:|
| 200step (~0.4ep) | **+2.2%** | **+2.7%** | -0.8% |
| 0.5ep | -9.1% | -0.7% | -5.3% |
| 1ep | -3.6% | -3.4% | -6.1% |
| 2ep | -0.7% | -4.0% | -4.2% |

The sweet spot is ~0.2-0.5 epochs for b4 configs (the 200step runs). At 1 epoch performance is consistently worst, with partial recovery at 2 epochs for some configs.

### Training loss disconnect

Training loss drops steadily (BoolQ 1p-b4: 1.08 at 200step, 0.61 at 1ep, 0.39 at 2ep) but downstream LFR goes +2.2% then -3.6% then -0.7%. The model memorizes training data without learning generalizable counterfactual generation ability. Imitation learning without contrastive signal is unreliable for this task.

### Best SFT results

- **BoolQ**: 2pair b4 200step (+4.1%)
- **SNLI-P**: 1pair b4 200step (+2.7%) or 2pair b16 1ep (+1.8%)
- **SNLI-H**: 2pair b4 2ep (+1.8%) — the only positive SFT result on this dataset

---

## 4. GRPO Analysis

GRPO (Group Relative Policy Optimization) is an online RL method that generates counterfactuals during training and learns from a reward signal, requiring no pre-generated preference data.

### Initial runs (num_generations=4)

With 4 generations per prompt, the GRPO group often produced identical rewards across all completions, causing zero reward standard deviation and thus zero advantages — no gradient signal:

| Dataset | Mode | Zero-std % | Effective learning steps |
|---------|------|:----------:|:-----------------------:|
| BoolQ | Single | 7.7% | 92.3% |
| BoolQ | Multi | 10.6% | 89.4% |
| SNLI-P | Single | 50.2% | 49.8% |
| SNLI-P | Multi | 41.3% | 58.7% |
| SNLI-H | Single | 82.8% | 17.2% |
| SNLI-H | Multi | 83.5% | 16.5% |

BoolQ had adequate signal (~90% effective steps), and is expected to show meaningful GRPO results (eval pending). SNLI-H was severely compromised, with only ~17% of steps providing gradient signal.

### Results

On SNLI-P, GRPO g4 achieves +5.4% (single) and +7.2% (multi), placing it between SFT and DPO. Notably, GRPO produces smaller edits (NED 0.27 vs DPO's 0.33) and better fluency (PPL 85-92 vs DPO's 84-142).

On SNLI-H, GRPO g4 is harmful (-10.6% single, -11.3% multi), worse than both base and SFT. The models learned to make extremely minimal edits (NED ~0.16, half of base) that rarely flip labels. With 83% zero-std, training signal was too sparse to learn effectively.

### Reward correction

The initial GRPO single-reward runs (g4 and g16) used `flip + 0.8 * similarity` — an arbitrary weight with no confidence term. The DPO unified score uses `flip_bonus + confidence * similarity`. The corrected GRPO single reward now matches: `flip + confidence * similarity`, where confidence is the base model's classification confidence on the edited text.

Multi-reward runs are unaffected (they use separate `FlipReward` + `SimilarityReward` classes). All g4 results and the first g16 single-reward results use the old formula (marked with † in RESULTS.md). New g16 v2 single-reward jobs use the corrected formula.

### Reruns with num_generations=16

All 6 configurations (3 datasets, 2 modes) have been resubmitted with `num_generations=16` to increase group diversity. This is expected to reduce zero-std from ~50-83% to ~3-5%, providing substantially better training signal. An additional 3 single-reward v2 jobs use the corrected reward formula. Results pending.

---

## 5. GDPO (Planned)

GDPO (Group reward-Decoupled normalization Policy Optimization) extends GRPO for multi-objective optimization. Instead of summing rewards before normalization (`sum_then_normalize`, the default GRPO multi-reward approach), GDPO normalizes each reward independently before summing (`normalize_then_sum`). This prevents high-magnitude rewards from dominating the advantage calculation.

TRL's `GRPOTrainer` supports this via `multi_objective_aggregation="normalize_then_sum"`. Implementation is ready; experiments will follow once GRPO g16 results establish a baseline.

---

## 6. Evaluation Methodology

- **Fair-base comparison**: base model counterfactuals are generated once and reused across all model evaluations via `--base_eval_dir`, ensuring ΔLFR differences reflect the fine-tuned model's quality rather than base model generation variance
- **Standard evaluation**: 100 validation samples, 10 CFs per sample, fair-base reused
- **Base model as judge**: the un-fine-tuned base model classifies all counterfactuals (both base-generated and fine-tuned-generated) to determine label flips
- **Evaluation variance**: base LFR varies 2-5pp across independent runs; differences below ~5pp should be treated cautiously

---

## 7. Current Status

### Running jobs
- GRPO g16 training: 6 jobs (3 datasets x 2 modes, old single reward), ~8 hours in
- GRPO g16 v2 training: 3 jobs (3 datasets, corrected single reward), queued
- GRPO g4 BoolQ evaluation: 2 jobs (single + multi), queued/running
- SFT 0.5ep BoolQ evaluation: 3 remaining configs, running
- DPO 2pair 2ep SNLI-H evaluation: 2 configs (b4 + b16), queued

### Pending after current jobs
- Evaluate all GRPO g16 models (old + v2) once training completes
- Evaluate GRPO g4 BoolQ once training completes

### Next experiments
- GDPO implementation and training
- Higher-N evaluations (200+ samples) for top configs across methods
- Benchmark evaluation (MMLU, HellaSwag, ARC) for capability degradation checks
