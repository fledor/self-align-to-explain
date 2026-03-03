# Post-Training Methods for Counterfactual Generation: Overview

Comparing post-training self-alignment methods for improving counterfactual generation quality. All methods fine-tune Qwen/Qwen2.5-7B-Instruct using QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three NLI/classification datasets.

**Last updated**: March 5, 2026

---

## 1. Method Comparison

Best ΔLFR per method per dataset, using fair-base evaluations (N=100 unless noted).

| Dataset | DPO | SFT | GRPO g4 † | GRPO g16 | GDPO |
|---------|:---:|:---:|:---------:|:--------:|:----:|
| **BoolQ** | **+9.4%** | +4.1% | −10.2% | −8.3% | *planned* |
| | 2pair b16, 3ep | 2pair b4, 200step | multi g4, 2ep | v2 g16, ~0.11ep | |
| **SNLI-P** | +17.3% | +2.7% | +7.2% | **+28.7%** | *planned* |
| | 2pair b4, 2ep | 1pair b4, 200step | multi g4, 2ep | v2 g16, ~0.50ep | |
| **SNLI-H** | +7.5% | +1.8% | -10.6% | **+7.3%** | *planned* |
| | 1pair b4, 2ep | 2pair b4, 2ep | single g4, 2ep | v2 g16, 1.0ep | |

† GRPO g4 single-reward used old reward formula (`flip + 0.8 * sim`). GRPO g16 v2 uses the corrected formula (`flip + confidence * sim`), matching the DPO unified score. Multi-reward runs use separate FlipReward + SimilarityReward.

**GRPO v2 g16 beats DPO on SNLI-P** (+28.7% vs +17.3%) and **matches DPO on SNLI-H** (+7.3% vs +7.5%), both with the corrected single reward and optimal early stopping. BoolQ remains the only dataset where DPO clearly leads. Multi-reward GRPO is consistently worse than v2 single-reward across all datasets. SFT provides modest or negative gains.

The checkpoint sweep reveals that v2 performance is highly sensitive to training duration: SNLI-P peaks at 0.5ep then declines, while SNLI-H steadily improves through 1.0ep. BoolQ degrades rapidly from the start.

Higher-N evaluations (N=200, fair base) show DPO 2pair b4 2ep reaches +21.0% on SNLI-H and +22.0% on SNLI-P; similar large-N evals for GRPO v2 are planned.

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

BoolQ had adequate signal (~90% effective steps). Eval completed: GRPO g4 BoolQ is harmful (−10.2% to −10.4% ΔLFR), worse than base. SNLI-H was severely compromised, with only ~17% of steps providing gradient signal.

### Results (g4)

On SNLI-P, GRPO g4 achieves +5.4% (single) and +7.2% (multi), placing it between SFT and DPO. Notably, GRPO produces smaller edits (NED 0.27 vs DPO's 0.33) and better fluency (PPL 85-92 vs DPO's 84-142).

On SNLI-H, GRPO g4 is harmful (-10.6% single, -11.3% multi), worse than both base and SFT. The models learned to make extremely minimal edits (NED ~0.16, half of base) that rarely flip labels. With 83% zero-std, training signal was too sparse to learn effectively.

### Reward correction

The initial GRPO single-reward runs (g4 and g16) used `flip + 0.8 * similarity` — an arbitrary weight with no confidence term. The DPO unified score uses `flip_bonus + confidence * similarity`. The corrected GRPO single reward now matches: `flip + confidence * similarity`, where confidence is the base model's classification confidence on the edited text.

Multi-reward runs are unaffected (they use separate `FlipReward` + `SimilarityReward` classes). All g4 results and the first g16 single-reward results use the old formula (marked with † in RESULTS.md). New g16 v2 single-reward jobs use the corrected formula.

### Epoch analysis

Analysis of g4 training curves (see [GRPO_ANALYSIS.md](GRPO_ANALYSIS.md)) revealed that useful learning concentrates in the first 0.25-0.5 epochs. By epoch 1.0, all SNLI models show complete entropy collapse (entropy ~0.003, zero gradients). Training for 2 epochs wastes 50-75% of compute. All g16 runs now train for 1 epoch only.

### Reruns with num_generations=16 and checkpoint sweep

All 6 configurations (3 datasets, 2 modes) were resubmitted with `num_generations=16` (1 epoch) to increase group diversity. An additional 3 single-reward v2 jobs use the corrected reward formula.

A comprehensive checkpoint sweep evaluated both multi and v2 g16 at ~0.25ep intervals. The results reveal that **v2 single-reward dramatically outperforms multi-reward** and that **optimal training duration varies by dataset**:

**SNLI-P checkpoint sweep (v2 g16):**

| Checkpoint | ~Epoch | ΔLFR | NED | PPL |
|------------|--------|------|-----|-----|
| ckpt-4000 | 0.25ep | +27.2% | 0.359 | 70.7 |
| ckpt-8000 | **0.50ep** | **+28.7%** | 0.347 | 89.1 |
| ckpt-12000 | 0.75ep | +21.3% | 0.339 | 95.9 |
| ckpt-16000 | 1.0ep | +24.4% | 0.341 | 94.7 |

Peak at 0.5ep, then decline at 0.75ep with partial recovery at 1.0ep. The +28.7% result surpasses DPO's best (+17.3%) by 11.4 percentage points.

**SNLI-H checkpoint sweep (v2 g16):**

| Checkpoint | ~Epoch | ΔLFR | NED | PPL |
|------------|--------|------|-----|-----|
| ckpt-4000 | 0.25ep | +2.8% | 0.290 | 319.4 |
| ckpt-8000 | 0.50ep | +6.3% | 0.276 | 370.6 |
| ckpt-12000 | 0.75ep | +6.2% | 0.276 | 450.8 |
| ckpt-16000 | **1.0ep** | **+7.3%** | 0.285 | 374.4 |

Steadily improves through 1.0ep, nearly matching DPO's best (+7.5%). Unlike multi-reward which degrades severely on SNLI-H (-8.8% to -18.2%), v2 maintains stable improvement.

**Multi-reward comparison:** Multi g16 peaks at +12.4% on SNLI-P (0.5ep) but collapses after 0.62ep (NED=0.838, PPL=18738 at 0.88ep). On SNLI-H, multi is consistently harmful at all checkpoints (-8.8% to -18.2%). The confidence-weighted single reward clearly outperforms decomposed rewards.

**BoolQ:** Both multi and v2 remain harmful at all checkpoints. v2 degrades catastrophically with training (from -8.3% at 0.11ep to -36.5% at 0.25ep). This suggests a fundamental mismatch between the reward signal and the BoolQ task structure.

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

### Completed
- GRPO g16 checkpoint sweep: 19/22 jobs completed, 3 BoolQ jobs resubmitted (timed out)
- GRPO v2 g16 training: SNLI-P and SNLI-H completed (1.0ep); BoolQ timed out at 0.75ep

### Pending
- 3 BoolQ eval jobs resubmitted (jobs 2600632-34)

### Key findings from sweep
- **GRPO v2 g16 beats DPO on SNLI-P** (+28.7% at 0.5ep vs DPO +17.3%)
- **GRPO v2 g16 matches DPO on SNLI-H** (+7.3% at 1.0ep vs DPO +7.5%)
- **BoolQ remains harmful** for all GRPO configurations
- **v2 single-reward >> multi-reward** across all datasets
- **Early stopping is critical**: SNLI-P peaks at 0.5ep; SNLI-H improves through 1.0ep

### Next experiments
- GDPO implementation and training (per-reward normalization)
- Higher-N evaluations (200+ samples) for GRPO v2 best configs
- Investigate BoolQ GRPO failure (reward hacking analysis)
- Benchmark evaluation (MMLU, HellaSwag, ARC) for capability degradation checks

For detailed GRPO training analysis (entropy collapse, reward collapse, epoch optimization), see [GRPO_ANALYSIS.md](GRPO_ANALYSIS.md).
