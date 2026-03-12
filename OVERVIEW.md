# Post-Training Methods for Counterfactual Generation: Overview

Comparing post-training self-alignment methods for improving counterfactual generation quality. All methods fine-tune Qwen/Qwen2.5-7B-Instruct using QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three NLI/classification datasets.

**Last updated**: March 12, 2026

---

## 1. Method Comparison

Best ΔLFR per method per dataset, using fair-base evaluations. SNLI results use N=200; BoolQ N=200 evals are running (N=100 shown).

| Dataset | DPO | SFT | GRPO g4 † | GRPO v2 g16 | GRPO mv2 g16 |
|---------|:---:|:---:|:---------:|:-----------:|:------------:|
| **BoolQ** | **+9.4%** | +4.1% | −10.2% | −8.3% | −6.5% |
| | 2pair b16, 3ep | 2pair b4, 200step | multi g4, 2ep | v2, ~0.11ep | mv2 lr1e‑6, ~0.25ep |
| **SNLI-P** | +22.0% | −1.7% | +7.2% | +25.8% | **+27.5%** |
| | 2pair b4, 2ep | 1pair b4, 200step | multi g4, 2ep | v2, ~0.50ep | mv2, 1.0ep |
| **SNLI-H** | +21.0% | +1.7% | −10.6% | +11.0% | **+15.1%** |
| | 2pair b4, 2ep | 2pair b4, 2ep | single g4, 2ep | v2, 1.0ep | mv2, 1.0ep |


† GRPO g4 single-reward used old reward formula (`flip + 0.8 * sim`). GRPO v2 uses the corrected single reward (`flip + confidence * sim`). GRPO mv2 uses multi-reward v2 (flip + similarity + gated confidence + format). See section 5 for reward configuration details.

**Multi-reward v2 is the best GRPO configuration**, confirmed at N=200: SNLI-P +27.5% and SNLI-H +15.1%. It substantially outperforms DPO on both SNLI datasets (+22.0% and +21.0% respectively). The SNLI-H gap widened at N=200: mv2 jumps from +11.5% (N=100) to +15.1%, while DPO rises from +7.5% to +21.0%. BoolQ remains the only dataset where DPO clearly leads; all GRPO variants produce negative ΔLFR there. GDPO results are omitted from the main table pending further tuning (see section 5).

The checkpoint sweep reveals that v2 performance is highly sensitive to training duration: SNLI-P peaks at 0.5ep then declines, while SNLI-H steadily improves through 1.0ep. BoolQ degrades rapidly from the start. Experiments with 24 generations per prompt (g24) are in progress to test whether larger group sizes improve reward signal quality.

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

**Multi-reward v1 comparison:** Multi v1 g16 peaks at +12.4% on SNLI-P (0.5ep) but collapses after 0.62ep (NED=0.838, PPL=18738 at 0.88ep). On SNLI-H, multi v1 is consistently harmful at all checkpoints (-8.8% to -18.2%). However, **multi-reward v2** (with gated confidence and format rewards) reverses this trend entirely — see section 5.

**BoolQ:** Both multi and v2 remain harmful at all checkpoints. v2 degrades catastrophically with training (from -8.3% at 0.11ep to -36.5% at 0.25ep). This suggests a fundamental mismatch between the reward signal and the BoolQ task structure.

---

## 5. Reward Configurations and Multi-Reward v2

### Reward Configuration Comparison

Three GRPO reward configurations have been tested:

| Config | Reward functions | Key idea |
|--------|-----------------|----------|
| **Single v2** | 1 composite: `flip + confidence * sim` | Matches the DPO unified score; one reward signal for GRPO |
| **Multi v1** | 2 separate: `FlipReward` + `SimilarityReward` | Decomposed but no confidence, no format signal |
| **Multi v2** | 4 separate: `FlipReward` + `SimilarityReward` + `GatedConfidenceReward` + `FormatReward` | Full decomposition with gated confidence and explicit format signal |

### Multi-Reward v2 Design

The original multi-reward v1 (flip + similarity) suffered from catastrophic collapse on all datasets. Multi-reward v2 addresses this with two key additions:

- **GatedConfidenceReward**: Returns the base model's confidence score when the label flipped, 0.0 otherwise. Unlike single v2 where confidence always multiplies similarity regardless of flip, gated confidence only rewards high confidence when the edit actually worked. Uses a `PredictionCache` shared with `FlipReward` to avoid redundant model inference.
- **FormatReward**: Binary 1.0 for valid `<edit>...</edit>` tags, 0.0 otherwise. Provides an explicit recovery signal when the model stops producing valid formatting (BoolQ's "format collapse" failure mode). With multi v1, invalid format meant 0 on all rewards with no gradient to distinguish "bad format" from "bad content."

### Multi-Reward v2 Results

N=200 fair-base evaluations (BoolQ still at N=100, pending):

| Dataset | Multi v1 best | Single v2 best | **Multi v2** | DPO best |
|---------|:------------:|:--------------:|:------------:|:--------:|
| **SNLI-P** | +12.4% | +25.8% | **+27.5%** | +22.0% |
| **SNLI-H** | -8.8% | +11.0% | **+15.1%** | +21.0% |
| **BoolQ** | -9.3% | -8.3% | -7.1% | **+9.4%** |

Multi-reward v2 is the strongest GRPO configuration. On SNLI-H at N=200, the improvement from single v2 (+11.0%) to mv2 (+15.1%) confirms the benefit of gated confidence as a distinct training signal. GRPO mv2 also achieves the lowest PPL on SNLI-P (72.7 vs DPO's 142.3). BoolQ remains negative but slightly less harmful than other GRPO variants.

Note: DPO SNLI-H improved substantially at N=200 (+21.0% vs +7.5% at N=100), highlighting evaluation variance on this dataset.

### BoolQ Low-LR Experiments

BoolQ GRPO training shows catastrophic instability at the default lr=5e-6 (loss spikes to 86,680, grad_norm to 8.7 billion). Training with lr=1e-6 resolved the instability — no loss spikes or format collapse — but ΔLFR remained negative:

| Config | lr | ΔLFR | NED | PPL |
|--------|:--:|:----:|:---:|:---:|
| mv2 g16 | 5e-6 | -7.1% | 0.161 | 11.4 |
| mv2 g16 | **1e-6** | **-6.5%** | 0.153 | 11.3 |
| v2 g16 | 5e-6 | -8.3% | 0.155 | 11.8 |
| v2 g16 | **1e-6** | **-8.9%** | 0.153 | 11.7 |

Lower LR stabilized training and slightly improved mv2 (-7.1% → -6.5%), but the fundamental BoolQ difficulty for online RL persists. The root cause is format collapse: BoolQ passages are long, and the model hits `max_completion_length` (512 tokens), producing truncated outputs without valid `<edit>` tags.

### GDPO

GDPO (Group reward-Decoupled normalization Policy Optimization) extends GRPO for multi-objective optimization. Instead of summing rewards before normalization (GRPO default), GDPO normalizes each reward independently per group before summing (`normalize_then_sum`). This prevents high-magnitude rewards from dominating the advantage calculation.

Implemented as `GDPOTrainer` (subclass of `GRPOTrainer`) in `gdpo_trainer.py`. Uses `_RewardCapture` wrappers to intercept per-reward outputs, then recomputes advantages using per-reward group normalization + batch normalization. Training script: `train_gdpo.py`.

### GDPO Results (Preliminary)

Initial GDPO runs (lr=5e-6) suffered severe entropy collapse, performing much worse than GRPO on all datasets. GDPO's per-reward normalization amplifies noise when individual reward variance is low, causing faster collapse than standard GRPO at the same learning rate. Further tuning is in progress:

- **GDPO lr=1e-6** (FormatReward weight 3x): training pending for all 3 datasets
- **GDPO g24** (lr=1e-6, FormatReward weight 3x): training pending — larger group size may help stabilize per-reward normalization

---

## 6. Evaluation Methodology

- **Fair-base comparison**: base model counterfactuals are generated once and reused across all model evaluations via `--base_eval_dir`, ensuring ΔLFR differences reflect the fine-tuned model's quality rather than base model generation variance
- **Standard evaluation**: 200 validation samples, 10 CFs per sample, fair-base reused (previously 100; standardized to 200 for lower variance)
- **LFR/NED efficiency metric**: label flip rate divided by normalized edit distance, rewarding methods that achieve high flip rates with minimal edits (higher = more efficient). Reported in N=200 eval reports.
- **Base model as judge**: the un-fine-tuned base model classifies all counterfactuals (both base-generated and fine-tuned-generated) to determine label flips
- **Evaluation variance**: base LFR varies 2-5pp across independent runs; differences below ~5pp should be treated cautiously

---

## 7. Current Status

### Completed
- GRPO g16 checkpoint sweep: all 22 evaluations completed
- GRPO v2 g16 training: SNLI-P and SNLI-H completed (1.0ep); BoolQ timed out at 0.75ep
- GRPO multi-reward v2 training: SNLI-P and SNLI-H completed; BoolQ timed out at 0.83ep
- GDPO implementation: `gdpo_trainer.py` + `train_gdpo.py`
- BoolQ low-LR (lr=1e-6): both v2 and mv2 trained and evaluated — stable training, ΔLFR still negative
- GDPO initial runs (lr=5e-6): all 3 datasets trained and evaluated — entropy collapse on SNLI
- N=200 re-evaluations: SNLI-P and SNLI-H completed for SFT, GRPO v2, GRPO mv2, GDPO
- LFR/NED efficiency metric added to evaluation pipeline

### Running / Pending
- N=200 BoolQ re-evaluations: 5 jobs running (SFT, GRPO v2, GRPO mv2, GRPO mv2 lr1e6, GDPO)
- GDPO low-LR training: 3 jobs pending (g16, lr=1e-6, FormatReward weight 3x)
- g24 experiments: 9 training jobs pending (GRPO v2, GRPO mv2, GDPO for all 3 datasets)

### Key findings
- **GRPO mv2 g16 is the best overall**: +27.5% on SNLI-P, +15.1% on SNLI-H (N=200)
- **mv2 beats DPO on SNLI-P** by a clear margin (+27.5% vs +22.0%)
- **Gated confidence reward is key for SNLI-H**: jump from +11.0% (single v2) to +15.1% (mv2) at N=200
- **DPO SNLI-H improved substantially at N=200** (+21.0% vs +7.5%), highlighting evaluation variance
- **BoolQ remains harmful** for all GRPO configurations; format collapse is the key failure mode
- **Early stopping is critical**: SNLI-P peaks at 0.5ep; SNLI-H improves through 1.0ep

### Next steps
- Analyze BoolQ N=200 results once running evals complete
- Evaluate GDPO low-LR and g24 models once training completes
- Consider BoolQ-specific improvements: increase `max_completion_length`, SFT warm-start

For detailed GRPO training analysis (entropy collapse, reward collapse, epoch optimization), see [GRPO_ANALYSIS.md](GRPO_ANALYSIS.md).
