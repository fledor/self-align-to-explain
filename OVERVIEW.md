# Post-Training Methods for Counterfactual Generation: Overview

Comparing post-training self-alignment methods for improving counterfactual generation quality. All methods fine-tune Qwen/Qwen2.5-7B-Instruct using QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three NLI/classification datasets.

**Last updated**: March 26, 2026

---

## 1. Method Comparison

Best ΔLFR per method per dataset, using fair-base N=200 evaluations only.

| Dataset | DPO | SFT | GRPO single | GRPO multi | GDPO best |
|---------|:---:|:---:|:-----------:|:----------:|:---------:|
| **BoolQ** | **+10.5%** | +1.8% | +0.4% | +1.7% | +1.1% |
| **SNLI-P** | +15.7% | −1.7% | +25.8% | **+27.5%** | +18.2% |
| **SNLI-H** | **+21.0%** | +1.7% | +11.0% | +15.1% | **+18.7%** |

**GRPO mv2 g16 remains the best configuration overall**: SNLI-P +27.5%, SNLI-H +15.1%, BoolQ +1.7% (with mcl768). It outperforms DPO on SNLI-P (+27.5% vs +15.7%). DPO leads on SNLI-H (+21.0% vs +15.1%) and BoolQ (+10.5% vs +1.7%).

**GDPO v6 with early stopping is competitive with GRPO**: On SNLI-P, GDPO v6 peaks at ckpt-2000 (~0.25ep): +18.2% ΔLFR, closing 2/3 of the gap to GRPO (+27.5%). On SNLI-H, GDPO v6 peaks at ckpt-6000 (~0.75ep): **+18.7% ΔLFR, surpassing GRPO mv2 (+15.1%) by 3.6pp**. This is the first dataset where GDPO outperforms GRPO. The learning rate (5e-6 vs 1e-6) is the dominant factor, followed by conditioned rewards + KL penalty. GDPO v7 (32 groups) matched v6 exactly (+15.7% at full epoch), confirming more groups don't help. See section 5 for full analysis.

**g24 does not improve over g16 in most cases**: On SNLI-P, mv2 g24 (+22.2%) underperforms mv2 g16 (+27.5%); v2 g24 (+8.6%) is far worse than v2 g16 (+25.8%) with degraded PPL (770). On BoolQ, g24 is consistently slightly worse. The one apparent improvement (SNLI-H v2 g24 +22.6%) comes with collapsed fluency. The additional generation overhead of g24 does not justify the results.

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

BoolQ GRPO training shows catastrophic instability at the default lr=5e-6 (loss spikes to 86,680, grad_norm to 8.7 billion). Training with lr=1e-6 resolved the instability — no loss spikes or format collapse.

N=200 re-evaluation revealed that the negative ΔLFR results at N=100 were largely due to evaluation variance (base LFR ~43% at N=100 vs ~36.5% at N=200):

| Config | lr | ΔLFR (N=100) | ΔLFR (N=200) | NED | PPL | LFR/NED |
|--------|:--:|:----:|:----:|:---:|:---:|:---:|
| mv2 g16 | 5e-6 | -7.1% | **+1.2%** | 0.170 | 11.6 | 2.23 |
| mv2 g16 | **1e-6** | -6.5% | **+1.1%** | 0.155 | 11.8 | 2.41 |
| v2 g16 | 5e-6 | -8.3% | **+0.4%** | 0.151 | 11.7 | 2.49 |

The v2 g16 model achieves the best LFR/NED efficiency (2.49) of any BoolQ configuration, with the lowest NED (0.151) indicating very targeted edits. The mv2 lr1e-6 model is a close second (2.41).

### GDPO

GDPO (Group reward-Decoupled normalization Policy Optimization) extends GRPO for multi-objective optimization. Instead of summing rewards before normalization (GRPO default), GDPO normalizes each reward independently per group before summing (`normalize_then_sum`). This prevents high-magnitude rewards from dominating the advantage calculation.

Implemented as `GDPOTrainer` (subclass of `GRPOTrainer`) in `gdpo_trainer.py`. Uses `_RewardCapture` wrappers to intercept per-reward outputs, then recomputes advantages using per-reward group normalization + batch normalization. Training script: `train_gdpo.py`.

### GDPO Results

Initial GDPO runs (lr=5e-6) suffered severe entropy collapse. GDPO's per-reward normalization amplifies noise when individual reward variance is low, causing faster collapse than standard GRPO at the same learning rate.

**GDPO lr=1e-6** (FormatReward weight 3x) resolved collapse but shows mixed results:

| Dataset | ΔLFR (g16 lr5e-6) | ΔLFR (g16 lr1e-6) | NED | PPL |
|---------|:--:|:--:|:---:|:---:|
| BoolQ | +1.1% | +0.2% | 0.153 | 11.6 |
| SNLI-P | +2.5% | -0.5% | 0.244 | 134.5 |
| SNLI-H | -22.0% | +0.1% | 0.307 | 283.8 |

The lower learning rate prevents catastrophic collapse (SNLI-H improved from -22.0% to +0.1%) but ΔLFR remains near zero for all datasets — GDPO lr1e-6 essentially preserves base model performance without meaningful improvement.

**GDPO g24** (lr=1e-6, all 3 datasets completed): BoolQ -0.6%, SNLI-P +1.1%, SNLI-H +1.1%. Slightly better than g16 lr1e-6 on SNLI but still near zero.

**GDPO v2 (conditioned rewards + KL + dynamic sampling)**: Resolved training dynamics (entropy stable, rewards active, zero-variance groups reduced to 7-28%). BoolQ +0.4%, SNLI-H +0.7%, SNLI-P +0.8%. Training dynamics improved dramatically vs original GDPO, but evaluation ΔLFR remains near zero.

**GDPO v3 (lr=5e-6 + NED penalty)**: Attempted to match GRPO's learning rate with NED penalty for non-flipping completions. SNLI training was unstable but eval from checkpoints showed: SNLI-P +8.3% (best pre-v5 GDPO result, despite KL spiking to 22.7), SNLI-H +1.8%. BoolQ v3 hurt performance (-1.1% ΔLFR). The NED penalty made BoolQ worse, tentatively confirming that the BoolQ issue is about edit *location* not edit *size*.

**GDPO v4 (lr=3e-6 + beta=0.005)**: Timed out at epoch ~0.97 with recurring KL spikes. Eval from checkpoint-15400: SNLI-P +4.0%.

**GDPO v5 (paper-aligned, critical bug fix)**: All prior GDPO experiments had `generation_batch_size=16`, meaning GDPO's batch-wise normalization operated on 1 group (16 samples) — mathematically identical to GRPO. The fix: `generation_batch_size=128` (8 groups of 16). Also aligned with paper: lr=1e-6, beta=0.0005, epsilon_high=0.28 (DAPO asymmetric clipping). VRAM-tested on single A100-80GB. SNLI-P v5 result: **+1.7% ΔLFR** — proper GDPO at low LR barely outperforms base.

**GDPO v6 (higher LR = breakthrough)**: Same as v5 (8 groups, conditioned rewards, beta=0.0005, epsilon_high=0.28) but with lr=5e-6 instead of lr=1e-6. Full epoch SNLI-P result: **+15.7% ΔLFR** (NED=0.287, PPL=135.4). Checkpoint sweeps reveal dataset-dependent peaks:

- **SNLI-P**: Peak at ckpt-2000 (~0.25ep): **+18.2%**. Trajectory: +18.2% → +18.1% → +16.5% → +15.7% (declining)
- **SNLI-H**: Peak at ckpt-6000 (~0.75ep): **+18.7%** (surpasses GRPO mv2 +15.1% by 3.6pp). Trajectory: +15.9% → +17.6% → **+18.7%** → +17.6% (1.0ep decline)

GDPO peaks earlier than GRPO on SNLI-P (0.25ep vs 0.5ep) but at a similar point on SNLI-H. The per-reward normalization may cause faster overfitting on tasks with stronger reward signal (SNLI-P) but not on harder tasks (SNLI-H).

**GDPO v7 (4 GPU + v6 innovations)**: 4 GPUs (32 groups) with all v6 innovations (conditioned rewards, beta=0.0005, epsilon_high=0.28). Result: **+15.7% ΔLFR** (NED=0.285, PPL=117.3) — identical ΔLFR to v6 (8 groups) but with better PPL. This confirms that **more groups do not improve ΔLFR**. The 32-group normalization produces smoother gradients (lower PPL) but the same label-flip performance.

**GDPO paper-match (maximum paper alignment)**: Uses TRL's built-in `multi_objective_aggregation="normalize_then_sum"` (not our custom GDPOTrainer), 4 A100-80GB GPUs for 32 groups, lr=5e-6, beta=0.0, standard (non-conditioned) rewards. Paired with a matched GRPO control (`sum_then_normalize`). Results: **GDPO +7.6%**, **GRPO control +25.9%**. The paper-match GDPO (+7.6%) underperforms our v6 (+15.7%) despite having 4x more groups, because it lacks conditioned rewards and KL regularization.

**Key GDPO insights**:
1. **Learning rate is dominant**: v5 +1.7% at lr=1e-6 vs v6 +15.7% at lr=5e-6 (9.2x improvement)
2. **Conditioned rewards + KL matter**: v6 +15.7% with cond/KL vs paper-match +7.6% without (2.1x)
3. **More groups don't help ΔLFR**: v7 32 groups = v6 8 groups = +15.7%
4. **Early stopping is critical**: SNLI-P peak at ~0.25ep (+18.2%); SNLI-H peak at ~0.75ep (+18.7%)
5. **GDPO can beat GRPO**: On SNLI-H, GDPO v6 ckpt-6000 (+18.7%) surpasses GRPO mv2 g16 (+15.1%) by 3.6pp

---

## 6. Fairness of Comparison: GDPO v6 vs GRPO mv2

The current GDPO v6 vs GRPO mv2 comparison is **confounded**: GDPO v6 benefits from several stabilization techniques that GRPO mv2 does not use:

| Technique | GDPO v6 | GRPO mv2 |
|-----------|:-------:|:--------:|
| Conditioned rewards (ConditionedSimilarityReward) | Yes | No (uses SimilarityReward) |
| Asymmetric clipping (epsilon_high=0.28) | Yes | No (symmetric, epsilon=0.2) |
| KL penalty (beta=0.0005) | Yes | No (beta=0.0) |
| Dynamic sampling (zero-variance filtering) | Yes | No |

These techniques (borrowed from the GDPO paper and DAPO) improve training stability and reward alignment. When GDPO v6 outperforms GRPO mv2 on SNLI-H (+18.7% vs +15.1%), it is unclear whether the gain comes from **per-reward normalization** (GDPO's core contribution) or from the additional stabilization tricks.

### Fair Comparison Runs (submitted)

To isolate the effect of per-reward normalization, we submitted **GRPO-fair** runs that give GRPO the same tricks as GDPO v6:

| Parameter | GRPO mv2 (current) | GRPO-fair (new) | GDPO v6 |
|-----------|:------------------:|:---------------:|:-------:|
| Reward functions | SimilarityReward | **ConditionedSimilarityReward** | ConditionedSimilarityReward |
| epsilon_high | None (symmetric) | **0.28** | 0.28 |
| beta (KL) | 0.0 | **0.0005** | 0.0005 |
| Learning rate | 5e-6 | 5e-6 | 5e-6 |
| Batch size | 1×4 (eff. 4) | **8×1 (eff. 8)** | 8×1 (eff. 8) |
| generation_batch_size | 16 | **128** | 128 |
| num_generations | 16 | 16 | 16 |

Runs submitted: SNLI-P (train 2714670, eval 2714671), SNLI-H (train 2714672, eval 2714673), BoolQ (train 2714674, eval 2714675).

If GRPO-fair matches or exceeds GDPO v6, the stabilization tricks (not per-reward normalization) explain the gains. If GDPO v6 still outperforms GRPO-fair, per-reward normalization provides genuine benefit.

---

## 7. Evaluation Methodology

- **Fair-base comparison**: base model counterfactuals are generated once and reused across all model evaluations via `--base_eval_dir`, ensuring ΔLFR differences reflect the fine-tuned model's quality rather than base model generation variance
- **Standard evaluation**: 200 validation samples, 10 CFs per sample, fair-base reused (previously 100; standardized to 200 for lower variance)
- **LFR/NED efficiency metric**: label flip rate divided by normalized edit distance, rewarding methods that achieve high flip rates with minimal edits (higher = more efficient). Reported in N=200 eval reports.
- **Base model as judge**: the un-fine-tuned base model classifies all counterfactuals (both base-generated and fine-tuned-generated) to determine label flips
- **Evaluation variance**: base LFR varies 2-5pp across independent runs; differences below ~5pp should be treated cautiously

---

## 8. Current Status

### Completed
- All GRPO, DPO, SFT evaluations completed (g16 checkpoint sweep, v2, mv2, g24, mcl768, etc.)
- GDPO v0-v7: all trained and evaluated. v6 ckpt sweep: SNLI-P peak at ~0.25ep (+18.2%), SNLI-H peak at ~0.50ep (+17.6%)
- GDPO v6 SNLI-H beats GRPO mv2: +17.6% vs +15.1% (first dataset where GDPO outperforms GRPO)
- GDPO v7 (4 GPU, 32 groups, v6 innovations): +15.7% — confirms more groups don't help
- GDPO paper-match: 4-GPU GDPO (+7.6%) vs GRPO control (+25.9%) completed
- SFT LR sweep: lr=1e-5 and lr=2e-5 both negative; lr=5e-6 confirmed optimal
- GRPO SFT warm-start BoolQ: +1.2% (no improvement over cold-start GRPO +1.2%)
- SFT BoolQ ml2048: +1.8% (improvement over -0.0% at default ml1024)
- DPO BoolQ ml2048 (2pair b4): +8.5% ΔLFR (below 1pair +10.5%; 2pair is worse for BoolQ DPO)
- GDPO v6 BoolQ mcl1024: +0.3% ΔLFR (longer completions didn't help GDPO on BoolQ)
- GRPO mv2 BoolQ mcl1024: TIMEOUT after 24h; eval pending on latest checkpoint

### Key findings
- **GRPO mv2 g16 is the best overall**: +27.5% on SNLI-P, +15.1% on SNLI-H, +1.7% on BoolQ mcl768 (N=200)
- **GDPO v6 beats GRPO on SNLI-H**: GDPO v6 ckpt-6000 +18.7% vs GRPO mv2 +15.1% — the first dataset where GDPO outperforms GRPO
- **GDPO v6 with early stopping achieves +18.2% on SNLI-P**: Peak at ckpt-2000 (~0.25ep), closing 2/3 gap to GRPO
- **LR is dominant for GDPO**: v5 +1.7% at lr=1e-6 vs v6 +15.7% at lr=5e-6
- **Conditioned rewards + KL > more groups**: v6 8 groups +15.7% > paper-match 32 groups +7.6%
- **32 groups = 8 groups for ΔLFR**: v7 matches v6 exactly (+15.7%), but has better PPL (117.3 vs 135.4)
- **BoolQ training truncation is NOT the bottleneck**: SFT ml2048 +1.8%, DPO ml2048 +8.5% (below 1pair +10.5%), GDPO v6 mcl1024 +0.3%, GRPO mcl1024 timed out. Longer sequences don't meaningfully help any method
- **Early stopping is dataset-dependent**: GDPO peaks at 0.25ep on SNLI-P, 0.75ep on SNLI-H; GRPO at 0.5ep on SNLI-P

### Running / Pending

| Job | Description | Status |
|-----|-------------|--------|
| 2714670/71 | GRPO-fair SNLI-P (train + eval) | Submitted |
| 2714672/73 | GRPO-fair SNLI-H (train + eval) | Submitted |
| 2714674/75 | GRPO-fair BoolQ (train + eval) | Submitted |
| 2714587 | GRPO mv2 BoolQ mcl1024 ckpt-2000 eval | Submitted |
| 2714597/98 | DPO 1pair b4 BoolQ ml2048 (train + eval) | Submitted |

### Next steps

- **Analyze BoolQ mcl1024/ml2048 results**: When all 4 BoolQ experiments complete, determine whether training truncation was a significant bottleneck
- **GDPO v6 early-stopped on all datasets**: Based on ckpt sweep results, apply optimal early stopping to SNLI-H and BoolQ
- **GDPO v6 early-stopped BoolQ ckpt sweep**: After training completes, eval at multiple checkpoints to find optimal BoolQ stopping point

For full GDPO root cause analysis and design decisions, see [GDPO_BOOLQ_IMPROVEMENTS.md](GDPO_BOOLQ_IMPROVEMENTS.md).

---

## 9. Implementation Changes

### Conditioned Rewards (`train_grpo.py`)
- New `ConditionedSimilarityReward` class: returns similarity only when `PredictionCache` reports flip=True
- Enabled via `--conditioned_rewards` flag in `train_gdpo.py`
- `run_gdpo.sh`: `CONDITIONED_REWARDS=1` env var

### GDPO Trainer Improvements (`gdpo_trainer.py`)
- Zero-variance group filtering: detects groups with `var < 1e-8` in combined advantage, sets advantages to 0
- Logs `zero_var_groups=N/M` to track filtered groups per step

### BoolQ Completion Length
- `run_grpo.sh`, `run_gdpo.sh`: `MAX_COMPLETION_LENGTH` env var (default: 512, use 768 for BoolQ)

### SFT Warm-Start Pipeline (removed)
- `extract_sft_data.py` and `merge_lora.py` have been deleted; the SFT warm-start experiment was confirmed a failure (+1.2% ΔLFR, no improvement over cold-start GRPO)

### NED Penalty (`train_grpo.py`)
- `ConditionedSimilarityReward` extended with `ned_penalty_alpha` parameter
- When non-flipping: returns `-alpha * NED` instead of 0.0 (recovers signal from zero-variance groups)
- Configurable via `--ned_penalty_alpha` in `train_gdpo.py`, `NED_PENALTY_ALPHA` env var
- Experimental result: hurt BoolQ (-1.1% vs +0.4% without); SNLI impact unclear due to training instability

### GDPO Beta Configuration (`run_gdpo.sh`)
- Added `BETA` env var to configure KL penalty coefficient
- Default: 0.001 (GDPO v2/v3), configurable to 0.005 (GDPO v4)
