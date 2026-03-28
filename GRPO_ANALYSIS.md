# GRPO Training Analysis: Challenges and Findings

This document analyzes the key challenges encountered during Group Relative Policy Optimization (GRPO) training for counterfactual generation, including entropy collapse, reward advantage collapse, and dataset-specific difficulties.

## 1. Training Setup

GRPO generates multiple completions per prompt during training and learns from relative reward differences within each group. Our configuration:

- **Model**: Qwen2.5-7B-Instruct with QLoRA (4-bit, rank 32)
- **Group sizes tested**: g4 (4 generations/prompt), g16 (16 generations/prompt)
- **Reward modes**: single combined reward, multi-reward (decomposed flip + similarity)
- **Temperature**: 1.2
- **KL penalty (beta)**: 0.0 (disabled, per DeepSeek R1)
- **Loss type**: DAPO
- **Batch**: per_device=1, grad_accum=4 (effective batch size 4)
- **LR schedule**: cosine decay, 5e-6 peak, 10% warmup

## 2. Reward Advantage Collapse

The core challenge: when all completions in a group receive identical rewards, the reward standard deviation is zero, all advantages are zero, and the gradient update is zero. The model learns nothing from that step.

This is tracked by `frac_reward_zero_std` (fraction of groups with zero reward std in each step).

### Collapse Rates by Dataset and Group Size

**Single-reward mode:**

| Dataset | g4 (2ep) | g16 (~0.4ep) |
|---------|----------|--------------|
| BoolQ   | 8.7% (347/4000)   | 22.8% (204/894)   |
| SNLI-P  | 50.2% (4018/8000) | 16.3% (266/1631)  |
| SNLI-H  | 82.8% (6625/8000) | 37.2% (653/1757)  |

**Multi-reward mode:**

| Dataset | g4 (2ep) | g16 (~0.4ep) |
|---------|----------|--------------|
| BoolQ   | 11.6% (463/4000)  | 28.9% (267/925)   |
| SNLI-P  | 41.4% (3314/8000) | 10.0% (158/1578)  |
| SNLI-H  | 83.5% (6679/8000) | 67.2% (1164/1732) |

**Observations:**

- **SNLI-H is consistently worst** across all configurations: 83% collapse (g4) and 37-67% collapse (g16).
- **g16 helps SNLI-P dramatically**: collapse drops from 50% to 10-16%.
- **g16 helps SNLI-H moderately**: collapse drops from 83% to 37-67%, but remains far too high.
- **BoolQ is relatively healthy** across all configurations.
- These g16 numbers are partial (~0.4 epochs); collapse rates worsen over training as entropy declines.

## 3. Entropy Collapse

Entropy measures the diversity of the model's output distribution. As training progresses, the policy becomes increasingly deterministic -- entropy collapses toward zero, and the model produces identical completions for each prompt.

### Entropy Trajectory (g4 single-reward, 2 epochs completed)

| Epoch | BoolQ   | SNLI-P  | SNLI-H  |
|-------|---------|---------|---------|
| 0.10  | 0.0497  | 0.1369  | 0.2493  |
| 0.25  | 0.1482  | 0.0904  | 0.0270  |
| 0.50  | 0.0572  | 0.0535  | 0.0190  |
| 0.75  | 0.0391  | 0.0275  | 0.0164  |
| 1.00  | 0.0337  | 0.0030  | 0.0056  |
| 1.50  | 0.0975  | 0.0030  | 0.0030  |
| 2.00  | 0.1065  | 0.0081  | 0.0218  |

**BoolQ** maintains entropy throughout training (it even increases toward the end), while **SNLI-P** collapses at epoch 1.0 and **SNLI-H** collapses by epoch 0.5. Once entropy reaches ~0.003, the model is functionally deterministic.

### Why BoolQ Is Different

BoolQ generates much longer completions (50-230 tokens) compared to SNLI-P (15-35 tokens) and SNLI-H (11-18 tokens). Longer outputs provide more token-level diversity, making it harder for all 4/16 completions to be identical.

## 4. Epoch Analysis: Why 1 Epoch Is Optimal

The g4 training data reveals a clear pattern: useful learning concentrates in the first 0.25-0.5 epochs.

### Learning Signal by Epoch (g4 single-reward)

| Epoch | BoolQ grad_norm | SNLI-P grad_norm | SNLI-H grad_norm |
|-------|-----------------|------------------|------------------|
| 0.10  | 0.085           | 0.142            | 0.520            |
| 0.25  | 0.051           | 0.231            | 1.406            |
| 0.50  | 0.056           | 0.461            | 0 (collapsed)    |
| 0.75  | 0 (collapsed)   | 1.523            | 0 (collapsed)    |
| 1.00  | 0.034           | 0 (collapsed)    | 0 (collapsed)    |
| 1.50  | 0.106           | 0 (collapsed)    | 0 (collapsed)    |
| 2.00  | 0.105           | 0 (collapsed)    | 0 (collapsed)    |

**Findings:**

- **SNLI-H**: No useful gradient signal after epoch 0.25 (with g4). The model collapses after ~2000 steps.
- **SNLI-P**: Gradient signal active until epoch 0.75, collapses at epoch 1.0. The model stops learning for the entire second epoch.
- **BoolQ**: Intermittent collapse but recovers due to high output diversity. Maintains some learning throughout.

**Conclusion**: Training for 2 epochs wastes 50-75% of compute on collapsed training. 1 epoch captures all useful learning and avoids training past the point of collapse.

### Impact on Future Runs

All GRPO training has been switched to 1 epoch (from 2). The currently running g16 old-reward jobs will naturally reach ~1 epoch within their 24h SLURM limit.

## 5. SNLI-H: A Structural Challenge for GRPO

SNLI hypothesis editing is fundamentally harder for online RL methods. The root causes are:

### 5.1 Short Output Space

NLI hypotheses are typically 5-15 words. The model generates very short completions (11-18 tokens), leaving minimal room for variation across the 16 completions in a group. When all outputs are nearly identical, the reward signal has zero variance.

### 5.2 Binary Reward Landscape

The flip reward is binary (0 or 1): either the NLI label changed or it didn't. For SNLI-H, most minimal edits to a hypothesis don't change the NLI prediction, so the model gets reward ~0.8 (just similarity, no flip) for most outputs. To actually flip the label, the edit needs to change the semantic relationship (entailment/neutral/contradiction), which often requires a specific non-minimal change.

### 5.3 Comparison with Offline Methods

DPO and SFT don't suffer from these issues because they learn from pre-generated pairs where the "right answer" is already known:

| Method | SNLI-H LFR | SNLI-H NED | Notes |
|--------|------------|------------|-------|
| Base model | 46.9% | 0.213 | - |
| SFT 1pair b4 1ep | 54.3% | 0.224 | +7.4% LFR |
| DPO 1pair b4 200step | 51.2% | 0.143 | +4.3% LFR, very low NED |
| **GRPO single g4 2ep** | **36.3%** | **0.167** | **-10.6% LFR (worse than base)** |
| **GRPO multi g4 2ep** | **35.6%** | **0.151** | **-11.3% LFR (worse than base)** |

GRPO makes SNLI-H *worse* than the base model. The model learns to produce very similar text (low NED) that doesn't flip the label, effectively gaming the similarity reward while the flip reward stagnates due to collapsed exploration.

### 5.4 Potential Mitigations (Not Yet Tested)

1. **Higher temperature** (1.5-2.0): Force more diverse generations to maintain reward variance.
2. **KL penalty (beta > 0)**: Prevent entropy collapse by penalizing divergence from the reference policy.
3. **Early stopping**: Evaluate intermediate checkpoints (step 200, 400) where entropy was still healthy.
4. **Task-specific reward shaping**: Weight the flip reward more heavily for SNLI-H.
5. **Accept as a finding**: GRPO may be structurally unsuitable for very short, constrained edit tasks.

## 6. g4 vs g16: Impact of Group Size

Increasing `num_generations` from 4 to 16 was intended to increase reward variance within groups. Results are mixed:

| Metric | g4 | g16 | Effect |
|--------|-----|------|--------|
| SNLI-P collapse rate | 50.2% | 10-16% | Significant improvement |
| SNLI-H collapse rate | 83% | 37-67% | Moderate improvement |
| BoolQ collapse rate | 8.7% | 22-29% | Slight worsening |
| Training speed | ~7s/step | ~10-13s/step | ~60% slower |
| GPU memory | ~40GB | ~60GB | Higher |

g16 helps most where it's needed most (SNLI), but doesn't fully solve the SNLI-H problem. BoolQ actually shows slightly more collapse with g16, possibly because the larger group allows the already-good BoolQ policy to more consistently produce similar high-reward outputs.

## 7. Checkpoint Sweep Results (g16)

A comprehensive checkpoint sweep evaluated both multi and v2 g16 models at ~0.25ep intervals across all datasets. This reveals the optimal training duration and the dramatic superiority of the v2 single reward.

### SNLI-Premise

| Checkpoint | ~Epoch | v2 ΔLFR | multi ΔLFR |
|------------|--------|---------|------------|
| ckpt-900   | 0.06ep | -1.0%   | —          |
| ckpt-4000  | 0.25ep | **+27.2%** | +12.0% |
| ckpt-8000  | 0.50ep | **+28.7%** | **+12.4%** |
| ckpt-9900  | 0.62ep | —       | +8.3%      |
| ckpt-12000 | 0.75ep | +21.3%  | —          |
| ckpt-14100 | 0.88ep | —       | +6.1% (collapsed: NED=0.838) |
| ckpt-16000 | 1.0ep  | +24.4%  | —          |

v2 peaks at 0.5ep (+28.7%), surpassing DPO's best (+17.3%) by 11pp. Multi peaks at 0.5ep (+12.4%) but collapses after 0.62ep.

### SNLI-Hypothesis

| Checkpoint | ~Epoch | v2 ΔLFR | multi ΔLFR |
|------------|--------|---------|------------|
| ckpt-900   | 0.06ep | -2.3%   | —          |
| ckpt-4000  | 0.25ep | +2.8%   | -11.5%     |
| ckpt-8000  | 0.50ep | +6.3%   | -8.8%      |
| ckpt-9900  | 0.62ep | —       | -18.2%     |
| ckpt-12000 | 0.75ep | +6.2%   | -11.8%     |
| ckpt-16000 | 1.0ep  | **+7.3%** | -16.5%  |

v2 steadily improves, reaching +7.3% at 1.0ep (matching DPO's +7.5%). Multi is consistently harmful at all checkpoints.

### BoolQ

| Checkpoint | ~Epoch | v2 ΔLFR | multi ΔLFR |
|------------|--------|---------|------------|
| ckpt-900   | 0.11ep | -8.3%   | -9.3%      |
| ckpt-2000  | 0.25ep | -36.5%  | -24.7%     |
| ckpt-4000  | 0.50ep | -35.4%  | *pending*  |
| ckpt-6000  | 0.75ep | *pending* | —        |
| ckpt-6800  | 0.85ep | —       | *pending*  |

Both v2 and multi degrade rapidly. BoolQ GRPO fails at all training durations.

### Key Findings

1. **v2 single-reward >> multi-reward**: The confidence-weighted single reward (`flip + confidence * similarity`) dramatically outperforms decomposed rewards (FlipReward + SimilarityReward) on all datasets. On SNLI-P, v2 reaches +28.7% vs multi's +12.4%; on SNLI-H, v2 reaches +7.3% vs multi's best of -8.8%.

2. **Optimal training duration varies**: SNLI-P peaks at 0.5ep then declines; SNLI-H steadily improves through 1.0ep. This suggests different datasets benefit from different early stopping strategies.

3. **Multi-reward collapses catastrophically**: At 0.88ep, SNLI-P multi produces degenerate output (NED=0.838, PPL=18738). The separate reward decomposition appears to create conflicting optimization pressures that destabilize training.

4. **BoolQ is structurally incompatible with GRPO**: Performance degrades from the earliest checkpoint and worsens dramatically. The long BoolQ passages may create reward signals that don't align with the evaluation's label flip criterion.

## 8. BoolQ: Format Collapse Analysis

While SNLI-H struggles with reward advantage collapse (section 2), BoolQ fails via a different mechanism: **format collapse**. Analysis of the BoolQ v2 g16 training logs (job 2579220) reveals:

### Training Progression

| Epoch | Reward | CompLen | Valid/16 | ZeroStd | Entropy |
|-------|--------|---------|----------|---------|---------|
| 0.00  | 1.80   | 153     | 16/16    | 0%      | 0.054   |
| 0.10  | 1.95   | 47      | 16/16    | 0%      | 0.061   |
| 0.20  | 0.92   | 108     | 16/16    | 0%      | 0.016   |
| 0.30  | 1.79   | 130     | 16/16    | 0%      | 0.037   |
| 0.35  | 0.81   | 287     | 14/16    | 0%      | 0.033   |
| 0.40  | 0.00   | 512     | 0/16     | 100%    | 0.007   |
| 0.50  | 0.00   | 512     | 0/16     | 100%    | 0.051   |
| 0.60  | 0.00   | 468     | 0/16     | 100%    | 0.013   |
| 0.75  | 0.00   | 512     | 0/16     | 100%    | 0.013   |

### Failure Mode

1. **Completion length explosion** (0.3-0.4ep): Mean completion length jumps from ~130 to 512 (max_completion_length). The model generates maximum-length text that hits the token limit.

2. **Format collapse** (0.35ep+): The model stops producing valid `<edit>` tags. `parse_edit_tag()` returns None, so all completions are treated as invalid (valid=0/16).

3. **Total reward collapse** (0.4ep+): With all completions invalid, all rewards=0.0, reward_std=0, frac_reward_zero_std=1.0. GRPO computes zero advantages and zero gradients. The model is stuck with no signal to recover.

### Why BoolQ Specifically

BoolQ passages are 100-500+ words. Model completions are correspondingly long (130-300 tokens) compared to SNLI-P (16-40 tokens) and SNLI-H (11-18 tokens). With long outputs, the model can easily drift into generating passage-like text that omits the `<edit>` tag formatting. The large output space gives more room for format degeneration.

Critically, entropy does NOT collapse to zero on BoolQ (it stays at 0.01-0.05 throughout). The model maintains output diversity at the token level -- it just loses the structural formatting. This contrasts with SNLI-H where entropy collapse causes identical outputs.

### Comparison with SNLI-P v2 (successful)

SNLI-P v2 maintains healthy training throughout 1.0ep: completion lengths stay in 16-39 tokens, valid outputs remain 16/16 for most steps, and reward signal is sustained. The short completion length leaves little room for format drift.

## 9. Multi-Reward v2: Addressing Format Collapse

The original multi-reward (flip + similarity) collapsed catastrophically on all datasets. Multi-reward v2 addresses this with two additional reward signals:

### New Reward Components

1. **GatedConfidenceReward**: Returns the base model's confidence score when the label flipped, 0.0 otherwise. Uses a `PredictionCache` shared with `FlipReward` to avoid redundant model calls. This incentivizes high-confidence flips without rewarding high-confidence non-flips.

2. **FormatReward**: Binary reward — 1.0 if the completion contains valid `<edit>...</edit>` tags, 0.0 otherwise. Directly addresses BoolQ's format collapse failure mode (section 8).

### MV2 Training Results

**SNLI-P and SNLI-H**: Training completed successfully (1.0ep). FormatReward stayed at 1.0 throughout — format collapse is not an issue for SNLI. Entropy collapse eventually occurs as expected. Evaluations pending.

**BoolQ**: Timed out at ~0.83ep but showed a qualitatively different training trajectory than earlier runs:
- FormatReward *does* provide differential signal (not always 1.0)
- Model partially recovered from format collapse around 0.775ep
- However, catastrophic loss spikes persisted (loss to 86,680 at 0.34ep, grad_norm to 8.7B at 0.56ep)
- Root cause: learning rate 5e-6 is too aggressive for BoolQ's long outputs

### BoolQ Low-LR Hypothesis

The loss/gradient explosions on BoolQ suggest the default lr=5e-6 causes catastrophic overshooting. Two low-LR experiments submitted (lr=1e-6):
- BoolQ v2 g16 lr=1e-6 (single-reward)
- BoolQ mv2 g16 lr=1e-6 (multi-reward v2)

If the FormatReward can prevent format collapse AND a lower LR prevents gradient explosions, BoolQ GRPO may finally produce stable training.

## 10. GDPO: Per-Reward Normalization

Standard GRPO multi-reward sums rewards then normalizes advantages per group. This allows high-magnitude rewards to dominate the advantage signal. GDPO (Group reward-Decoupled normalization) reverses this:

1. Normalize each reward independently per group (subtract group mean, divide by group std)
2. Weight and sum the normalized per-reward advantages
3. Apply batch-level normalization to the combined advantages

Implemented as `GDPOTrainer` in `gdpo_trainer.py`, subclassing TRL's `GRPOTrainer`. Uses `_RewardCapture` wrappers around reward functions to intercept raw per-reward outputs, then recomputes advantages using the GDPO formula. Training script: `train_gdpo.py`.

GDPO training jobs submitted for all 3 datasets (g16, 1ep, lr=5e-6). Uses the same mv2 reward set (flip, similarity, gated_confidence, format).

## 11. Summary of Experiments

| Experiment | Status | Key Result |
|------------|--------|------------|
| g4 2ep (6 models) | Completed | SNLI-P multi +7.2%, rest negative |
| g4 SNLI-H checkpoint sweep | Completed | Best at ckpt-400 (~0.1ep), -3.3% |
| g16 old-reward (6 models) | Completed | Baseline g16, ~0.8-1.1ep |
| g16 v2 corrected reward (3 models) | Completed | SNLI-P +28.7%, SNLI-H +7.3% |
| g16 checkpoint sweep (22 evals) | Completed | v2 >> multi; early stopping essential |
| g16 multi-reward v2 (3 models) | Completed | SNLI-P +27.5%, SNLI-H +15.1% (N=200) |
| BoolQ low-LR (v2 + mv2) | Completed | Stable training, ΔLFR still negative |
| GDPO g16 lr5e-6 (3 models) | Completed | Entropy collapse; SNLI-H -22.0% |
| GDPO g16 lr1e-6 fw3 (3 models) | Completed | Collapse resolved; ΔLFR near zero (BoolQ +0.2%, SNLI-P -0.5%, SNLI-H +0.1%) |
| g24 (9 models: GRPO v2/mv2 + GDPO) | Completed | g24 does not improve over g16; v2 g24 SNLI-H has PPL collapse (9651) |

## 12. Reward Formula Correction

The original single reward was `flip + 0.8 * similarity`, where the 0.8 weight was arbitrary. This has been corrected to `flip + confidence * similarity`, matching the DPO unified score structure. The v2 g16 jobs use the corrected formula. Multi-reward runs (separate FlipReward + SimilarityReward) are unaffected.

The reward correction proved transformative: v2 results on SNLI-P (+25.8% at N=200) and SNLI-H (+11.0% at N=200) are dramatically better than the old single-reward g4 results (+5.4% and -10.6%), suggesting that confidence weighting provides a much more informative gradient signal than a fixed weight.

## 13. N=200 Re-Evaluation

All best models re-evaluated at N=200 (up from N=100) for more reliable comparisons. All 13 N=200 evaluations completed.

- **GRPO mv2 remains the best overall method**: SNLI-P +27.5%, SNLI-H +15.1%, BoolQ +1.2%
- **All BoolQ GRPO variants positive at N=200**: v2 +0.4%, mv2 +1.2%, mv2 lr1e6 +1.1%. Base LFR at N=100 was ~43% vs ~37% at N=200, explaining the apparent poor performance at N=100
- **BoolQ SFT neutral at N=200**: ΔLFR -0.0% (was +4.1% at N=100 with 200step config)
- **BoolQ GRPO v2 has best LFR/NED**: 2.49, indicating very targeted edits (lowest NED of 0.151)
- **DPO SNLI-H improved substantially**: +21.0% (N=200) vs +7.5% (N=100), indicating high evaluation variance on this dataset at N=100
- **LFR/NED efficiency metric** added to evaluation pipeline

## 14. GDPO lr1e-6 Results

GDPO lr1e-6 with FormatReward weight 3x resolved the entropy collapse seen at lr5e-6 but ΔLFR remains near zero:

| Dataset | ΔLFR (lr5e-6) | ΔLFR (lr1e-6) | NED | PPL |
|---------|:--:|:--:|:---:|:---:|
| BoolQ | +1.1% | +0.2% | 0.153 | 11.6 |
| SNLI-P | +2.5% | -0.5% | 0.244 | 134.5 |
| SNLI-H | -22.0% | +0.1% | 0.307 | 283.8 |

The per-reward normalization amplifies noise when individual reward variance is low within a group. Larger group sizes (g24) provide marginal improvement on SNLI but the fundamental issue remains.

## 15. g24 Results (24 Generations per Prompt)

All 9 g24 training jobs completed (8 required checkpoint resume after 24h timeout). Full comparison with g16:

### g24 vs g16: ΔLFR Comparison

| Dataset | GRPO v2 g16 | GRPO v2 g24 | GRPO mv2 g16 | GRPO mv2 g24 | GDPO g16 lr1e6 | GDPO g24 lr1e6 |
|---------|:-----------:|:-----------:|:------------:|:------------:|:--------------:|:--------------:|
| BoolQ | +0.4% | +0.2% | **+1.2%** | -0.6% | +0.2% | -0.6% |
| SNLI-P | **+25.8%** | +8.6% | **+27.5%** | +22.2% | -0.5% | +1.1% |
| SNLI-H | +11.0% | +22.6% ⚠ | **+15.1%** | +12.6% | +0.1% | +1.1% |

⚠ GRPO v2 g24 SNLI-H has catastrophic PPL (9651), indicating incoherent text that happens to flip labels.

### Key Observations

1. **g24 does not improve over g16 for GRPO**: On SNLI-P, both v2 and mv2 g24 are substantially worse than g16. The best SNLI-P method remains mv2 g16 (+27.5% vs g24's +22.2%). On BoolQ, g24 is consistently slightly worse.

2. **PPL degradation at g24**: GRPO v2 g24 shows extreme perplexity on SNLI (769.8 on SNLI-P, 9651 on SNLI-H), much worse than v2 g16 (110.6 and 317.3). The larger group may be encouraging the model to exploit reward artifacts at the expense of fluency. GRPO mv2 g24 shows less PPL degradation (146.6 and 453.2), suggesting the format reward provides some regularization.

3. **GDPO g24 marginal improvement**: GDPO g24 shows +1.1% on both SNLI datasets, up from near-zero at g16. The larger group size helps GDPO's per-reward normalization slightly, but the improvement is too small to be practically significant.

4. **Training efficiency**: g24 takes ~50% longer per step than g16 (24 vs 16 completions per prompt). SNLI datasets require 24000 optimizer steps at g24 vs 16000 at g16, exceeding the 24h SLURM limit. Added `--resume_from_checkpoint` support to handle this.

### Conclusion

Increasing group size from 16 to 24 does not improve results and often degrades them. The additional computational cost (~2x for SNLI) is not justified. The g16 configuration remains optimal for all methods.

---

## 16. Root Cause Analysis: Why GDPO Underperforms

### Diagnosis

Examined actual GDPO counterfactual outputs from SNLI-Premise evaluation (N=200):

| Metric | GDPO Tuned | GRPO mv2 Tuned | Base |
|--------|:----------:|:--------------:|:----:|
| LFR | 51.4% | **79.7%** | ~52% |
| NED | 0.244 | 0.348 | 0.241 |
| PPL | 134.5 | **72.7** | 136.2 |
| Zero edits | 44 | 7 | — |

GDPO produces 6x more zero-edits than GRPO, with no LFR improvement over base.

### Root cause: Similarity reward dominates when flip signal is sparse

When no generations in a group flip the label (common early in training and on harder prompts):
- FlipReward = 0 for all generations → zero variance → zero advantage
- GatedConfidenceReward = 0 for all → zero variance → zero advantage
- SimilarityReward = high for all (model copies original) → non-zero variance → drives gradient
- FormatReward = 1 for all → zero variance → zero advantage

Result: only SimilarityReward contributes. Per-reward normalization amplifies it. The model learns "change as little as possible" which is the opposite of what's needed for label flips.

### Training log evidence

From GDPO SNLI-H logs:
- Entropy: 0.24 → 0.006 (severe collapse)
- Many batches: `flip_reward=0`, `gated_confidence=0` — only similarity drives learning
- `frac_reward_zero_std=1` in some steps — zero variance in all rewards within groups

### Gaps between our implementation and the GDPO paper

| Aspect | GDPO Paper | Our Previous Implementation |
|--------|-----------|---------------------------|
| KL penalty | kl_coef=0.0005-0.001 | beta=0.0 (none) |
| Conditioned rewards | Yes (gate easy on hard) | No |
| Dynamic sampling | filter_groups.enable=True | No |
| Clipping | clip_low=0.2, clip_high=0.28 | epsilon=0.2 (symmetric) |
| Batch size | 512 | ~4 prompts (64 samples) |

---

## 17. Root Cause Analysis: Why BoolQ Struggles

### Diagnosis

Examined actual BoolQ counterfactual outputs across methods:

**Passage length**: BoolQ passages average 562 chars (up to 3310 chars), ~8x longer than SNLI (avg 71 chars). This is the fundamental challenge.

**Wrong edit targets**: Many edits change parts of the passage that don't answer the question. Example: question "is here comes the sun a Beatles song?", edits change "best-known compositions" instead of the part identifying it as a Beatles song.

**Format collapse**: Model generates max-length text (512 tokens) without valid `<edit>` tags. This happens because BoolQ outputs are 130-300 tokens (vs SNLI's 11-40 tokens), and longer outputs have more opportunity for structural drift.

**DPO succeeds where GRPO fails**: DPO achieves +14.6% ΔLFR on BoolQ, proving the task is learnable with supervised signal. GRPO's struggle is a reward/exploration issue, not a fundamental task limitation.

### max_completion_length bottleneck

- `max_completion_length=512` tokens
- BoolQ passages: up to ~827 tokens
- Model must output full edited passage in `<edit>...</edit>` tags
- Truncation at 512 tokens cuts off the closing `</edit>` tag → format failure → zero reward → no learning signal

---

## 18. GDPO and BoolQ Improvements (Phase 1+2)

### 18.1 Conditioned Rewards (GDPO paper Sec 4.2)

**Problem**: SimilarityReward provides strong gradient signal even when no label flip occurs, pulling the model toward minimal edits.

**Solution**: Created `ConditionedSimilarityReward` — returns cosine similarity only when FlipReward=1. When the label doesn't flip, returns 0.0 instead of the similarity score. This mirrors the GDPO paper's `R_tilde_length` which requires `R_correct=1`.

**Design decision**: Only SimilarityReward is conditioned. GatedConfidenceReward is already gated on flip by design. FormatReward remains unconditional (it provides independent normalization signal for format compliance).

Implementation: `ConditionedSimilarityReward` class in `train_grpo.py`, enabled via `--conditioned_rewards` flag in `train_gdpo.py`.

### 18.2 KL Penalty

**Problem**: `beta=0.0` allows unconstrained policy drift → entropy collapse (0.24 → 0.006).

**Solution**: Changed GDPO default `beta` from `0.0` to `0.001` (paper uses 0.0005-0.001).

This adds a soft constraint keeping the policy close to the reference model, preventing the deterministic collapse observed in training logs.

### 18.3 Dynamic Sampling (Zero-Variance Group Filtering)

**Problem**: When all generations in a group have the same rewards (e.g., all fail to flip), group_std=0, advantages ≈ 0. These groups waste training steps with no learning signal.

**Solution**: In `gdpo_trainer.py`, detect groups where the combined weighted advantage has zero variance (`var < 1e-8`). Set those groups' advantages to exactly 0.0, preventing them from corrupting the batch-level normalization.

Inspired by DAPO's `filter_groups.enable=TRUE` used in the GDPO paper. Logs `zero_var_groups=N/M` to track how many groups are filtered.

### 18.4 Increased max_completion_length for BoolQ

**Problem**: BoolQ passages up to 3310 chars (~827 tokens) truncated at 512 tokens → format collapse.

**Solution**: `MAX_COMPLETION_LENGTH` env var in `run_grpo.sh` and `run_gdpo.sh`. Default remains 512 (backward compatible); BoolQ runs use 768.

### 18.5 Configurable Batch Size

**Problem**: GDPO paper uses train_batch_size=512; our effective batch was 4 prompts (64 samples). Batch-level normalization (step 3 of GDPO) is noisy with few samples.

**Solution**: `GRAD_ACCUM_STEPS` env var in `run_gdpo.sh`. Default 4 (backward compatible); experiments will test 8 and 16 for more stable batch normalization.

### 18.6 SFT Warm-Start for BoolQ

**Problem**: GRPO on BoolQ starts from a base model with poor format compliance for long passages.

**Solution**: `extract_sft_data.py` script that extracts successful counterfactuals from DPO evaluation output (those that flipped labels with low NED), formats them for SFT training. The resulting SFT checkpoint serves as the starting point for GRPO/GDPO, giving the model initial format compliance and edit quality.

### Expected impact

The conditioned rewards + KL penalty address GDPO's core failure mode (similarity domination + entropy collapse). If GDPO's per-reward normalization works as designed (better signal preservation), these fixes should bring GDPO performance close to or above standard GRPO on SNLI. For BoolQ, increased completion length addresses the mechanical bottleneck, and SFT warm-start provides the supervised initialization that DPO's success suggests is necessary.
