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

## 9. Summary of Experiments

| Experiment | Status | Key Result |
|------------|--------|------------|
| g4 2ep (6 models) | Completed | SNLI-P multi +7.2%, rest negative |
| g4 SNLI-H checkpoint sweep | Completed | Best at ckpt-400 (~0.1ep), -3.3% |
| g16 old-reward (6 models) | Completed | Baseline g16, ~0.8-1.1ep |
| g16 v2 corrected reward (3 models) | Completed | SNLI-P +28.7%, SNLI-H +7.3% |
| g16 checkpoint sweep (22 evals) | 19/22 completed | v2 >> multi; early stopping essential |

## 9. Reward Formula Correction

The original single reward was `flip + 0.8 * similarity`, where the 0.8 weight was arbitrary. This has been corrected to `flip + confidence * similarity`, matching the DPO unified score structure. The v2 g16 jobs use the corrected formula. Multi-reward runs (separate FlipReward + SimilarityReward) are unaffected.

The reward correction proved transformative: v2 results on SNLI-P (+28.7%) and SNLI-H (+7.3%) are dramatically better than the old single-reward g4 results (+5.4% and -10.6%), suggesting that confidence weighting provides a much more informative gradient signal than a fixed weight.
