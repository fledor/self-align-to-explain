# Post-Training Methods for Counterfactual Generation: Overview

Comparing post-training self-alignment methods for improving counterfactual generation quality. **Primary results** below fine-tune **Qwen/Qwen2.5-7B-Instruct** using QLoRA (4-bit, LoRA r=32, alpha=16, lr=5e-6) on three NLI/classification datasets. Scale-ups to **Qwen2.5-14B-Instruct** and **Qwen2.5-3B-Instruct** are fully model-native (see [RESULTS.md](RESULTS.md)). Use `submit_model_pipeline.sh` / `submit_3b_pipeline.sh` to switch models.

**Last updated**: Jun 30, 2026. **Paper-readiness pass**: full 12-cell × 6-method matrix (4 models × 3 datasets) is fair N=200 with medians for NED (NED>0) and PPL throughout. **Every featured cell now clears base** — the last holdout, **Llama SNLI-H DPO**, was lifted from −0.8% to **+0.5%** by a **β-sweep** (β=0.3 ckpt100, 90% parse, median PPL 146); the earlier "offline ceiling" was a default-β=0.1 artifact, not a fundamental limit (online RL still wins this cell decisively — GRPO multi +9.6%). Offline methods only marginally clear base here (DPO +0.5, SimPO +0.7, SFT +0.6). **New § Parse-gate exclusions** (§7) documents the 18 runs the 15%-of-base parse gate drops — all Llama, none Qwen — and the offline-vs-online `<edit>`-tag training asymmetry behind the Llama offline parse collapses.

### Qwen2.5-14B-Instruct scale-up

- **14B matrix complete** (6 methods × 3 datasets, all fair N=200). Full results: [RESULTS.md — 14B](RESULTS.md#qwen25-14b-instruct).
- **Fair base:** BoolQ → `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/`, SNLI-P → `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/`, SNLI-H → `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_14b_v2fix/`. Script: `run_eval_14b_fair.sh`.
- **Headline 14B fair N=200:** BoolQ best **SimPO** (+7.0%), **GDPO** (+6.0%), **GRPO single** (+2.9%); SNLI-P best **GRPO multi** (**+29.8%** fair, original mv2 1ep), **GDPO v6** (+26.5%), **SimPO**/**GRPO single** (+22.7% ea.; the v2fix GRPO-multi retrain only reached +15.0%); SNLI-H best **GRPO multi** (+14.9%), **SimPO** (+13.7%), **GRPO single** (+11.6%).
- **Charts:** [results_charts.html](results_charts.html).

### Qwen2.5-3B-Instruct scale-up

- **3B pipeline:** `submit_3b_pipeline.sh`, evals via `run_eval_3b_fair.sh`. Fair anchors: BoolQ → `evaluation_boolq_200s_dpo_1pair_qwen25_3b/`, SNLI-P/H analogous.
- **3B matrix complete** (6 methods × 3 datasets, all fair N=200). Hyperparam LR variants now also complete. See [RESULTS.md — 3B](RESULTS.md#qwen25-3b-instruct).
- **Headline 3B results:** SNLI-P best **GRPO single lr=1e-5** (**+27.7%**) > **SimPO** (+25.5%) ≈ **GRPO multi lr=1e-5** (+25.2%); **GDPO** (+18.5%); SNLI-H best **GDPO** (**+15.2%** fair), **DPO/GRPO multi** (+7.3% ea.), **SimPO β=3** (+6.8%); BoolQ best **GRPO single lr=1e-5** (+10.4%), **GRPO multi lr=1e-5** (+7.6%), **SimPO β=2** (+6.1%). **Key finding**: lr=1e-5 dramatically helps 3B GRPO — +10.8pp on SNLI-P single, +1.6pp on BoolQ multi vs default lr.

### KTO (Kahneman-Tversky Optimization)

- Training on 7B for all 3 datasets (**2811932–34**, RTXA6000). Uses `KTOTrainer` with expanded binary `(prompt, completion, label)` format. Tests KT asymmetric loss vs symmetric DPO log-ratio on same paired data. See Training Configuration in RESULTS.md.
---

## 1. Method Comparison

Best ΔLFR per method per dataset, using fair-base N=200 evaluations only.

| Dataset | DPO | SFT | GRPO mv2 | GDPO v6 | GRPO-fair |
|---------|:---:|:---:|:--------:|:-------:|:---------:|
| **BoolQ** | **+10.5%** | +1.8% | +1.7% | +1.1% | +0.1% |
| **SNLI-P** | +15.7% | +0.9% | **+27.5%** | +18.2% | +12.3% |
| **SNLI-H** | **+21.0%** | +1.7% | +15.1% | **+18.7%** | +6.8% |

**Each method has a clear strength**: GRPO mv2 dominates SNLI-P (+27.5%). **Sigmoid DPO** leads the headline BoolQ fair row (**+10.5%**, 48.0% LFR vs 37.5% base in that eval). **SimPO** best on the shared GRPO-mv2 CF export is **β3γ0.5** (**+21.8%**, 59.0% LFR vs 37.2% base — see `RESULTS.md` / [results_charts.html](results_charts.html)); the earlier **1p b4** SimPO default is **+9.8%** on that same export. On SNLI-H, **sigmoid DPO** leads among conservative offline runs (+21.0% vs GRPO +15.1%); **SimPO** with **β=3, γ=0.5** reaches **+29.7% ΔLFR** (grid best), beating the first SimPO run (+26.4%) and DPO, at **NED 0.428** and **~1202** mean PPL over finite CFs (other SimPO cells have **much** higher PPL — the objective is not a free lunch). A follow-up **`simpo_qual`** sweep (extra CPO α and β×γ points; [RESULTS.md](RESULTS.md) `simpo_qual_*` rows) reaches **+32.1% ΔLFR** at **~4.4k** mean PPL and **fewer parsed tuned CFs** — higher headline ΔLFR, worse fluency/coverage tradeoff. **SimPO SNLI-P** default **2p b4** stays **+27.1%** (beats **β3γ0.5** +23.5% on that dataset). **SFT SNLI-P** best fair N=200 on the GRPO-mv2 base is now **+0.9%** (2p b16 1ep / 2p b4 200step grid). GDPO v6 peak (+18.7% SNLI-H) is still the reference **GDPO** tradeoff when you want moderate NED/PPL. **DPO BoolQ follow-ups**: β=0.05 hurts; IPO lr=1e-6 fixes the −11.7% disaster but only **+0.3%** vs sigmoid.

**SimPO vs DPO column**: The table column “DPO” is standard **sigmoid** DPO only; SimPO is documented here and in `RESULTS.md` as its own method row.

**GRPO-fair confirms stabilization tricks hurt GRPO**: GRPO-fair (= GRPO mv2 + conditioned rewards, asymmetric clipping, KL penalty) underperforms vanilla GRPO mv2 on every dataset (−2.4pp BoolQ, −15.2pp SNLI-P, −8.3pp SNLI-H). This proves the tricks specifically benefit GDPO's per-reward normalization rather than being universally helpful regularization. See section 6.

**GDPO v6 with early stopping is competitive with GRPO**: On SNLI-P, GDPO v6 peaks at ckpt-2000 (~0.25ep): +18.2% ΔLFR, closing 2/3 of the gap to GRPO (+27.5%). On SNLI-H, GDPO v6 peaks at ckpt-6000 (~0.75ep): **+18.7% ΔLFR, surpassing GRPO mv2 (+15.1%) by 3.6pp**. GDPO v7 (32 groups) matched v6 exactly (+15.7% at full epoch), confirming more groups don't help. See section 5.

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

### SimPO (reference-free, CPOTrainer)

SimPO uses length-normalized average log probability as the implicit reward (no reference model). **SNLI-H β×γ grid** (6 cells, `submit_simpo_grid_snli_h.sh`, completed): **β=3, γ=0.5** wins on **ΔLFR (+29.7%)** and **NED (0.428)** with **~1202** mean PPL (finite CFs only; see `RESULTS.md`). **γ=0.5** consistently beats **γ=1.4** at the same β on ΔLFR. Several cells show **very high PPL** (e.g. ~16k–30k) despite decent ΔLFR — SimPO can maximize margin while assigning low likelihood to its own edits. **Duplicate** config **β=2, γ=1.4** in the grid (**+24.3%**) differed from the **first** β2γ1.4 run (**+26.4%**) by ~2pp (training/eval variance). First-run details: **BoolQ +9.8%** (second offline to sigmoid DPO; PPL 14.0 vs DPO 23.9). **SNLI-P +27.1%** ≈ GRPO. **PPL `nan` in old reports**: a handful of degenerate edits yielded **NaN loss**; averaging them poisoned the mean — fixed in `evaluate_models.py` (finite-only mean, stricter `compute_perplexity`).

### CF coverage and parse rates

Each eval generates 10 CFs per entry (N=200 → 2000 raw generations). Many fail to parse into the expected structured format and are **discarded**. The parse rate varies significantly across methods and represents a hidden quality dimension not captured by ΔLFR, NED, or mean PPL alone.

**7B parse rates (parseable CFs / 2000 attempts):**

| Method | BoolQ | SNLI-P | SNLI-H |
|--------|:-----:|:------:|:------:|
| Base (untuned) | 82% | 69% | 58% |
| DPO | 67% | 59% | 49% |
| SimPO β3γ0.5 | 70% | 61% (v2fix) | **24%** |
| GRPO mv2 g16 | — | 50% | 28% |

SimPO SNLI-H stands out: only **24%** of generations parse (484/2000), and **27/200 entries get zero valid CFs** — complete coverage failure for 13.5% of inputs. By contrast DPO always produces at least 2 CFs per entry.

**PPL is bimodal, not uniformly bad.** SimPO's mean PPL (~1202 on SNLI-H) is skewed by a ~10% tail of degenerate fragments (e.g. `"with letters"`, `"or possibly receiving"` — PPL >5k). The **median PPL is 161**, comparable to DPO (114) and GRPO (144). The high-PPL outputs are short garbled fragments rather than full sentences — SimPO's length-normalized objective incentivizes short, high-margin outputs that may not follow the structured generation format.

**Implication for ΔLFR comparison:** SimPO's +29.7% ΔLFR is computed over only 484 CFs while DPO's +21.0% is over 977 CFs. Since SimPO fails on harder entries (those with 0 valid CFs), its ΔLFR is measured on a **self-selected easier subset**. A coverage-adjusted comparison would need to account for entries where SimPO produces nothing.

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
- **SNLI-P**: fair N=200 on GRPO-mv2 base: 2p b16 1ep / 2p b4 200step (**+0.9%**); other slices still show 1p b4 200step (+2.7%) at N=100 / different bases — see `RESULTS.md`
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
| ~~Dynamic sampling (zero-variance filtering)~~ | ~~Yes~~ *Removed (v2fix)* | No |

These techniques (borrowed from the GDPO paper and DAPO) improve training stability and reward alignment. When GDPO v6 outperforms GRPO mv2 on SNLI-H (+18.7% vs +15.1%), it is unclear whether the gain comes from **per-reward normalization** (GDPO's core contribution) or from the additional stabilization tricks.

### Fair Comparison Runs (submitted)

To isolate the effect of per-reward normalization, we submitted **GRPO-fair** runs that give GRPO the same tricks as GDPO v6:

| Parameter | GRPO mv2 (current) | GRPO-fair (new) | GDPO v6 |
|-----------|:------------------:|:---------------:|:-------:|
| Reward functions | SimilarityReward | **ConditionedSimilarityReward** | ConditionedSimilarityReward |
| epsilon_high | None (symmetric) | **0.28** | 0.28 |
| beta (KL) | 0.0 | **0.0005** | 0.0005 |
| Learning rate | 5e-6 | 5e-6 | 5e-6 |
| Batch size | 1×4 (eff. 4) | **1×4 (eff. 4)** | 8×1 (eff. 8) |
| generation_batch_size | 16 | **16** | 128 |
| num_generations | 16 | 16 | 16 |

Runs submitted: SNLI-P (train 2714737, eval 2714738), SNLI-H (train 2714739, eval 2714740), BoolQ (train 2714741, eval 2714742). Note: batch size matches GRPO mv2 (BS=1, gen_batch=16) since GRPO doesn't use multi-group normalization.

### GRPO-fair Results

| Dataset | GRPO mv2 | GRPO-fair | GDPO v6 | Impact of tricks on GRPO |
|---------|:--------:|:---------:|:-------:|:------------------------:|
| **BoolQ** | +1.7% | +0.1% | +1.1% | −1.6pp |
| **SNLI-P** | +27.5% | +12.3% | +18.2% | −15.2pp |
| **SNLI-H** | +15.1% | +6.8% | +18.7% | −8.3pp |

**Conclusion**: The stabilization tricks **hurt** GRPO across every dataset by 2-15pp, while they clearly help GDPO. This confirms that GDPO's advantage on SNLI-H (+18.7% vs GRPO's +15.1%) comes from genuine benefit of **per-reward normalization** interacting with these tricks, not from the tricks alone. The tricks are specifically tuned to GDPO's normalization-then-sum architecture and actively harmful for GRPO's sum-then-normalize approach.

Note: GRPO-fair SNLI-P training timed out at epoch 0.995 (checkpoint-15900 evaluated). GRPO-fair BoolQ PPL was NaN (generation quality issue), but LFR confirms no meaningful improvement.

### GDPO batch size across model scales

GDPO's normalization operates within each group of `num_generations=16` completions per prompt, so `per_device_batch_size` only affects the number of simultaneous groups per gradient step (gradient diversity), not the per-reward normalization quality directly. Nevertheless, for consistency the following batch sizes were used:

| Model | BATCH_SIZE | gen_batch | Partition | Notes |
|-------|:----------:|:---------:|-----------|-------|
| Qwen 7B | 8 | 128 (8×16) | RTXA6000 | Reference config |
| Qwen 3B | 8 | 128 (8×16) | RTXA6000 | Same as 7B |
| Qwen 14B | **4** | 64 (4×16) | H200 | Model too large for bs=8 on any available 48 GB GPU |
| Llama 3.1 8B | 8 | 128 (8×16) | H200 / A100-80GB | OOM at bs=8 on RTXA6000 (needed 44.6 GB, card has 44.4 GB); resubmitted on H200/A100-80GB for parity with 7B/3B |

**Comparison fairness**: Llama 8B ↔ 7B/3B GDPO comparisons are fully fair (same bs=8). Qwen 14B uses bs=4 due to model size constraints; comparisons involving 14B GDPO should note this. The practical impact is expected to be small — 14B at bs=4 achieved the best GDPO result overall (+26.2% on SNLI-P).

---

## 7. Evaluation Methodology

- **Fair-base comparison**: base model counterfactuals are generated once and reused across all model evaluations via `--base_eval_dir`, ensuring ΔLFR differences reflect the fine-tuned model's quality rather than base model generation variance
- **Standard evaluation**: 200 validation samples, 10 CFs per sample, fair-base reused (previously 100; standardized to 200 for lower variance)
- **LFR/NED efficiency metric**: label flip rate divided by normalized edit distance, rewarding methods that achieve high flip rates with minimal edits (higher = more efficient). Reported in N=200 eval reports.
- **NED aggregation (updated)**: Best-result N=200 NED values now use **median NED over CFs with NED > 0** (identical copies excluded — they are not counterfactuals by definition, and their inclusion inflates the LFR denominator unfairly; see discussion in `RESULTS.md` and RGF, ACL 2022). Historical/secondary entries still report mean NED from `eval_summary.avg_norm_edit_distance`. Note: NED=0 entries **are still counted in the LFR denominator** — they represent a genuine model failure to produce an edit.
- **Base model as judge**: the un-fine-tuned base model classifies all counterfactuals (both base-generated and fine-tuned-generated) to determine label flips
- **Evaluation variance**: base LFR varies 2-5pp across independent runs; differences below ~5pp should be treated cautiously

### Parse-gate exclusions

**The gate.** A counterfactual only counts if the model emits a parseable `<edit>…</edit>` span (`utils.parse_edit_tag`). Parse rate = parsed CFs / (N×10). For featuring a result we require **parse ≥ 15% of that cell's base parse rate**; runs below the gate are excluded from selection (the charted run is the highest-ΔLFR run *among gate-passers*). Rationale: a run that only parses a few percent of the time reports LFR over a tiny, self-selected sample (the model only "answers" on the inputs it happens to handle), so a high ΔLFR there is an artifact, not a real improvement. If *every* run in a cell were sub-gate the code falls back to all runs — but this never happens; every cell has at least one gate-passing run.

**Scope — this only ever bit Llama-3.1-8B.** All 18 gate-excluded runs are Llama; **no Qwen-3B / 7B / 14B run was excluded** (the lowest 7B run, SNLI-H GDPO g16 at 18.5%, still clears its 8.7% gate). This is because Llama's base parse is very high (~89% on every dataset → gate ≈ 13.3%), while several of Llama's *offline* runs at aggressive LR collapse their parse to single digits. (Gates: 3B 6.4–13.4%, 7B 8.7–12.4%, 14B 9.3–13.2%, Llama 13.3–13.4%.)

**Cells where the gate actually changed which run is featured (4, all Llama):**

| Cell | Excluded (higher ΔLFR, sub-gate) | Featured (gate-passing) |
|------|----------------------------------|--------------------------|
| Llama SNLI-H DPO   | lr=2e-5 **+9.1%** @ 2.3% parse | β=0.3 ckpt100 **+0.5%** @ 90% parse |
| Llama SNLI-H SimPO | β2γ0.5 **+11.6%** @ 4.6% parse | β3γ0.5 **+0.7%** @ 70% parse |
| Llama SNLI-H SFT   | 2ep **+2.8%** @ 11.1% parse | ckpt100 **+0.6%** @ 72% parse |
| Llama SNLI-P DPO   | 1pair lr=1e-5 **+10.3%** @ 3.9% parse | 2pair lr=2e-6 **+2.4%** @ 74% parse |

In every other cell the gate dropped some weak candidate runs but the best run was already gate-passing, so the featured pick is unchanged.

**Full list of the 18 gate-excluded runs** (parse < 15% of base; ΔLFR shown is the *reported* value over the collapsed sample and is not trustworthy):

| Model | Dataset | Method | Parse | Gate | reported ΔLFR | Run dir |
|-------|---------|--------|-------|------|---------------|---------|
| Llama | BoolQ  | DPO        |  5.2% | 13.3% |  +3.4 | `..._boolq_200s_dpo_1pair_lr1e5_llama31_8b` |
| Llama | BoolQ  | DPO        |  6.5% | 13.3% |  +2.0 | `..._boolq_200s_dpo_sigmoid_lr1e5_llama31_8b` |
| Llama | BoolQ  | GRPO multi | 11.6% | 13.3% | −47.4 | `..._boolq_200s_grpo_mv2g16_lr1e5_fair_llama` |
| Llama | BoolQ  | GRPO multi | 11.8% | 13.3% | −46.8 | `..._boolq_200s_grpo_mv2g16_lr1e5_llama31_8b` |
| Llama | SNLI-H | DPO        |  0.9% | 13.4% |  −3.3 | `..._snli_hypothesis_200s_dpo_2pair_lr5e5_llama31_8b` |
| Llama | SNLI-H | DPO        |  2.3% | 13.4% |  +9.1 | `..._snli_hypothesis_200s_dpo_2pair_lr2e5_llama31_8b` |
| Llama | SNLI-H | DPO        |  3.6% | 13.4% |  −0.6 | `..._snli_hypothesis_200s_dpo_2pair_lr1e5_fair_llama31_8b` |
| Llama | SNLI-H | DPO        |  4.2% | 13.4% |  −1.6 | `..._snli_hypothesis_200s_dpo_2pair_lr1e5_llama31_8b` |
| Llama | SNLI-H | DPO        |  8.9% | 13.4% | −13.2 | `..._snli_hypothesis_200s_dpo_1pair_lr1e5_llama31_8b` |
| Llama | SNLI-H | SFT        |  9.7% | 13.4% |  −6.5 | `..._snli_hypothesis_200s_sft_2ep_fair_llama31_8b` (stale-resume; discard) |
| Llama | SNLI-H | SFT        | 11.1% | 13.4% |  +2.8 | `..._snli_hypothesis_200s_sft_2ep_llama31_8b` |
| Llama | SNLI-H | SimPO      |  0.2% | 13.4% | −15.7 | `..._snli_hypothesis_200s_simpo_b3g05_llama31_8b` |
| Llama | SNLI-H | SimPO      |  0.3% | 13.4% | −38.7 | `..._snli_hypothesis_200s_simpo_b3g05_fair_llama31_8b` (diverged) |
| Llama | SNLI-H | SimPO      |  4.6% | 13.4% |  +7.2 | `..._snli_hypothesis_200s_simpo_b2g05_llama31_8b` |
| Llama | SNLI-H | SimPO      |  4.6% | 13.4% | +11.6 | `..._snli_hypothesis_200s_simpo_b2g05_fair_llama31_8b` |
| Llama | SNLI-P | DPO        |  2.6% | 13.3% |  −9.1 | `..._snli_premise_200s_dpo_2pair_lr1e5_llama31_8b` |
| Llama | SNLI-P | DPO        |  3.9% | 13.3% | +10.3 | `..._snli_premise_200s_dpo_1pair_lr1e5_llama31_8b` |
| Llama | SNLI-P | DPO        | 13.2% | 13.3% |  +2.2 | `..._snli_premise_200s_dpo_2pair_lr5e6_llama31_8b` (just under gate) |

**Why these collapsed — a known asymmetry, not a silent bug.** Every excluded run is an *offline* method (DPO/SimPO/SFT) at an aggressive LR, plus two lr=1e-5 GRPO-multi runs that genuinely diverged. The offline preference/SFT targets are trained on the **bare edited text with the `<edit>…</edit>` wrapper stripped** (`construct_dpo_pairs.py` uses `chosen_cf["edited_text"]`; `train_sft.py` uses `prompt + chosen`), whereas the eval parser and the online GRPO/GDPO rewards (`FormatReward`, `MinimalityReward`) require the tags. At low LR the few-shot prompt keeps the format and parse stays healthy (DPO/SimPO/SFT featured runs are 70–94% parse); at high LR the offline objective overrides the prompt and the model stops emitting tags, collapsing parse. The gate is what prevents these collapsed-but-high-LFR runs from being mistaken for wins. Fully removing the asymmetry would require wrapping the offline training targets in `<edit>` tags and retraining DPO/SimPO/SFT — deferred (results already clear the gate in every featured cell).

### General-capability benchmarks (MMLU + ANLI)

All 76 models (4 base + 72 best adapters) were evaluated zero-shot with `lm-eval` on MMLU and ANLI (r1/r2/r3) to check whether CF fine-tuning degrades general capability. **It does not.** Across all 72 adapters, ΔMMLU vs base averages **+0.04pp** (range −0.37 to +0.21) and ΔANLI averages **+0.07pp** (range −0.68 to +1.19); the largest single MMLU drop is **−0.4pp** (Llama SNLI-P GRPO-single). LoRA-based CF training is minimally invasive — it does not trade away general knowledge or NLI ability. Full per-cell table: `BENCHMARKS.md`. (Base references: Qwen-3B MMLU 65.2 / Qwen-7B 71.0 / Qwen-14B 78.8 / Llama-8B 67.8.)

### Training data sizes (reference)

Counts for DPO preference JSONL (see `RESULTS.md` main table for the legacy 7B 1-pair / 2-pair totals). **v2fix** (April 2026): SNLI pair construction filters rejected CFs to the same `target_label` as the chosen CF; BoolQ unchanged (binary labels). 1-pair v2fix: SNLI-P **1,919** entries with pairs (was 1,955, −1.8%); SNLI-H **1,626** (was 1,735, −6.3%). **14B BoolQ DPO** (native pipeline): **1,531** pairs in `dpo_pairs_boolq_2000e40c_1pair_qwen25_14b`. 2-pair v2fix JSONL for SNLI re-training lives under `dpo_pairs_snli_*_2000e40c_v2fix/` once the rebuild job completes.

### 14B parity (no cross-size mixing)

For the **14B-native BoolQ DPO** path, CF generation, CF evaluation, pair construction, DPO training, and `evaluate_models.py` (with `--base_model Qwen/Qwen2.5-14B-Instruct`) all use **Qwen2.5-14B** — no 7B checkpoints or 7B-generated preference files. **14B GRPO / GDPO** jobs set `MODEL_NAME_OR_PATH` to the 14B instruct checkpoint; SNLI 14B still needs its own CF/pair dirs before any 14B SNLI DPO. Mixing only occurs if you manually reuse a 7B `evaluation_*` tree as `--base_eval_dir` for a 14B adapter (disallowed for fair ΔLFR).

---

## 8. Current Status

### Completed
- All GRPO, DPO, SFT evaluations completed (g16 checkpoint sweep, v2, mv2, g24, mcl768, etc.)
- GDPO v0-v7: all trained and evaluated. v6 ckpt sweep: SNLI-P peak at ~0.25ep (+18.2%), SNLI-H peak at ~0.75ep (+18.7%)
- GDPO v6 SNLI-H beats GRPO mv2: +18.7% vs +15.1% (first dataset where GDPO outperforms GRPO)
- GDPO v7 (4 GPU, 32 groups, v6 innovations): +15.7% — confirms more groups don't help
- GDPO paper-match: 4-GPU GDPO (+7.6%) vs GRPO control (+25.9%) completed
- SFT LR sweep: lr=1e-5 and lr=2e-5 both negative; lr=5e-6 confirmed optimal
- GRPO SFT warm-start BoolQ: +1.2% (no improvement over cold-start GRPO +1.2%)
- SFT BoolQ ml2048: +1.8% (improvement over -0.0% at default ml1024)
- DPO BoolQ ml2048 (2pair b4): +8.5% ΔLFR (below 1pair +10.5%; 2pair is worse for BoolQ DPO)
- DPO BoolQ ml2048 (1pair b4): +3.8% ΔLFR (ml2048 hurts 1pair even more)
- GDPO v6 BoolQ mcl1024: +0.3% ΔLFR (longer completions didn't help GDPO on BoolQ)
- GRPO mv2 BoolQ mcl1024 ckpt-2000: −29.9% ΔLFR (severely degraded; confirms mcl1024 is harmful for GRPO on BoolQ)
- **GRPO-fair completed** (all 3 datasets): tricks hurt GRPO by 2-15pp across the board (see section 6)
- **DPO variants completed** (IPO, Robust, DiscoPOP on all 3 datasets): standard sigmoid DPO beats all variants everywhere. IPO catastrophic on BoolQ (-11.7%), DiscoPOP near-zero learning. See DPO Variant Results table above
- **SimPO SNLI-P / SNLI-H completed**: +27.1% / +26.4% ΔLFR (fair N=200); see `RESULTS.md` and SimPO subsection in section 2
- **GDPO v6 SNLI-H trick ablation completed**: removing conditioned rewards destroys performance (**-4.6%** vs +18.7% peak); removing KL inflates ΔLFR to +37.1% with PPL≈1360 — KL is a **quality** regularizer, not merely a cap on LFR
- **DPO sweep round 2 (BoolQ) completed**: SimPO **+9.8%**; DPO β=0.05 **+5.8%** (worse than β=0.1); IPO lr=1e-6 **+0.3%** (vs IPO −11.7% at lr=5e-6)
- **SimPO SNLI-H β×γ grid completed**: best **β=3, γ=0.5** — **+29.7%** ΔLFR, **NED 0.428**, ~**1202** mean PPL (finite CFs); see `RESULTS.md`
- **Code audit (April 7)**: Three bugs found and fixed. (1) GDPO zero-variance group filtering removed -- biased 25-63% of groups per batch. (2) Confidence default 0.5→0.0 for unparseable verifications. (3) SNLI DPO target-label mismatch fix -- 53% of pairs had mismatched labels. All affected runs resubmitted. See section 10.
- **14B pipeline restructured (April 7)**: All 14B runs now fully model-native (no 7B data). Prior 14B DPO (7B pairs) invalidated. `submit_model_pipeline.sh` created for easy model switching.

### Key findings
- **Each method excels on different datasets**: GRPO mv2 on SNLI-P (+27.5%), **sigmoid DPO** on BoolQ (+10.5%) and best offline on SNLI-H (+21.0%). **SimPO** is a strong second on BoolQ (+9.8%) and on SNLI-H can exceed DPO/GRPO on **ΔLFR** (up to **+29.7%**) but **PPL varies wildly** by (β, γ).
- **GDPO v6 beats GRPO on SNLI-H (at best checkpoint)**: GDPO v6 ckpt-6000 +18.7% vs GRPO mv2 +15.1% — still the best **GDPO** tradeoff vs GRPO when NED/PPL matter.
- **LR is dominant for both GDPO and GRPO multi**: GDPO v5 +1.7% at lr=1e-6 vs v6 +15.7% at lr=5e-6. **3B GRPO multi SNLI-P: +4.9% at default lr=5e-6 vs +25.0% at lr=1e-5 (+20pp)** — same model, same data, different LR. Optimal LR is scale-dependent: higher for small models, lower for large.
- **SimPO β sensitivity is scale-dependent**: β=2 fixes 3B BoolQ collapse (−13% → +6%) and helps SNLI-H slightly, but hurts 14B (β=3 remains better). β=2 also fixes Llama 8B SNLI-H collapse (−15.7% → +7.2%).
- **GRPO-fair confirms tricks are GDPO-specific**: Adding conditioned rewards, asymmetric clipping, and KL penalty to GRPO hurts it by 2-15pp. Specifically complement GDPO's per-reward normalization.
- **Early stopping is dataset-dependent**: GDPO peaks at 0.25ep on SNLI-P, 0.75ep on SNLI-H; GRPO at 0.5ep on SNLI-P.
- **Llama 8B: GDPO leads (+16.4% BoolQ, +10.7% SNLI-P)**: GDPO is the best-performing method for Llama 8B so far. DPO was suboptimal at lr=5e-6 and used wrong loss_type flag — proper lr=1e-5 sigmoid reruns in progress.

### DPO Variant Results (completed)

All variants tested with same hyperparameters as best standard DPO (lr=5e-6, beta=0.1, b4 2ep):

| Loss | BoolQ (1pair) | SNLI-P (2pair) | SNLI-H (2pair) |
|------|:---:|:---:|:---:|
| **Sigmoid (standard)** | **+10.5%** | **+15.7%** | **+21.0%** |
| IPO | -11.7% | +14.9% | +11.7% |
| Robust (ls=0.01) | +3.8% | +13.7% | +7.4% |
| DiscoPOP | +0.2% | -1.1% | +0.9% |

**Conclusion**: Standard DPO (sigmoid) dominates all variants on every dataset. IPO was catastrophic on BoolQ (-11.7%), possibly due to hyperparameter mismatch (IPO loss scale ~25 vs sigmoid ~0.7). **IPO lr=1e-6 on BoolQ** recovers to **+0.3%** (no longer harmful, but still far below sigmoid). **DPO β=0.05 on BoolQ** underperforms β=0.1 (**+5.8%** vs +10.5%). Robust DPO underperformed despite clean synthetic data not needing noise tolerance. DiscoPOP essentially failed to learn.

### GDPO v6 Trick Ablation (SNLI-H, full 1.0ep, fair N=200)

Compared to **peak** GDPO v6 (+18.7% at ~0.75ep), these are **full-epoch** checkpoints; the baseline row is **+17.6%** at 1.0ep for apples-to-apples duration.

| Ablation | ΔLFR | NED | PPL | Interpretation |
|----------|------|-----|-----|----------------|
| GDPO v6 (all tricks, 1.0ep) | **+17.6%** | 0.356 | 281.1 | Reference row (same duration as ablations) |
| **No conditioned rewards** | **-4.6%** | 0.216 | 345.9 | **Essential** — without conditioning, similarity reward fires on non-flips and training collapses |
| No asymmetric clipping | +13.0% | 0.336 | 336.4 | Helpful (~4–5pp vs full v6 at this checkpoint) |
| No KL penalty | +37.1% | 0.442 | **1360** | KL strongly **regularizes quality**; removing it maximizes LFR at the cost of disfluent, oversized edits — not a free win |

### Running / Pending

**All Qwen models complete (as of May 4):** 7B, 14B, and 3B fully evaluated. Key hyperparam variants complete: 3B GRPO multi lr=1e-5 (+25.0% SNLI-P), 3B SimPO β=2, 14B SimPO β=2. 14B DPO lr=2e-6 and 14B GRPO lr=2e-6 pending (resubmitted with sigmoid fix).

**Bug fix — DPO LOSS_TYPE=sigmoid (May 4):** All jobs submitted with `LOSS_TYPE=dpo` were silently failing — `train_dpo.py` accepts `sigmoid` not `dpo`. 7 affected jobs (Llama DPO lr=1e-5 all variants, 14B DPO lr=2e-6) resubmitted. Previous Llama DPO results (lr=5e-6) used `sigmoid` implicitly and are unaffected.

**SDPA attention (May 4):** `attn_implementation` changed from `"eager"` to `"sdpa"` in all training/eval scripts. This removes the naive O(n²) attention and uses PyTorch's fused CUDA kernels — meaningful speedup for GRPO/GDPO generation. No `flash_attn` package needed.

**Llama 3.1 8B (in progress — May 4):** DPO/SFT/GDPO complete. SimPO β=3 collapsed on all datasets; β=2 fixes SNLI-H (+7.2%) and BoolQ/SNLI-P evals resubmitted (2888681–82). GRPO training resumed on A100-80GB.

| Method | Dataset | Train status | Eval job |
|--------|---------|-------------|----------|
| DPO (anchor) | BoolQ / SNLI-P / SNLI-H | done ✓ | **2859764–66** (RUNNING — generates anchor) |
| SFT | BoolQ / SNLI-P / SNLI-H | done ✓ | **2860174–76** (dep: 2859764–66) |
| SimPO β3γ0.5 | BoolQ / SNLI-P / SNLI-H | done ✓ | **2860177–79** (dep: 2859764–66) |
| GRPO multi (mv2 g16) | BoolQ / SNLI-P / SNLI-H | resuming **2859767–69** | **2860180–82** (dep: anchors + resumes) |
| GDPO v6 (bs=8 H200) | BoolQ / SNLI-P / SNLI-H | training **2859808–10** | **2860183–85** (dep: anchors + training) |
| GRPO single (v2 g16) | BoolQ / SNLI-P / SNLI-H | TIMEOUT; BoolQ resumed **2887432** | BoolQ eval **2887445** (dep: 2887432); SNLI partial-epoch evals **2887440–41** |
| GRPO multi (BoolQ resume) | BoolQ | TIMEOUT; resumed **2887431** | BoolQ eval **2887444** (dep: 2887431) |
| SimPO β3γ0.5 BoolQ/SNLI-P | — | done ✓ (model exists) | FAILED (CUDA error); retried **2887442–43** |

**Hyperparameter sensitivity study (results in May 4):** Groups A–F complete. Group G still running (resumes submitted).

Key findings (complete):
- **Group B**: β=2 completely fixes 3B SimPO BoolQ collapse (+6.0% vs −13.0% with β=3)
- **Group C**: 0.75ep is optimal for 3B GDPO SNLI-H; 14B GDPO converges by 0.5ep
- **Group D**: 3B GRPO needs full epoch — 0.25ep/0.5ep barely improve
- **Group E**: 14B SimPO β=2 slightly worse than β=3 (scale-dependent — β=2 only helps at 3B)
- **Group F**: lr=1e-5 strongly improves 3B DPO on SNLI (+11.5% vs +7.2% SNLI-P; +6.3% vs +2.0% SNLI-H); lr=2e-6 hurts 14B DPO; **lr=1e-5 for 3B GRPO multi SNLI-P: +25.0% vs +4.9% default (+20pp!) — ties SimPO as best 3B SNLI-P**
- **Group G**: 3B GRPO lr=2e-6 SNLI-P: +8.6% (improves over default +4.9% but not as good as lr=1e-5 +25.0%); 14B still running

**Llama 8B hyperparam results (May 4):**
- **SimPO β=2 SNLI-H: +7.2%** — fixes β=3 collapse (−15.7%); now best Llama SNLI-H method
- **GDPO ckpt300 BoolQ: +14.5%** (median PPL 13.7) — slightly below full 1ep (+16.4%); 1ep remains best
- DPO lr=1e-5 submitted (sigmoid fix needed — see below); results pending

**Hyperparameter sensitivity study (submitted April 29):** 30 jobs queued to test whether 7B sweet spots generalize across scales.
- **Group A** — 3B DPO full epochs (200step → 2ep): BoolQ 1pair, SNLI-P/H 2pair (trains **2859875–77**, evals **2859893–95**)
- **Group B** — 3B SimPO softer margin (β=3 → β=2, γ=0.5): BoolQ + SNLI-H (trains **2859878–79**, evals **2859896–97**)
- **Group C** — GDPO early-stopping at 3B and 14B (eval-only): 3B GDPO SNLI-P ckpt-500/1000, SNLI-H ckpt-1000/1500; 14B GDPO SNLI-H ckpt-1000/1500 (**2859867–72**)
- **Group D** — 3B GRPO multi early-stopping on SNLI-P (eval-only): ckpt-4000/8000 (**2859873–74**)
- **Group E** — 14B SimPO softer margin (β=2, γ=0.5) on SNLI-H (train **2859880**, eval **2859898**)
- **Group F** — LR sensitivity: 3B DPO lr=1e-5 SNLI-P/H, 14B DPO lr=2e-6 SNLI-H, 3B GRPO multi lr=1e-5 SNLI-P (trains **2859881–84**, evals **2859899–2859902**)
- **Group G** — GRPO multi lr=2e-6 to test if lower lr recovers multi's gap vs single: 3B SNLI-P, 14B SNLI-P, 14B SNLI-H (trains **2860143–45**, evals **2860146–48**)

**7B v2fix re-runs** (April 7–11): trains **2765018** / **2765019** and evals **2771529** / **2771530** / **2771568** / **2771570** / **2773355**–**2773358** are **completed** — see [RESULTS.md](RESULTS.md). **7B GRPO BoolQ v2fix** fair eval **2782111** completed (`evaluation_boolq_200s_grpo_mv2_g16_v2fix/`); headline **−28.3%** ΔLFR — likely **policy collapse** after long training; see [RESULTS.md](RESULTS.md) BoolQ table and job note.

**7B SNLI pair reconstruction:** **2765026** / **2765027** **completed** (v2fix pair dirs). **SimPO SNLI-H simpo_qual** sweep **2771585**–**2771596** **completed** — extra rows in [RESULTS.md](RESULTS.md) (`simpo_qual_*`).

**Stale / cancelled:** 14B resume jobs 2764834, 2764835 (old code); duplicates 2765016, 2765017, 2765021, 2765022; **2765023** (14B GRPO fresh) **TIMEOUT** — superseded by **2773359** resume path + eval **2777871**.

**Eval parity note:** Any `evaluate_models.py` run completed *before* the tightened perplexity policy (finite-only CE in fp32; `None` only for `<2` tokens or non-finite loss) and related eval behavior is **not apples-to-apples** with newer reports. The jobs above were reset so pending/fresh evals use the current script; older completed rows in `RESULTS.md` remain comparable only where metrics were already computed from finite per-CF values—when in doubt, **re-run eval** on the same checkpoint (no retrain needed). If you re-run with `--resume`, delete `tuned_cfs_progress.jsonl` (or the whole output dir) first so old per-CF PPL values are not carried forward.

### Next steps (recommended priority)

1. **3B BoolQ evals:** Await SimPO/SFT/GRPO evals (**2819352–54**), GDPO interim eval (**2819409**), and 5ep GDPO retraining (**2819410**).
2. **3B SNLI GRPO multi evals:** Await **2819355–56**.
3. **3B + 14B GRPO single evals:** When **2819412–17** complete, submit fair evals via `run_eval_3b_fair.sh` / `run_eval_14b_fair.sh`.
4. **KTO results:** When **2811932–34** complete, eval and update RESULTS.md / results_charts.html.
5. **LoRA alpha ablation**: `lora_alpha=32` vs `16` on one dataset (section 10).
6. **Other base models**: `MODEL=... TAG=... ./submit_model_pipeline.sh`.

For full GDPO root cause analysis and design decisions, see the GDPO sections above (section 5).

---

## 9. Implementation Changes

### Conditioned Rewards (`train_grpo.py`)
- New `ConditionedSimilarityReward` class: returns similarity only when `PredictionCache` reports flip=True
- Enabled via `--conditioned_rewards` flag in `train_gdpo.py`
- `run_gdpo.sh`: `CONDITIONED_REWARDS=1` env var

### GDPO Trainer Improvements (`gdpo_trainer.py`)
- ~~Zero-variance group filtering~~ **Removed** (v2fix, April 7): the filtering zeroed 25-63% of groups per batch and biased batch normalization statistics. NVLabs reference implementation does not filter zero-variance groups — they produce zero per-reward advantages naturally and are handled correctly by batch normalization. See audit notes in section 10.

### BoolQ Completion Length
- `run_grpo.sh`, `run_gdpo.sh`: `MAX_COMPLETION_LENGTH` env var (default: 512, use 768 for BoolQ)

### SFT Warm-Start Pipeline (removed)
- `extract_sft_data.py` and `merge_lora.py` have been deleted; the SFT warm-start experiment was confirmed a failure (+1.2% ΔLFR, no improvement over cold-start GRPO)

### NED Penalty (`train_grpo.py`)
- `ConditionedSimilarityReward` extended with `ned_penalty_alpha` parameter
- When non-flipping: returns `-alpha * NED` instead of 0.0 (recovers signal from zero-variance groups)
- Configurable via `--ned_penalty_alpha` in `train_gdpo.py`, `NED_PENALTY_ALPHA` env var
- Experimental result: hurt BoolQ (-1.1% vs +0.4% without); SNLI impact unclear due to training instability

### DPO Loss Variants (`train_dpo.py`)
- `--loss_type` expanded: `sigmoid` (default DPO), `ipo` (bounded optimization), `robust` (noise-tolerant), `discopop` (LLM-discovered), `simpo` (reference-free, via CPOTrainer), `hinge`, `exo_pair`, `nca_pair`, `bco_pair`, `sppo_hard`, `apo_zero`, `apo_down`
- `--label_smoothing` added for Robust DPO (models annotation noise probability, 0.0-0.5)
- `--simpo_gamma` and `--cpo_alpha` added for SimPO (target reward margin and BC regularizer weight)
- SimPO uses `CPOTrainer` from `trl.experimental.cpo` instead of `DPOTrainer` — no reference model, length-normalized avg log probability as implicit reward
- `run_dpo_sweep.sh`: generic DPO sweep script accepting env vars for any loss type, LR, beta, etc.
- `submit_simpo_grid_snli_h.sh`: submits 6× SimPO SNLI-H (β×γ grid) + dependent fair evals; logs job IDs under `logs/simpo_grid_snli_h_submit_*.txt`

### Perplexity aggregation (`evaluate_models.py`)
- `compute_perplexity` returns `None` for very short strings, `<2` tokens after tokenization, or **non-finite** loss / exp(loss)
- `compute_metrics` averages only **finite** per-CF perplexities; `avg_perplexity` is `null` in JSON if none qualify; reports show **—**
- `ppl_valid_count` added to metrics dict for debugging coverage

### GDPO Beta Configuration (`run_gdpo.sh`)
- Added `BETA` env var to configure KL penalty coefficient
- Default: 0.001 (GDPO v2/v3), configurable to 0.005 (GDPO v4)

### Model Pipeline Orchestration (April 7)
- `submit_model_pipeline.sh`: parameterized orchestrator — set `MODEL` and `TAG` to run the full pipeline (DPO data prep + train, GRPO, GDPO) for any model. Handles sharded CF generation with SLURM dependencies.
- `run_prepare_dpo_data.sh`: SLURM wrapper for DPO data pipeline (generate CFs → evaluate → construct pairs). Supports `MODE=generate` (sharded), `MODE=build` (merge + eval + pairs), and `MODE=pairs` (merge + construct pairs only — skips CF re-evaluation when `*_evaluated.jsonl` already exists; use after a timed-out `build` or to rebuild pairs quickly).
- `run_dpo_sweep.sh`: now accepts optional `DATASET_PATH` env var to override the default pair directory, enabling model-specific pair paths.

---

## 10. Code Audit (April 7, 2026)

Thorough audit of GDPO, GRPO, and DPO implementations against the NVLabs GDPO reference (`trl-GDPO/trl-0.18.0-gdpo/trl/trainer/grpo_trainer.py`) and respective papers. Three bugs found and fixed; all affected runs resubmitted. Full audit plan: `.cursor/plans/training_method_audit_7709e119.plan.md`.

### Fix 1: GDPO zero-variance group filtering removed (`gdpo_trainer.py`)

**Bug**: Lines 101-116 zeroed out advantages for groups with zero combined variance, then included those zeros in batch normalization statistics. Training logs showed **25-63% of groups** were filtered per batch, significantly biasing the batch mean toward 0 and distorting the std. The NVLabs reference implementation has no such filtering — zero-variance groups produce zero per-reward advantages naturally and participate normally in batch normalization.

**Cause**: Added proactively (inspired by DAPO `filter_groups`), never validated with before/after comparison. The file was introduced in a single commit with the filtering already present.

**Fix**: Removed the filtering block entirely. Batch normalization now matches the NVLabs reference exactly: `advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)`.

**Impact**: All GDPO runs (7B and 14B). Resubmitted: 7B SNLI-H v6 (**2765019**), 14B SNLI-H fresh (**2765024**).

### Fix 2: Confidence default 0.5 → 0.0 (`train_grpo.py`)

**Bug**: When the verification model produced unparseable output, confidence defaulted to 0.5 instead of 0.0 (4 locations: `CounterfactualReward._predict_label`, standalone `_predict_label`, `CounterfactualReward.__call__` fallback, `PredictionCache.get` default). This inflated rewards for broken verifications — a failed parse contributed `0.5 * similarity` to the reward rather than 0.

**Fix**: Changed all 4 defaults from 0.5 to 0.0.

**Impact**: All GRPO and GDPO runs (they share the same reward functions). Resubmitted: 7B GRPO SNLI-P (**2765018**), 7B GDPO SNLI-H (**2765019**), 14B GRPO (**2765023**), 14B GDPO (**2765024**).

### Fix 3: DPO SNLI target-label mismatch (`construct_dpo_pairs.py`)

**Bug**: For SNLI (3 labels), the chosen and rejected CFs could target **different** labels. The 40 CFs per entry are split ~20/20 between two alternative labels. The pairing logic picked the globally best chosen and globally worst rejected regardless of target label. The prompt was formatted for the chosen's target (e.g., "change to neutral"), but the rejected might have been generated for a different target (e.g., "change to contradiction"). **53% of all SNLI pairs** had this mismatch. In the worst case, the rejected was a *successful* CF for a different target that the model learns to avoid.

**Fix**: Filter rejected candidates to the same `target_label` as the chosen CF before pairing.

**Impact**: SNLI-P and SNLI-H DPO/SimPO runs only (BoolQ has 2 labels -- no mismatch possible). 7B pair reconstruction completed: **2765026** (SNLI-P: 1955->1919 pairs), **2765027** (SNLI-H: 1735->1626 pairs). 14B SNLI pairs will need reconstruction when 14B SNLI DPO is attempted. DPO/SimPO SNLI re-training to follow.

### Open item: LoRA alpha/rank ratio

All three training scripts default to `lora_r=32, lora_alpha=16`, giving a scaling factor of `alpha/r = 0.5`. Standard practice is `alpha = r` (factor 1) or `alpha = 2*r` (factor 2). The 0.5 factor heavily attenuates LoRA updates. Changing alpha to 32 would require re-running all experiments — deferred pending a controlled ablation on one dataset.

### Additional observation: format_reward has zero variance

Training logs show `format_reward=0.000±0.000` in nearly every GDPO step — all completions produce valid format, so this reward contributes no learning signal. Consider removing it from the GDPO reward function list to reduce overhead (low priority).

---

## Per-Model Result Notes

Notes on each model/dataset combination — what was tried, what worked, and what to note.

### Qwen2.5-7B-Instruct

**BoolQ**: `GRPO mv2 g16 v2fix` (full 1ep, fair) shows **−28.3%** ΔLFR — likely policy collapse after long GRPO training. Compare only to short-run (~0.25ep) GRPO rows. All other methods positive.

### Qwen2.5-3B-Instruct

**BoolQ**: DPO +0.0% — 3B cannot learn BoolQ CFs from DPO pairs alone; 2ep DPO fair eval confirms −1.1% (more training hurts). SimPO β=3 collapses on 3B (−13.0%), β=2 gives modest +3.5%. GRPO single lr=1e-5 nominally +10.2% but ⚠ 22% of CFs have NED=0.000 (identical text, no edit); GRPO multi at +6.0% with NED=0.234 is more reliable — higher lr causes degenerate copy-paste outputs. GDPO: ckpt100 (+1.1%) marginally better than full 1ep (−0.1%), base_LFR 30.2% vs anchor 30.5% (~0.3pp generous bias). GRPO multi lr=1e-5 eval pending (job 2929565).

**SNLI-P**: GRPO multi lr=1e-5 canonical fair: **+25.2%** (was biased +25.3%). SimPO b3 canonical fair: **+24.4%** (was +25.5%). DPO 2pair lr=1e-5 fair: **+10.6%** (was +11.5%). GDPO v6 canonical: **+17.1%** (was biased +18.5%). GRPO single lr=1e-5 eval pending (job 2932698 — checkpoint path fix). GRPO mv2 ckpt4000 fair: **+1.9%** (intermediate checkpoint, confirms default-lr multi is weak at early stages).

**SNLI-H**: GDPO v6 ckpt1500 fair rerun is best at **+15.2%**. GRPO multi lr=1e-5 eval pending (job 2932697 — checkpoint path fix). GRPO single lr=1e-5 (+6.3%) outperforms default lr (+3.0%). GRPO multi fair rerun: **+7.1%** (up from biased +5.2%). GRPO single fair: **+2.3%** (down from biased +3.0%). DPO 2pair 2ep fair: **+2.1%** (new entry). SimPO median PPL 90.2.

### Llama-3.1-8B-Instruct

**BoolQ**: GDPO ckpt500 is best (+17.3%); checkpoint profile: ckpt300 +14.5% → ckpt500 +17.3% → ckpt700 +16.2% → full +16.4% (peak at ckpt500). DPO 2pair lr=1e-5 is second-best offline (+11.5%, NED=0.133 notably low). SimPO stable (lr=2e-6, grad clip): β=2 +9.5%, β=3 +7.9%. GRPO mv2 mcl768 ckpt1600 (+13.2%): pre-collapse peak — reward healthy at step 1601 then drops to 0 at step 1701. Standard GRPO (no mcl768) collapses earlier. DPO 1pair lr=1e-5 (5% parse) measured on self-selected subset. GRPO single mcl768 training submitted (job 2929500).

**SNLI-P**: GRPO single ckpt7500 at **+31.3%** (fair) is best. GRPO-fair multi is surprisingly competitive: ckpt4500 **+28.0%**, ckpt7000 **+29.0%** — much better than vanilla GRPO multi (+3% range). This is close to GRPO single, suggesting KL regularization + conditioned rewards significantly help multi-reward GRPO on this dataset. Profile: ckpt4500 +28.0% → ckpt7000 +29.0% → ckpt11200 (pending 2932699). SimPO (lr=2e-6): β=3 +18.6%, β=2 +17.4%. GDPO ckpt500 (+11.4%) beats full 1ep. DPO 2pair fails (3% parse). Remaining GRPO single ckpt10100/ckpt8500 results still pending fair correction.

**SNLI-H**: All 15 biased entries now have fair re-evals. **SimPO β=2 at +11.6%** is the new best result (was biased +7.2% — significant revision upward). SimPO β=3 confirmed collapsed (−38.7%, only 5 CFs). **GRPO multi ckpt14900** confirms +9.6% (stable). **GDPO ckpt1000** +6.2%, ckpt1500 +6.0%, full +5.5%. **DPO lr=2e-5** +9.1% (biased, fair rerun in progress 2932695); **DPO lr=5e-5** training complete, eval pending (2932694). DPO lr=1e-5 confirms −0.6%. SFT 2ep reverses from +2.8% (biased) to **−6.5%** (fair) — this is a significant finding: the +2.8% was entirely due to favorable base CFs, the model actually degrades vs base.

---

## Underexplored Combination Audit (May 2026)

Comprehensive audit of which model/method/dataset combinations had <2 config variants tried, and what was done.


### Coverage summary (configs per method, unique base hyperparameters)

| Model  | Dataset | DPO | SimPO | SFT | GDPO | GRPO-s | GRPO-m |
|--------|---------|-----|-------|-----|------|--------|--------|
| 7B     | BoolQ   | 4   | 3     | 2   | 3    | 2      | 3      |
| 7B     | SNLI-P  | 2   | 2     | 1   | 3    | 2      | 3      |
| 7B     | SNLI-H  | 2   | 2     | 1   | 3    | 2      | 3      |
| 14B    | BoolQ   | 4   | 2     | 1   | 2    | 3      | 2      |
| 14B    | SNLI-P  | 5   | 2     | 1   | 2    | 2      | 2      |
| 14B    | SNLI-H  | 5   | 2     | 1   | 2    | 2      | 2      |
| 3B     | BoolQ   | 4   | 2     | 1   | 2    | 3      | **1→2** |
| 3B     | SNLI-P  | 6   | 2     | 1   | 3    | **1→2**| 5      |
| 3B     | SNLI-H  | 6   | 2     | 1   | 3    | 3      | **1→2** |
| Llama  | BoolQ   | 8   | 2     | 1   | 4    | 3      | 5      |
| Llama  | SNLI-P  | 6   | 2     | 1   | 3    | 4      | 9+     |
| Llama  | SNLI-H  | 10  | 3     | 2   | 4    | 7      | 2      |

Bold **1→2** = previously 1 config, new eval submitted. 7B counts use different naming convention (not captured by automated scan, but all methods present in results).

### Gaps addressed in May 2026 batch

1. **3B BoolQ GRPO multi lr=1e-5** (job 2929565): model already trained, eval submitted. Prior single config used default lr; lr=1e-5 gave +25% on SNLI-P for multi — high-value test.
2. **3B SNLI-H GRPO multi lr=1e-5** (job 2929566): same rationale. Single lr=1e-5 improved SNLI-H single (+3.0%→+6.3%); multi may show similar gain.
3. **3B SNLI-P GRPO single lr=1e-5** (job 2929567): only 1 single config (default lr +16.9%); model already trained to 13200 steps.
4. **All 15 biased Llama SNLI-H entries** (jobs 2929543–2929557): GDPO/GRPO/SFT/DPO evals that used own base CFs (1.2–3.0pp bias) — replaced with anchor-anchored fair evals.
5. **1 biased Llama SNLI-P entry** (job 2929558): GRPO single ckpt7500 had -0.7pp anchor bias.
6. **3 biased 3B SNLI-P entries** (jobs 2929559–2929561): GRPO multi lr=1e-5, SimPO b3g05, DPO 2pair 2ep had +1.1–1.9pp bias.

### Fair re-eval policy

All reported results must reuse the canonical anchor base CFs. Any entry where the eval generated its own base CFs (base_LFR deviates from anchor by >0.5pp) is considered **unfair** and a canonical fair re-eval is submitted. The "rerun(±Xpp)" notation in tables indicates bias before fair re-eval completes; once the `_fair` eval dir exists its result supersedes.

**Complete fair re-eval sweep (May 2026)**:
- 58 total fair re-eval jobs submitted (2929456, 2929543–2929561, 2929594–2929629) covering:
  - Llama BoolQ: 8 entries (DPO 2pair lr=1e-5, GRPO single/multi early ckpts and full, GRPO mv2 lr=1e-5)
  - Llama SNLI-P: 6 entries (GRPO single ckpt5600/7500/10100, GRPO mv2 lr=2e-6 ckpt11700, SFT, SimPO b2)
  - Llama SNLI-H: 17 entries (GDPO 4, GRPO single 7, GRPO multi 2, DPO 3, SFT 1)
  - 14B BoolQ: 2 entries (SimPO b2/b3)
  - 14B SNLI-P: 2 entries (GDPO, GRPO multi)
  - 3B BoolQ: 4 entries (GRPO single, GRPO single lr=1e-5, SimPO b2/b3)
  - 3B SNLI-P: 6 entries (GDPO, GRPO mv2 ckpt4000/lr=1e-5/lr=2e-6, SimPO b2/b3)
  - 3B SNLI-H: 5 entries (DPO 2pair 2ep, DPO 2pair 2ep lr=1e-5, GRPO multi, GRPO single, SimPO b2)

**7B fair re-evals** (jobs 2929639–2929648): 10 biased 7B entries identified and re-evaluated. Created `run_eval_7b_fair.sh` anchored to `evaluation_boolq/snli_premise/snli_hypothesis_200s_grpo_mv2_g16` (the established 7B base). Biased entries: 6 BoolQ (DPO 1pair/2pair, GRPO mv2 mcl1024 ckpt2000, GRPO mv2 g24 lr=1e-6, GRPO v2 g16, SFT), 3 SNLI-P (DPO 2pair, GDPO g16, SFT), 1 SNLI-H (DPO 2pair).

- **SFT** has only 1 config everywhere (lr=5e-6, 200step). SFT LR sweep (1e-5 negative, 2e-5 NaN PPL on BoolQ) confirmed lr=5e-6 is optimal — defensible.
- **Llama BoolQ GRPO single mcl768**: training (job 2929500) — will confirm if mcl768 fix generalizes beyond multi-reward variant.
- **Llama SNLI-H DPO lr=5e-5**: training (job 2929501) — will determine peak LR regime for this dataset.

---

## Training Configuration

By default, reported runs use QLoRA (4-bit quantization), LoRA r=32, alpha=16, lr=5e-6 on **Qwen/Qwen2.5-7B-Instruct**. **Qwen2.5-14B-Instruct** experiments are expected to use the same recipe unless memory limits require smaller effective batch or adjusted `max_completion_length` / generations (especially for GRPO/GDPO).

- **b4**: batch size 1, gradient accumulation 4 (effective batch 4)
- **b16**: batch size 4, gradient accumulation 4 (effective batch 16, requires A100-80GB)
- **200step**: `max_steps=200` (effective epochs vary: 1pair b4 ~0.4-0.5ep, 1pair b16 ~1.6-1.9ep, 2pair b4 ~0.2-0.3ep, 2pair b16 ~0.8-1.0ep)
- **DPO pairs**: chosen = label-flipping CFs, rejected = non-flipping CFs; 2pair = best+worst and 2nd-best+2nd-worst; 1pair = best+worst only
- **SFT**: trains on chosen examples only (no rejected/contrast data)
- **GRPO g4/g16**: online RL, 4 or 16 generations per prompt, temperature=1.2, combined or decomposed reward. g4 trained for 2ep; g16 trains for 1ep (see [GRPO_ANALYSIS.md](GRPO_ANALYSIS.md) for epoch analysis)
- **GRPO † (old reward)**: single reward = `flip + 0.8 * similarity` (no confidence weighting)
- **GRPO v2 (corrected reward)**: single reward = `flip + confidence * similarity` (matches DPO unified score)
- **GRPO mv2 (multi-reward v2)**: 4 decomposed rewards: FlipReward + SimilarityReward + GatedConfidenceReward + FormatReward
- **GRPO lr1e6**: same reward as parent config but trained at lr=1e-6 (vs default 5e-6) to prevent BoolQ instability
- **GDPO**: GRPO with per-reward normalization (normalize each reward independently per group, then sum). Uses mv2 reward functions. See `gdpo_trainer.py`
- **g24**: experiments with 24 generations per prompt (vs 16) for better group diversity and reward signal. SNLI g24 requires ~24000 optimizer steps (2x BoolQ due to 2 prompts/entry), exceeding the 24h SLURM limit; resumed from checkpoints
- **GDPO lr1e6**: GDPO with lr=1e-6 (vs default 5e-6) and FormatReward weight 3x to prevent entropy collapse
- **GDPO v2 (conditioned)**: GDPO with conditioned rewards (similarity gated on flip), beta=0.001 KL penalty, lr=1e-6. ~~Dynamic sampling (zero-variance group filtering)~~ removed in v2fix (April 7) -- biased batch normalization. See [GDPO_BOOLQ_IMPROVEMENTS.md](GDPO_BOOLQ_IMPROVEMENTS.md)
- **GDPO v3 (NED penalty)**: GDPO v2 + ned_penalty_alpha=0.2 (penalizes NED for non-flipping completions), lr=5e-6 for SNLI / lr=1e-6 for BoolQ. SNLI training was unstable (KL spike/entropy collapse)
- **GDPO v4 (intermediate LR)**: GDPO v2 with lr=3e-6, beta=0.005 (5x stronger KL penalty), no NED penalty. Timed out at epoch ~0.97 with recurring KL spikes
- **GDPO v5 (paper-aligned)**: Critical fix: `generation_batch_size=128` (8 groups) instead of 16 (1 group). All prior GDPO experiments had batch normalization operating on only 1 group, making GDPO mathematically identical to GRPO. v5 also uses paper hyperparameters: lr=1e-6, beta=0.0005, epsilon_high=0.28 (DAPO asymmetric clipping). BS=8 on single A100-80GB
- **GRPO mcl768**: GRPO mv2 g16 with max_completion_length=768 for BoolQ (vs default 512)
- **SFT lr1e5/lr2e5**: SFT LR sweep with lr=1e-5 or 2e-5 (vs default 5e-6). All previous SFT runs used lr=5e-6; standard SFT with LoRA typically uses 1e-5 to 2e-5
- **GDPO v6**: GDPO v5 config (8 groups, conditioned rewards) with lr=5e-6 (matching GRPO). Tests whether higher LR helps with proper batch normalization
- **GDPO paper-match**: Maximum paper alignment. 4 GPUs (32 groups), TRL built-in `normalize_then_sum`, lr=5e-6, beta=0.0, standard (non-conditioned) rewards. Uses standard GRPOTrainer (not custom GDPOTrainer)
- **GRPO paper-match**: Control for GDPO paper-match. Same 4-GPU setup with `sum_then_normalize` (standard GRPO aggregation). Isolates the effect of per-reward normalization from larger batch size
- **GRPO mv2 sft-ws**: SFT warm-start GRPO: base model first fine-tuned via SFT on BoolQ CFs, LoRA merged, then used as initialization for GRPO mv2 g16 training. Result: no improvement over cold-start GRPO
- **SFT LR sweep results**: lr=1e-5 uniformly negative across datasets; lr=2e-5 causes heavy memorization (token accuracy 0.96, NaN PPL on BoolQ). Confirms lr=5e-6 is optimal for SFT with LoRA
- **GDPO v7 (4GPU + v6 innovations)**: 4 GPUs (32 groups), conditioned rewards, beta=0.0005, epsilon_high=0.28, lr=5e-6. Uses custom GDPOTrainer via torchrun. Result: +15.7% on SNLI-P — same as v6 (8 groups), confirming more groups don't improve ΔLFR
- **GDPO v6 checkpoint sweep (SNLI-P)**: Evaluations at ckpt-2000 (~0.25ep), ckpt-4000 (~0.50ep), ckpt-6000 (~0.75ep). Peak at ckpt-2000 (+18.2%), confirming GDPO peaks earlier than GRPO (0.25ep vs 0.50ep). Pattern: +18.2% → +18.1% → +16.5% → +15.7% (full epoch)
- **GDPO v6 checkpoint sweep (SNLI-H)**: Peak at ckpt-6000 (~0.75ep): **+18.7% ΔLFR**, surpassing GRPO mv2 g16 (+15.1%). Trajectory: +15.9% (0.25ep) → +17.6% (0.50ep) → +18.7% (0.75ep) → +17.6% (1.0ep). Peak later than SNLI-P
- **DPO BoolQ ml2048**: 2pair b4 2ep with max_length=2048: +8.5% ΔLFR. Below 1pair b4 (+10.5%), confirming 2pair is worse for BoolQ DPO
- **GDPO v6 BoolQ mcl1024**: max_completion_length=1024: +0.3% ΔLFR. Longer completions did not help GDPO on BoolQ
- **GRPO mv2 BoolQ mcl1024**: TIMEOUT after 24h (mcl1024 makes generation ~4x slower). Reached checkpoint-4400; eval pending
- **SFT BoolQ ml2048**: max_seq_length=2048 (vs default 1024). Modest improvement +1.8% (vs -0.0% at ml1024), suggesting training truncation was a minor factor for SFT on BoolQ
- **DPO BoolQ ml2048**: max_length=2048 (vs default 1024). Tests whether DPO also benefits from longer training sequences on BoolQ. Pending (fixed TRL API: removed deprecated max_prompt_length from DPOConfig)
- **KTO**: `train_dpo.py --loss_type kto` uses TRL's `KTOTrainer` with the Kahneman-Tversky utility loss (gains/losses weighted asymmetrically — losses loom larger than gains). Data is converted from the standard DPO pair format `(prompt, chosen, rejected)` into the binary `(prompt, completion, label: True/False)` format expected by `KTOTrainer` via `expand_to_kto_format`. Each DPO pair becomes 2 independent rows (1 desirable + 1 undesirable). This tests whether the KT asymmetric loss outperforms the symmetric DPO log-ratio loss on the same paired data; it does **not** test KTO's other advertised advantage (unpaired binary feedback, i.e., no `construct_dpo_pairs.py` pairing required).
- **DPO IPO**: `--loss_type ipo`. Identity Preference Optimization — uses squared error instead of sigmoid, bounded optimization that prevents overfitting. Beta maps to regularization parameter tau. All default hyperparameters (lr=5e-6, beta=0.1). Result: catastrophic on BoolQ (-11.7%), decent on SNLI-P (+14.9%), moderate on SNLI-H (+11.7%). Possible LR mismatch — IPO loss scale (~25) is fundamentally different from sigmoid DPO (~0.7)
- **DPO Robust**: `--loss_type robust --label_smoothing 0.01`. Noise-tolerant DPO that models annotation noise probability. All default hyperparameters. Result: worse than standard DPO everywhere (BoolQ +3.8%, SNLI-P +13.7%, SNLI-H +7.4%). Our synthetic data is clean, so noise tolerance isn't needed
- **DPO DiscoPOP**: `--loss_type discopop`. LLM-discovered loss function. All default hyperparameters. Result: near-zero learning (BoolQ +0.2%, SNLI-P -1.1%, SNLI-H +0.9%). Essentially failed to learn on our task
- **SimPO** (`train_dpo.py --loss_type simpo`, CPOTrainer): reference-free preference; `beta=2.0`, `simpo_gamma=1.4`, `cpo_alpha=0`, same b4 2ep / pair choice as best DPO per dataset. **BoolQ +9.8%** (second-best offline, 0.7pp below sigmoid DPO +10.5%; NED 0.111, PPL 14.0). **SNLI-P +27.1%** (within 0.4pp of GRPO mv2). **SNLI-H +26.4%** with high NED (0.477); **SNLI-H β×γ grid**: best **β=3, γ=0.5** (+29.7% ΔLFR, NED 0.428, lowest PPL in grid ~1202 vs very high PPL for other cells — SimPO is sensitive to β). See SNLI-H table for all six cells. **Fair-base SimPO b3γ0.5** (reuse GRPO mv2 g16 base CFs): BoolQ **+21.8%**, SNLI-P **+23.5%** (tables above). **SimPO b3γ0.5 + CPO α=0.1** (SNLI-H): +25.2% ΔLFR; aggregate PPL can be inflated by a few bad CFs — inspect `eval_full.json` / generations when the headline PPL looks pathological.
- **Perplexity in evaluation** (`evaluate_models.py`): A few degenerate edits (e.g. single character `"+"`) produced **NaN loss** under the quantized fine-tuned model; averaging those into mean PPL made the **entire** average NaN. **Fix**: skip non-finite loss in `compute_perplexity`, require ≥2 characters and ≥2 tokens, aggregate only **finite** per-CF PPLs; report `—` if none valid. Re-run eval to refresh markdown with the new aggregation
- **GDPO v6 SNLI-H ablation** (same v6 setup except one trick removed; full 1.0ep): removing **conditioned rewards** collapses ΔLFR to **-4.6%** (baseline +18.7%). Removing asymmetric clipping → +13.0%. Removing KL → raw ΔLFR +37.1% but PPL 1360 and NED 0.442 — KL mainly constrains disfluent / oversized edits, not “holding back” good performance at the baseline setting
- **DPO BoolQ β=0.05**: `beta=0.05` vs default 0.1, else same as 1pair b4 2ep. Result: **+5.8%** ΔLFR vs **+10.5%** at β=0.1; PPL **29.2** vs 23.9 — weaker KL hurts BoolQ DPO
- **DPO BoolQ IPO lr=1e-6**: same as IPO variant but `learning_rate=1e-6`. Result: **+0.3%** ΔLFR (vs **−11.7%** at lr=5e-6) — LR was the main issue for IPO on BoolQ, but tuned IPO still loses to sigmoid DPO

