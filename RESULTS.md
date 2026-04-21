# Evaluation Results

This file records counterfactual-generation **evaluation** runs. **Primary tables** (through [Training Configuration](#training-configuration)) are **Qwen/Qwen2.5-7B-Instruct** unless a row says otherwise. **[Qwen2.5-14B-Instruct](#qwen25-14b-instruct)** uses the same metrics and separate fair-base dirs; pipeline and parity notes are in [OVERVIEW.md](OVERVIEW.md).

**Metrics**:

- **LFR** (Label Flip Rate): % of counterfactuals where the base model's prediction changed from the original label (higher = better)
- **ΔLFR**: LFR improvement over the base model on the same evaluation set
- **NED** (Normalized Edit Distance): edit distance / max text length (lower = more minimal edits)
- **PPL** (Perplexity): fluency of generated text (lower = more fluent). **Aggregation**: mean over counterfactuals with **finite** per-CF PPL only (since 2026-04: `evaluate_models.py` skips non-finite loss and omits degenerate strings from the mean; legacy reports that showed `nan` had a few NaN CFs that poisoned the average — see Training Configuration).
- **LFR/NED**: label flip rate / normalized edit distance — edit efficiency (higher = more effective per unit of change). Reported in individual eval reports for N=200 runs.
- **Fair**: whether base model counterfactuals were reused across evaluations (`--base_eval_dir`) for consistent comparison
- **N**: number of validation samples evaluated (10 CFs generated per sample)
- **Parsed CFs only**: LFR, NED, and PPL in `eval_summary.json` are averaged over counterfactuals with a **non-null parsed** `edited_text` (`evaluate_models.py` `compute_metrics`). The tuned model generates its own CFs, so **`total_cfs` in `tuned_metrics` can be lower than in `base_metrics`** when the policy emits more unparseable outputs; this also appears in pre-v2fix GRPO fair runs. Fair evals still use the **same sampled entries** and, with `--base_eval_dir`, the **same stored base CFs** for the base side.
- **Parse rate** (hidden quality dimension): each entry gets 10 generation attempts; only those matching the structured format count. Parse rates vary: Base ~58–82%, DPO ~49–67%, SimPO ~24–70%, GRPO ~28–50% (dataset-dependent). **SimPO SNLI-H** is the extreme case: only 484/2000 (24%) parse, and 27/200 entries yield zero CFs. High ΔLFR methods with low parse rates are measured on a **self-selected subset** — see [OVERVIEW.md § CF coverage](OVERVIEW.md#cf-coverage-and-parse-rates) for details.

**Methods**: DPO (preference optimization), SimPO (reference-free preference via CPOTrainer), SFT (supervised fine-tuning on chosen CFs), GRPO (online RL with reward signal), GDPO (GRPO with per-reward normalization), KTO (Kahneman-Tversky Optimization — asymmetric loss via `KTOTrainer`; see Training Configuration)

**Configs**: `{pairs} {batch}` where pairs = number of preference pairs per entry, batch = effective batch size (b4 = bs1×ga4, b16 = bs4×ga4). GRPO configs: `single` = combined reward, `multi` = decomposed rewards, `g4`/`g16` = generations per prompt.

**GRPO reward versions**: GRPO g4 and g16 (non-v2) used an incorrect single reward: `flip + 0.8 * similarity` (no confidence). GRPO g16 v2 uses the corrected reward: `flip + confidence * similarity`, matching the DPO unified score. Multi-reward runs are unaffected (they use separate FlipReward + SimilarityReward). Results marked with † use the old reward.

**Duration**: `200step` = fixed 200 gradient steps (variable effective epochs), `0.5ep`–`3ep` = epoch-controlled training.

---

## Qwen2.5-7B-Instruct

Tables in **BoolQ**, **SNLI-Premise**, and **SNLI-Hypothesis** are for this base model (QLoRA adapters on `Qwen/Qwen2.5-7B-Instruct`).

### BoolQ


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| DPO    | 2pair b4  | 2ep      | 51.8% | +14.7% | 0.144 | 13.1 | no   | 200 |
| DPO    | 2pair b4  | 200step  | 57.6% | +13.9% | 0.130 | 10.2 | no   | 50  |
| DPO    | 1pair b4  | 2ep      | 48.0% | +10.5% | 0.110 | 23.9 | yes  | 200 |
| SimPO  | 1pair b4 simpo | 2ep | 46.7% | +9.8%  | 0.111 | 14.0 | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 | 2ep | 59.0% | +21.8% | 0.167 | 12.5 | yes  | 200 |
| DPO    | 1pair b4 beta005 | 2ep | 42.8% | +5.8%  | 0.103 | 29.2 | yes  | 200 |
| DPO    | 2pair b16 | 3ep      | 52.2% | +9.4%  | 0.122 | 11.2 | yes  | 100 |
| DPO    | 2pair b4  | 2ep      | 52.5% | +9.3%  | 0.138 | 12.0 | yes  | 100 |
| DPO    | 1pair b4  | 2ep      | 50.7% | +8.1%  | 0.116 | 10.9 | yes  | 100 |
| DPO    | 2pair b16 | 2ep      | 47.6% | +4.9%  | 0.121 | 11.1 | yes  | 100 |
| SFT    | 2pair b4  | 200step  | 49.2% | +4.1%  | 0.192 | 10.0 | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 48.0% | +3.1%  | 0.116 | 10.2 | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 47.7% | +2.7%  | 0.164 | 10.1 | no   | 100 |
| DPO    | 1pair b4  | 200step  | 52.2% | +2.5%  | 0.109 | 10.3 | no   | 50  |
| DPO    | 2pair b4  | 200step  | 47.4% | +2.4%  | 0.162 | 9.9  | yes  | 100 |
| DPO    | 1pair b16 | 3ep      | 45.4% | +2.3%  | 0.109 | 11.2 | yes  | 100 |
| SFT    | 1pair b4  | 200step  | 47.4% | +2.2%  | 0.194 | 10.1 | yes  | 100 |
| SFT    | 1pair b4  | 2ep      | 41.8% | -0.7%  | 0.194 | 12.9 | yes  | 100 |
| SFT    | 2pair b4  | 2ep      | 41.6% | -0.7%  | 0.222 | 17.7 | yes  | 100 |
| DPO    | 2pair b16 | 200step  | 40.5% | -2.8%  | 0.123 | 11.2 | yes  | 100 |
| SFT    | 1pair b4  | 1ep      | 39.6% | -3.6%  | 0.158 | 11.7 | yes  | 100 |
| SFT    | 2pair b4  | 1ep      | 38.3% | -3.7%  | 0.185 | 13.3 | yes  | 100 |
| SFT    | 1pair b16 | 2ep      | 38.2% | -4.5%  | 0.164 | 11.5 | yes  | 100 |
| SFT    | 2pair b16 | 200step  | 37.7% | -5.0%  | 0.152 | 11.4 | yes  | 100 |
| SFT    | 1pair b16 | 200step  | 37.3% | -5.6%  | 0.166 | 11.4 | yes  | 100 |
| SFT    | 1pair b16 | 1ep      | 37.0% | -5.6%  | 0.170 | 11.8 | yes  | 100 |
| SFT    | 2pair b16 | 2ep      | 35.5% | -6.5%  | 0.167 | 11.7 | yes  | 100 |
| DPO    | 1pair b16 | 2ep      | 36.1% | -6.7%  | 0.108 | 11.7 | yes  | 100 |
| SFT    | 2pair b16 | 1ep      | 35.6% | -7.0%  | 0.158 | 11.4 | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 35.4% | -7.8%  | 0.130 | 11.7 | yes  | 100 |
| SFT    | 1pair b4  | 0.5ep    | 33.6% | -9.1%  | 0.169 | 11.8 | yes  | 100 |
| GDPO   | mv2 g16   | ckpt200  | 37.7% | +1.1%  | 0.166 | 11.8 | yes  | 200 |
| GDPO   | mv2 g16   | ckpt200  | 36.3% | -6.1%  | 0.166 | 11.5 | yes  | 100 |
| GRPO   | mv2 g16   | ~0.25ep  | 37.8% | +1.2%  | 0.170 | 11.6 | yes  | 200 |
| GRPO   | mv2 g16 sft-ws| ~0.25ep | 37.8% | +1.2%  | 0.170 | 11.9 | yes  | 200 |
| GRPO   | mv2 g16 lr1e6 | ~0.25ep | 37.4% | +1.1% | 0.155 | 11.8 | yes  | 200 |
| GRPO   | v2 g16    | ~0.11ep  | 37.7% | +0.4%  | 0.151 | 11.7 | yes  | 200 |
| GDPO   | mv2 g16 lr1e6 | 1.0ep | 36.8% | +0.2%  | 0.153 | 11.6 | yes  | 200 |
| GRPO   | v2 g24 lr1e6 | 1.0ep | 36.7% | +0.2%  | 0.161 | 11.6 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 37.2% | -0.0%  | 0.160 | 11.5 | yes  | 200 |
| SFT    | 2pair b4 ml2048| 200step | 38.4% | +1.8%  | 0.162 | 11.6 | yes  | 200 |
| SFT    | 2pair b4 lr1e5| 200step | 40.2% | -2.0%  | 0.175 | 12.0 | yes  | 100 |
| SFT    | 2pair b4 lr2e5| 200step | 44.0% | +1.9%  | 0.272 | —    | yes  | 100 |
| DPO    | 2pair b4 ml2048| 2ep      | 45.4% | +8.5%  | 0.123 | 26.9 | yes  | 200 |
| DPO    | 1pair b4 ml2048| 2ep      | 40.5% | +3.8%  | 0.114 | 540.4| yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep | 8.7% | **-28.3%** | 0.214 | 12.0 | yes  | 200 |
| GRPO   | mv2 g16 mcl768 | ~0.25ep | 38.2% | +1.7% | 0.159 | 12.0 | yes  | 200 |
| GRPO-fair| mv2 g16 fair | 1.0ep   | 36.5% | +0.1%  | 0.142 | —    | yes  | 200 |
| GDPO v6| mv2 g16 mcl1024| 1.0ep  | 36.7% | +0.3%  | 0.137 | 11.8 | yes  | 200 |
| GDPO v2| mv2 g16 cond  | 1.0ep | 36.8% | +0.4%  | 0.166 | 11.8 | yes  | 200 |
| GDPO   | mv2 g24 lr1e6 | 1.0ep | 36.0% | -0.6%  | 0.162 | 11.5 | yes  | 200 |
| GRPO   | mv2 g24 lr1e6 | 1.0ep | 36.5% | -0.6%  | 0.158 | 11.5 | yes  | 200 |
| GDPO v3| mv2 g16 ned02 | 1.0ep | 35.3% | -1.1%  | 0.153 | 11.5 | yes  | 200 |
| GRPO   | mv2 g16 lr1e6 | ~0.25ep | 36.2% | -6.5% | 0.153 | 11.3 | yes  | 100 |
| GRPO   | mv2 g16   | ~0.25ep  | 35.4% | -7.1%  | 0.161 | 11.4 | yes  | 100 |
| GRPO   | v2 g16    | ~0.11ep  | 34.9% | -8.3%  | 0.155 | 11.8 | yes  | 100 |
| GRPO   | v2 g16 lr1e6 | ~0.11ep | 33.9% | -8.9% | 0.153 | 11.7 | yes  | 100 |
| GRPO   | multi g16 | ~0.11ep  | 33.9% | -9.3%  | 0.163 | 11.5 | yes  | 100 |
| GRPO † | single g4 | 2ep      | 33.0% | -10.4% | 0.126 | 11.0 | yes  | 100 |
| GRPO   | multi g4  | 2ep      | 33.1% | -10.2% | 0.127 | 43.3 | yes  | 100 |
| GRPO   | multi g16 | ~0.25ep  | 18.5% | -24.7% | 0.085 | 11.7 | yes  | 100 |
| GRPO   | multi g16 | ~0.85ep  | 11.0% | -32.9% | 0.295 | 9.0  | yes  | 100 |
| GRPO   | v2 g16    | ~0.75ep  | 10.3% | -33.6% | 0.261 | 10.0 | yes  | 100 |
| GRPO   | v2 g16    | ~0.50ep  |  8.4% | -35.4% | 0.241 | 10.6 | yes  | 100 |
| GRPO   | multi g16 | ~0.50ep  |  8.3% | -35.5% | 0.168 | 14.1 | yes  | 100 |
| GRPO   | v2 g16    | ~0.25ep  |  7.2% | -36.5% | 0.088 | 21.4 | yes  | 100 |
| GRPO   | mv2 g16 mcl1024| ckpt2000 | 7.6% | -29.9% | 0.084 | 17.5 | yes  | 200 |
| DPO    | 1pair b4 ipo   | 2ep      | 25.1% | -11.7% | 0.076 | 13.2 | yes  | 200 |
| DPO    | 1pair b4 ipo lr1e6 | 2ep  | 36.6% | +0.3%  | 0.160 | 11.8 | yes  | 200 |
| DPO    | 1pair b4 robust| 2ep      | 40.7% | +3.8%  | 0.115 | 11.3 | yes  | 200 |
| DPO    | 1pair b4 disco | 2ep      | 36.5% | +0.2%  | 0.168 | 11.6 | yes  | 200 |
| KTO    | 1pair b4 kto  | 2ep      | training (**2811932**) | | | | | |

**GRPO mv2 g16 v2fix** (BoolQ, fair, `evaluation_boolq_200s_grpo_mv2_g16_v2fix/`, job **2782111**): **−28.3%** ΔLFR vs **~+1%** for shorter mv2 g16 runs — likely **policy collapse** after full 1.0ep; inspect `eval_full.json` / generations before comparing to other GRPO rows.

### SNLI-Premise


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | mv2 g16 v2fix | 1.0ep | 80.9% | +29.1% | 0.336 | 92.9  | yes  | 200 |
| SimPO  | 2pair b4 simpo v2fix | 2ep | 83.4% | +30.5% | 0.346 | 157.6 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 79.7% | +27.5% | 0.348 | 72.7  | yes  | 200 |
| SimPO  | 2pair b4 simpo | 2ep | 79.9% | +27.1% | 0.358 | 154.7 | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 | 2ep | 76.1% | +23.5% | 0.362 | 131.2 | yes  | 200 |
| GRPO   | paper 4GPU| 1.0ep    | 78.3% | +25.9% | 0.348 | 103.2 | yes  | 200 |
| GRPO   | v2 g16    | ~0.50ep  | 78.0% | +25.8% | 0.345 | 110.6 | yes  | 200 |
| GRPO   | mv2 g24   | 1.0ep    | 74.2% | +22.2% | 0.387 | 146.6 | yes  | 200 |
| DPO    | 2pair b4  | 2ep      | 75.0% | +22.0% | 0.327 | 142.3 | no   | 200 |
| DPO    | 2pair b4  | 2ep      | 71.5% | +17.3% | 0.326 | —     | yes  | 100 |
| DPO    | 2pair b16 | 2ep      | 70.5% | +17.2% | 0.321 | 83.6  | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 68.1% | +16.6% | 0.296 | 89.8  | yes  | 100 |
| DPO    | 2pair b16 | 3ep      | 69.9% | +16.5% | 0.323 | 79.6  | yes  | 100 |
| DPO    | 1pair b4  | 2ep      | 68.0% | +15.7% | 0.301 | 133.3 | yes  | 200 |
| DPO    | 2pair b4 v2fix | 2ep | 66.1% | +14.3% | 0.296 | 118.0 | yes  | 200 |
| DPO    | 2pair b4  | 200step  | 64.2% | +12.7% | 0.275 | 93.8  | yes  | 100 |
| DPO    | 1pair b4  | 2ep      | 65.7% | +12.2% | 0.310 | 94.8  | yes  | 100 |
| DPO    | 2pair b16 | 200step  | 65.5% | +12.2% | 0.311 | 90.1  | yes  | 100 |
| DPO    | 1pair b16 | 2ep      | 63.8% | +10.5% | 0.303 | 85.9  | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 61.2% | +9.7%  | 0.276 | 94.8  | no   | 100 |
| DPO    | 1pair b16 | 3ep      | 63.0% | +9.3%  | 0.297 | 94.1  | yes  | 100 |
| GRPO   | v2 g24    | 1.0ep    | 61.1% | +8.6%  | 0.307 | 769.8 | yes  | 200 |
| DPO    | 1pair b4  | 200step  | 60.2% | +7.9%  | 0.257 | 75.0  | no   | 50  |
| GDPO   | mv2 g16   | 1.0ep    | 54.2% | +2.5%  | 0.222 | 157.4 | yes  | 200 |
| GDPO   | mv2 g24 lr1e6 | 1.0ep | 53.1% | +1.1%  | 0.245 | 131.8 | yes  | 200 |
| GDPO   | mv2 g16 lr1e6 | 1.0ep | 51.4% | -0.5%  | 0.244 | 134.5 | yes  | 200 |
| GDPO v2| mv2 g16 cond  | ~0.96ep | 53.2% | +0.8%  | 0.249 | 162.8 | yes  | 200 |
| GDPO v3| mv2 g16 lr5e6 | ~0.92ep | 60.5% | +8.3%  | 0.265 | 120.1 | yes  | 200 |
| GDPO v4| mv2 g16 lr3e6 | ~0.97ep | 56.1% | +4.0%  | 0.254 | 140.9 | yes  | 200 |
| GDPO v5| mv2 g16 8grp  | 1.0ep  | 53.6% | +1.7%  | 0.252 | 136.0 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.25ep| 70.4% | +18.2% | 0.303 | 145.4 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.50ep| 70.5% | +18.1% | 0.297 | 116.1 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.75ep| 68.9% | +16.5% | 0.290 | 121.6 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | 1.0ep  | 67.9% | +15.7% | 0.287 | 135.4 | yes  | 200 |
| GDPO v6| mv2 g16 v2fix | 1.0ep  | 65.2% | +13.0% | 0.285 | 128.4 | yes  | 200 |
| GDPO v7| 4GPU lr5e6    | 1.0ep  | 67.9% | +15.7% | 0.285 | 117.3 | yes  | 200 |
| GDPO   | paper 4GPU    | 1.0ep  | 59.4% | +7.6%  | 0.258 | 121.6 | yes  | 200 |
| GRPO-fair| mv2 g16 fair | ~0.99ep| 64.6% | +12.3% | 0.281 | 142.3 | yes  | 200 |
| DPO    | 2pair b4 ipo   | 2ep      | 67.1% | +14.9% | 0.315 | 143.7 | yes  | 200 |
| DPO    | 2pair b4 robust| 2ep      | 65.8% | +13.7% | 0.302 | 137.6 | yes  | 200 |
| DPO    | 2pair b4 disco | 2ep      | 51.1% | -1.1%  | 0.248 | 133.8 | yes  | 200 |
| KTO    | 1pair b4 kto  | 2ep      | training (**2811933**) | | | | | |
| SFT    | 1pair b4  | 200step  | 50.5% | -1.7%  | 0.246 | 144.4 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 82.7% | +29.0% | 0.356 | 69.7  | yes  | 100 |
| GRPO   | v2 g16    | ~0.50ep  | 82.3% | +28.7% | 0.347 | 89.1  | yes  | 100 |
| GRPO   | v2 g16    | ~0.25ep  | 80.7% | +27.2% | 0.359 | 70.7  | yes  | 100 |
| GRPO   | v2 g16    | 1.0ep    | 77.6% | +24.4% | 0.341 | 94.7  | yes  | 100 |
| GRPO   | v2 g16    | ~0.75ep  | 74.8% | +21.3% | 0.339 | 95.9  | yes  | 100 |
| GRPO   | multi g16 | ~0.50ep  | 65.6% | +12.4% | 0.303 | 102.7 | yes  | 100 |
| GRPO   | multi g16 | ~0.25ep  | 65.5% | +12.0% | 0.299 | 84.5  | yes  | 100 |
| GRPO   | multi g16 | ~0.62ep  | 61.2% | +8.3%  | 0.276 | 151.1 | yes  | 100 |
| GRPO   | multi g4  | 2ep      | 60.2% | +7.2%  | 0.275 | 85.1  | yes  | 100 |
| GDPO   | mv2 g16   | 1.0ep    | 60.6% | +7.2%  | 0.228 | 146.9 | yes  | 100 |
| GDPO   | mv2 g16   | 1.0ep    | 54.2% | +2.5%  | 0.222 | 157.4 | yes  | 200 |
| GRPO   | multi g16 | ~0.88ep  | 59.3% | +6.1%  | 0.838 | 18738 | yes  | 100 |
| GRPO † | single g4 | 2ep      | 58.7% | +5.4%  | 0.266 | 92.4  | yes  | 100 |
| GRPO   | v2 g16    | ~0.06ep  | 52.9% | -1.0%  | 0.256 | 111.0 | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 56.4% | +4.6%  | 0.244 | 69.8  | no   | 50  |
| DPO    | 1pair b4  | 200step  | 57.6% | +4.5%  | 0.274 | 96.5  | yes  | 100 |
| SFT    | 1pair b4  | 200step  | 54.5% | +2.7%  | 0.248 | 109.9 | yes  | 100 |
| SFT    | 2pair b16 | 1ep      | 55.2% | +1.8%  | 0.249 | 105.9 | yes  | 100 |
| SFT    | 2pair b4  | 200step  | 53.3% | +1.5%  | 0.243 | 110.6 | yes  | 100 |
| SFT    | 2pair b4 lr1e5| 200step | 51.2% | -2.5%  | 0.253 | 374.1 | yes  | 100 |
| SFT    | 2pair b4 lr2e5| 200step | 48.8% | -2.9%  | 0.298 | 201.5 | yes  | 100 |
| SFT    | 2pair b4  | 0.5ep    | 52.8% | +0.5%  | 0.262 | 126.8 | yes  | 100 |
| SFT    | 2pair b16 | 0.5ep    | 54.0% | +0.4%  | 0.256 | 152.2 | yes  | 100 |
| SFT    | 2pair b4  | 2ep      | 51.9% | -0.0%  | 0.272 | 115.2 | yes  | 100 |
| SFT    | 2pair b16 | 200step  | 52.4% | -0.6%  | 0.252 | 125.4 | yes  | 100 |
| SFT    | 1pair b4  | 0.5ep    | 52.8% | -0.7%  | 0.248 | 111.1 | yes  | 100 |
| SFT    | 1pair b16 | 2ep      | 52.6% | -0.7%  | 0.253 | 122.6 | yes  | 100 |
| SFT    | 1pair b16 | 0.5ep    | 52.4% | -0.7%  | 0.251 | 117.4 | yes  | 100 |
| SFT    | 1pair b16 | 200step  | 52.5% | -0.9%  | 0.249 | 107.3 | yes  | 100 |
| SFT    | 2pair b16 | 2ep      | 50.3% | -2.2%  | 0.247 | 106.9 | yes  | 100 |
| SFT    | 1pair b16 | 1ep      | 50.7% | -2.9%  | 0.250 | 104.4 | yes  | 100 |
| SFT    | 1pair b4  | 1ep      | 49.3% | -3.4%  | 0.261 | 137.0 | yes  | 100 |
| SFT    | 1pair b4  | 2ep      | 48.4% | -4.0%  | 0.261 | 188.1 | yes  | 100 |
| SFT    | 2pair b4  | 1ep      | 47.4% | -4.7%  | 0.265 | 115.1 | yes  | 100 |
| SFT    | 2pair b16 | 1ep      | 53.2% | +0.9%  | 0.250 | 130.5 | yes  | 200 |
| SFT    | 2pair b4  | 200step  | 53.1% | +0.9%  | 0.249 | 131.4 | yes  | 200 |
| SFT    | 2pair b16 | 0.5ep    | 49.9% | -2.2%  | 0.237 | 136.7 | yes  | 200 |
| SFT    | 2pair b4  | 0.5ep    | 49.8% | -2.0%  | 0.253 | 145.6 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 48.9% | -2.7%  | 0.266 | 355.3 | yes  | 200 |


### SNLI-Hypothesis


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | v2 g24    | 1.0ep    | 67.4% | +22.6% | 0.302 | 9651  | yes  | 200 |
| SimPO  | 2pair b4 simpo b3g05 | 2ep | 74.2% | +29.7% | 0.428 | 1202  | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 v2fix | 2ep | 73.6% | +29.4% | 0.390 | 617.1 | yes  | 200 |
| SimPO  | 2pair b4 simpo b2g05 | 2ep | 71.2% | +26.8% | 0.459 | 30329 | yes  | 200 |
| SimPO  | 2pair b4 simpo | 2ep (β2 γ1.4) | 70.9% | +26.4% | 0.477 | 15519 | yes  | 200 |
| SimPO  | 2pair b4 simpo b3g14 | 2ep | 70.1% | +25.2% | 0.464 | 1606  | yes  | 200 |
| SimPO  | 2pair b4 simpo b1g05 | 2ep | 68.5% | +24.5% | 0.505 | 16403 | yes  | 200 |
| SimPO  | 2pair b4 simpo b2g14 | 2ep | 68.8% | +24.3% | 0.476 | 23282 | yes  | 200 |
| SimPO  | 2pair b4 simpo b1g14 | 2ep | 67.4% | +23.0% | 0.507 | 17970 | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 CPO α=0.1 | 2ep | 69.5% | +25.2% | 0.470 | 6427.5 | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b3γ0.65 CPO α=0.1 | 2ep | 76.8% | +32.1% | 0.447 | 4424  | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b2.5γ0.5 CPO α=0.1 | 2ep | 73.9% | +29.4% | 0.467 | 7894  | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b2.5γ0.5 CPO α=0.015 | 2ep | 73.1% | +28.8% | 0.449 | 9009  | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b3γ0.8 CPO α=0.005 | 2ep | 72.5% | +28.1% | 0.467 | 1593  | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b3γ0.65 | 2ep | 70.7% | +26.2% | 0.439 | 1475  | yes  | 200 |
| SimPO  | 2pair b4 simpo_qual b3γ0.5 CPO α=0.015 | 2ep | 68.2% | +23.7% | 0.442 | 38417 | yes  | 200 |
| DPO    | 2pair b4  | 2ep      | 65.9% | +21.0% | 0.378 | 296.5 | yes  | 200 |
| DPO    | 2pair b4 v2fix | 2ep | 49.5% | +5.4%  | 0.315 | 285.7 | yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep | 67.0% | +22.6% | 0.358 | 508.3 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 59.4% | +15.1% | 0.266 | 702.9 | yes  | 200 |
| GRPO   | mv2 g24   | 1.0ep    | 57.4% | +12.6% | 0.253 | 453.2 | yes  | 200 |
| GRPO   | v2 g16    | 1.0ep    | 55.7% | +11.0% | 0.269 | 317.3 | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 53.4% | +8.7%  | 0.334 | 368.0 | no   | 200 |
| GRPO-fair| mv2 g16 fair | 1.0ep  | 51.1% | +6.8%  | 0.316 | 285.1 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 46.1% | +1.7%  | 0.329 | 286.8 | yes  | 200 |
| GDPO   | mv2 g24 lr1e6 | 1.0ep | 45.6% | +1.1%  | 0.302 | 247.6 | yes  | 200 |
| GDPO v2| mv2 g16 cond  | 1.0ep | 45.0% | +0.7%  | 0.316 | 329.0 | yes  | 200 |
| GDPO v3| mv2 g16 ned02 | ~0.95ep | 46.3% | +1.8% | 0.276 | 308.3 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.25ep| 60.3% | +15.9% | 0.365 | 248.9 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.50ep| 62.0% | +17.6% | 0.363 | 329.6 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.75ep| 63.1% | +18.7% | 0.361 | 334.9 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | 1.0ep  | 61.8% | +17.6% | 0.356 | 281.1 | yes  | 200 |
| GDPO v6| mv2 g16 v2fix | 1.0ep | 57.0% | +12.1% | 0.365 | 293.4 | yes  | 200 |
| GDPO v6 abl | no cond | 1.0ep | 39.9% | -4.6%  | 0.216 | 345.9 | yes  | 200 |
| GDPO v6 abl | no clip | 1.0ep | 57.1% | +13.0% | 0.336 | 336.4 | yes  | 200 |
| GDPO v6 abl | no KL   | 1.0ep | 81.4% | +37.1% | 0.442 | 1360.0| yes  | 200 |
| GDPO   | mv2 g16 lr1e6 | 1.0ep | 44.4% | +0.1%  | 0.307 | 283.8 | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 53.4% | +8.7%  | 0.334 | 368.0 | no   | 200 |
| DPO    | 2pair b4  | 200step  | 44.3% | +8.4%  | 0.276 | 164.1 | no   | 50  |
| DPO    | 1pair b4  | 2ep      | 54.3% | +7.5%  | 0.339 | 390.6 | yes  | 100 |
| DPO    | 2pair b16 | 200step  | 50.3% | +3.6%  | 0.355 | 229.1 | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 48.9% | +3.0%  | 0.321 | 196.3 | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 49.2% | +2.9%  | 0.307 | 225.2 | yes  | 100 |
| DPO    | 1pair b16 | 3ep      | 49.1% | +2.4%  | 0.319 | 253.9 | yes  | 100 |
| SFT    | 2pair b4  | 2ep      | 48.6% | +1.8%  | 0.355 | 253.1 | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 47.1% | +1.2%  | 0.327 | 238.6 | no   | 100 |
| DPO    | 1pair b4  | 200step  | 39.2% | +0.2%  | 0.254 | 154.5 | no   | 50  |
| DPO    | 1pair b4  | 200step  | 46.8% | +0.0%  | 0.334 | 274.5 | yes  | 100 |
| SFT    | 1pair b4  | 200step  | 45.3% | -0.8%  | 0.307 | 302.1 | yes  | 100 |
| DPO    | 1pair b16 | 2ep      | 45.3% | -1.9%  | 0.325 | 275.7 | yes  | 100 |
| SFT    | 2pair b4  | 0.5ep    | 45.0% | -2.0%  | 0.335 | 218.5 | yes  | 100 |
| SFT    | 1pair b16 | 1ep      | 43.8% | -2.6%  | 0.329 | 209.6 | yes  | 100 |
| SFT    | 1pair b16 | 0.5ep    | 43.7% | -2.8%  | 0.331 | 225.0 | yes  | 100 |
| SFT    | 2pair b16 | 2ep      | 44.0% | -2.8%  | 0.335 | 205.6 | yes  | 100 |
| SFT    | 2pair b4  | 200step  | 43.0% | -3.3%  | 0.315 | 206.4 | yes  | 100 |
| SFT    | 2pair b4 lr1e5| 200step | 45.9% | -0.9%  | 0.351 | 246.3 | yes  | 100 |
| SFT    | 2pair b4 lr2e5| 200step | 47.8% | +1.3%  | 0.371 | —    | yes  | 100 |
| SFT    | 2pair b16 | 1ep      | 43.9% | -3.5%  | 0.330 | 208.3 | yes  | 100 |
| SFT    | 2pair b4  | 1ep      | 43.3% | -3.5%  | 0.347 | 272.5 | yes  | 100 |
| SFT    | 1pair b4  | 2ep      | 42.5% | -4.2%  | 0.346 | 233.3 | yes  | 100 |
| SFT    | 1pair b16 | 200step  | 42.4% | -4.6%  | 0.325 | 254.0 | yes  | 100 |
| SFT    | 2pair b16 | 200step  | 42.3% | -5.1%  | 0.329 | 194.3 | yes  | 100 |
| SFT    | 1pair b16 | 2ep      | 41.9% | -5.1%  | 0.332 | 218.5 | yes  | 100 |
| SFT    | 2pair b16 | 0.5ep    | 41.9% | -5.1%  | 0.326 | 385.0 | yes  | 100 |
| SFT    | 1pair b4  | 0.5ep    | 41.7% | -5.3%  | 0.328 | 271.4 | yes  | 100 |
| SFT    | 1pair b4  | 1ep      | 40.7% | -6.1%  | 0.343 | 211.0 | yes  | 100 |
| GRPO   | mv2 g16   | 1.0ep    | 58.4% | +11.5% | 0.285 | 455.6 | yes  | 100 |
| GRPO   | v2 g16    | 1.0ep    | 54.3% | +7.3%  | 0.285 | 374.4 | yes  | 100 |
| GRPO   | v2 g16    | ~0.50ep  | 53.5% | +6.3%  | 0.276 | 370.6 | yes  | 100 |
| GRPO   | v2 g16    | ~0.75ep  | 52.9% | +6.2%  | 0.276 | 450.8 | yes  | 100 |
| GRPO   | v2 g16    | ~0.25ep  | 49.7% | +2.8%  | 0.290 | 319.4 | yes  | 100 |
| GRPO   | v2 g16    | ~0.06ep  | 44.2% | -2.3%  | 0.327 | 224.3 | yes  | 100 |
| GRPO † | single g4 | ckpt400 (~0.1ep) | 43.7% | -3.3% | 0.337 | — | yes | 100 |
| GRPO † | single g4 | ckpt1000 (~0.25ep) | 39.9% | -7.5% | 0.265 | — | yes | 100 |
| GRPO † | single g4 | ckpt2000 (~0.5ep) | 39.7% | -7.1% | 0.222 | — | yes | 100 |
| GRPO   | multi g16 | ~0.50ep  | 38.5% | -8.8%  | 0.182 | 233.5 | yes  | 100 |
| GRPO † | single g4 | 2ep      | 36.3% | -10.6% | 0.167 | 227.9 | yes  | 100 |
| GRPO   | multi g4  | 2ep      | 35.6% | -11.3% | 0.151 | 359.5 | yes  | 100 |
| GRPO   | multi g16 | ~0.25ep  | 35.7% | -11.5% | 0.174 | 314.4 | yes  | 100 |
| GRPO   | multi g16 | ~0.75ep  | 35.4% | -11.8% | 0.095 | 251.8 | yes  | 100 |
| GRPO   | multi g16 | 1.0ep    | 30.9% | -16.5% | 0.091 | 300.2 | yes  | 100 |
| GRPO   | multi g16 | ~0.62ep  | 28.6% | -18.2% | 0.089 | 363.8 | yes  | 100 |
| GDPO   | mv2 g16   | 1.0ep    | 26.5% | -20.0% | 0.083 | 379.7 | yes  | 100 |
| GDPO   | mv2 g16   | 1.0ep    | 22.4% | -22.0% | 0.075 | 349.0 | yes  | 200 |
| DPO    | 2pair b4 ipo   | 2ep      | 56.0% | +11.7% | 0.356 | 469.0 | yes  | 200 |
| DPO    | 2pair b4 robust| 2ep      | 51.8% | +7.4%  | 0.332 | 277.8 | yes  | 200 |
| DPO    | 2pair b4 disco | 2ep      | 45.5% | +0.9%  | 0.318 | 266.2 | yes  | 200 |
| KTO    | 1pair b4 kto  | 2ep      | training (**2811934**) | | | | | |


---

## Qwen2.5-14B-Instruct

All rows below use **Qwen/Qwen2.5-14B-Instruct** as generator and **judge** (`evaluate_models.py --base_model`). **Fair** = base CFs from the same 14B base model on the same N=200 seed=42 validation slice. The first eval per dataset established the **anchor dir**; all subsequent 14B evals reuse it via `--base_eval_dir` (`run_eval_14b_fair.sh`). Anchors:
- **BoolQ**: `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/`
- **SNLI-P**: `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/`
- **SNLI-H**: `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_14b_v2fix/`

### BoolQ (14B)

| Method | Config    | Duration | LFR   | ΔLFR | NED   | PPL  | Fair | N   |
| ------ | --------- | -------- | ----- | ---- | ----- | ---- | ---- | --- |
| SimPO  | 2pair b4 β3γ0.5 | 2ep | 62.5% | +7.0% | 0.248 | 9.1  | yes  | 200 |
| GDPO v6| g16 bs4   | 1.0ep    | 60.8% | +6.0% | 0.186 | 8.3  | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 56.7% | +1.7% | 0.264 | 8.5  | yes  | 200 |
| SFT    | 2pair b4 ml2048 | 200step | 55.7% | +0.8% | 0.298 | 8.5  | yes  | 200 |
| GDPO v6| g16 bs8   | ckpt-200 | 55.3% | +0.5% | 0.292 | 8.8  | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 55.2% | +0.3% | 0.268 | 8.4  | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep | training (**2819415**) | | | | | |

SimPO: `dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_boolq_200s_simpo_b3g05_qwen25_14b/` (train **2777928**, eval **2780617**).
GDPO v6 full epoch: `gdpo_model_boolq_1ep_g16_qwen25_14b_v6` (resumed to ckpt-2000, bs=4) → `evaluation_boolq_200s_gdpo_v6_qwen25_14b/` (eval **2805602**). NED 0.186 is notably lower than the ckpt-200 partial (0.292), suggesting the full training teaches more minimal edits.
GRPO: `grpo_model_boolq_1ep_g16_multi_qwen25_14b_boolq_mv2` → `evaluation_boolq_200s_grpo_mv2_g16_qwen25_14b/` (train **2777920**, eval **2794877**).
SFT: `sft_model_boolq_2pair_b4_ml2048_200step_qwen25_14b` → `evaluation_boolq_200s_sft_2pair_b4_ml2048_200step_qwen25_14b/` (train **2780681**, eval **2781361**).
GDPO v6 ckpt-200: partial result (OOM at ~0.22ep); superseded by full-epoch eval above.
DPO: `dpo_model_boolq_2000e40c_1pair_qwen25_14b` → `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/` (eval **2777870**).

### SNLI-Premise (14B)

| Method | Config        | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------ | ------------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| GDPO v6| g16 bs4       | 1.0ep    | 89.6% | +26.2% | 0.361 | 79.2 | yes  | 200 |
| SimPO  | 2pair b4 β3γ0.5 | 2ep    | 84.0% | +21.5% | 0.421 | 92.0 | yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep    | 78.0% | +15.0% | 0.311 | 82.6 | yes  | 200 |
| DPO    | 2pair b4      | 2ep      | 69.4% | +6.6%  | 0.318 | 83.1 | yes  | 200 |
| GDPO v6| g16 bs8   | ckpt-500 | 69.7% | +6.6%  | 0.316 | 93.1 | yes  | 200 |
| SFT    | 2pair b16     | 1ep      | 63.1% | -0.0%  | 0.321 | 98.9 | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep    | training (**2819416**) | | | | | |

GDPO v6 full epoch: `gdpo_model_snli_premise_1ep_g16_qwen25_14b_v6` (resumed to ckpt-4000, bs=4) → `evaluation_snli_premise_200s_gdpo_v6_qwen25_14b/` (eval **2805603**). Note: total_cfs 607 vs base 1288 (47% parse rate) — high self-selection; model only outputs CFs when highly confident they flip. Best 14B result.
SimPO: `dpo_model_snli_premise_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_snli_premise_200s_simpo_b3g05_qwen25_14b/` (train **2777926**, eval **2781365**).
GRPO: `grpo_model_snli_premise_1ep_g16_multi_qwen25_14b_v2fix` → `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/` (train **2773359**, eval **2777871**).
DPO: `dpo_model_snli_premise_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_premise_200s_dpo_2pair_b4_2ep_qwen25_14b/` (train **2777924**, eval **2781363**).
GDPO v6 ckpt-500: partial; superseded by full-epoch eval above.
SFT: `sft_model_snli_premise_2pair_b16_1ep_qwen25_14b` → `evaluation_snli_premise_200s_sft_2pair_b16_1ep_qwen25_14b/` (train **2780682**, eval **2781362**).

### SNLI-Hypothesis (14B)

| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| SimPO  | 2pair b4 β3γ0.5 | 2ep | 89.9% | +13.7% | 0.426 | 352.4 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 85.5% | +9.2%  | 0.454 | 242.6 | yes  | 200 |
| GDPO   | v6 g16 v2fix | 1.0ep | 85.0% | +8.8%  | 0.439 | 241.9 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 80.2% | +3.9%  | 0.482 | 186.9 | yes  | 200 |
| DPO    | 2pair b4  | 2ep      | 76.7% | +0.7%  | 0.467 | 167.3 | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep | training (**2819417**) | | | | | |

SimPO: `dpo_model_snli_hypothesis_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_snli_hypothesis_200s_simpo_b3g05_qwen25_14b/` (train **2777927**, eval **2781366**).
GRPO: `grpo_model_snli_hypothesis_1ep_g16_multi_qwen25_14b_snli_h_mv2` → `evaluation_snli_hypothesis_200s_grpo_mv2_g16_qwen25_14b/` (train **2777921**, eval **2794878**).
GDPO: `gdpo_model_snli_hypothesis_1ep_g16_qwen25_14b_v6_v2fix` → `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_14b_v2fix/` (eval **2777872**).
SFT: `sft_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_hypothesis_200s_sft_2pair_qwen25_14b/` (train **2786445**, eval **2794799**).
DPO: `dpo_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_hypothesis_200s_dpo_2pair_b4_2ep_qwen25_14b/` (train **2777925**, eval **2781364**).
**14B matrix complete** (all 5 methods × 3 datasets evaluated). GDPO BoolQ and SNLI-P partial-checkpoint results will be superseded by full-epoch evals (**2805602**, **2805603**).

---

## Qwen2.5-3B-Instruct

All rows below use **Qwen/Qwen2.5-3B-Instruct** as generator and judge. Same pipeline as 14B: QLoRA 4-bit, RTXA6000/RTXA6000-SLT, fair-base anchors via `run_eval_3b_fair.sh`. The first eval per dataset (DPO) establishes the anchor; subsequent evals reuse it via `--base_eval_dir`. Anchors: BoolQ → `evaluation_boolq_200s_dpo_1pair_qwen25_3b/`, SNLI-P → `evaluation_snli_premise_200s_dpo_1pair_qwen25_3b/`, SNLI-H → `evaluation_snli_hypothesis_200s_dpo_1pair_qwen25_3b/`.

### BoolQ (3B)

Base LFR 30.5% (notably lower than 7B 37.2% / 14B 54.9% — 3B struggles with boolean reasoning). Anchor established via DPO eval.

| Method  | Config          | Duration | LFR   | ΔLFR  | NED   | PPL  | Fair | N   |
| ------- | --------------- | -------- | ----- | ----- | ----- | ---- | ---- | --- |
| DPO     | 1pair b4        | 200step  | 30.5% | +0.0% | 0.211 | 13.1 | yes  | 200 |
| SimPO   | 2pair b4 β3γ0.5 | 2ep      | pending (**2819352**) | | | | | |
| SFT     | 2pair 200step   |          | pending (**2819353**) | | | | | |
| GRPO    | mv2 g16         | 1ep      | pending (**2819354**) | | | | | |
| GRPO    | v2 g16 (single) | 1ep      | training (**2819412**) | | | | | |
| GDPO v6 | g16 bs8 ckpt-100| ~1.6ep   | pending (**2819409**) | | | | | |
| GDPO v6 | g16 bs8 5ep     | 5ep      | retraining (**2819410**) | | | | | |

DPO +0.0% is striking — the 3B model cannot learn meaningful BoolQ counterfactuals from DPO pairs alone. BoolQ requires boolean reasoning beyond 3B capacity. Results for other methods pending anchor eval completion.

### SNLI-Premise (3B)

Base LFR ~32.6% (much lower than 7B 52% / 14B 63% — 3B struggles with NLI reasoning). Anchor: `evaluation_snli_premise_200s_dpo_1pair_qwen25_3b/`.

| Method  | Config          | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------- | --------------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| SimPO   | 2pair b4 β3γ0.5 | 2ep      | 59.9% | +25.5% | 0.309 | 95.1  | yes  | 200 |
| GDPO v6 | g16 bs8         | 1ep      | 50.6% | +17.5% | 0.269 | 76.3  | yes  | 200 |
| DPO     | 1pair b4        | 200step  | 40.4% | +7.9%  | 0.228 | 83.1  | yes  | 200 |
| SFT     | 2pair 200step   |          | 37.7% | +5.2%  | 0.229 | 86.5  | yes  | 200 |
| GRPO    | mv2 g16         | 1ep      | pending (**2819355**) | | | | | |
| GRPO    | v2 g16 (single) | 1ep      | training (**2819413**) | | | | | |

GDPO 3B SNLI-P +17.5%: strong result for a 3B model — nearly matches 7B GDPO v6 (+15.7%) and well ahead of 3B DPO/SFT.

### SNLI-Hypothesis (3B)

Base LFR 48.1% (similar to 7B 44%). Anchor: `evaluation_snli_hypothesis_200s_dpo_1pair_qwen25_3b/`.

| Method  | Config          | Duration | LFR   | ΔLFR   | NED   | PPL    | Fair | N   |
| ------- | --------------- | -------- | ----- | ------ | ----- | ------ | ---- | --- |
| GDPO v6 | g16 bs8         | 1ep      | 61.9% | +13.6% | 0.409 | 203.9  | yes  | 200 |
| SimPO   | 2pair b4 β3γ0.5 | 2ep      | 54.9% | +6.8%  | 0.381 | 2672   | yes  | 200 |
| SFT     | 2pair 200step   |          | 51.2% | +3.1%  | 0.396 | 174.5  | yes  | 200 |
| DPO     | 1pair b4        | 200step  | 50.0% | +1.9%  | 0.374 | 131.6  | yes  | 200 |
| GRPO    | mv2 g16         | 1ep      | pending (**2819356**) | | | | | |
| GRPO    | v2 g16 (single) | 1ep      | training (**2819414**) | | | | | |

GDPO v6 3B SNLI-H +13.6%: **best 3B result on SNLI-H by a wide margin** (+6.8pp over SimPO). SimPO SNLI-H PPL=2672 exceeds the 650 cap — excluded from visual summary. 3B produces less fluent SNLI-H edits than 7B/14B at most methods, but GDPO maintains reasonable PPL (203.9).

**Training / eval status (3B):**

| Dataset     | DPO                  | SimPO                | SFT                  | GRPO multi           | GRPO single           | GDPO v6               |
| ----------- | -------------------- | -------------------- | -------------------- | -------------------- | --------------------- | --------------------- |
| BoolQ       | ✓ +0.0%              | eval pending **2819352** | eval pending **2819353** | eval pending **2819354** | training **2819412** | 5ep retraining **2819410**; ckpt-100 eval **2819409** |
| SNLI-Premise| ✓ +7.9%              | ✓ +25.5%             | ✓ +5.2%              | eval pending **2819355** | training **2819413** | ✓ +17.5%              |
| SNLI-H      | ✓ +1.9%              | ✓ +6.8% (PPL excl.)  | ✓ +3.1%              | eval pending **2819356** | training **2819414** | ✓ +13.6%              |

---

## Completed Evaluations

All prior experiments have completed. Key recent completions:

| Job ID | Name | Result |
|--------|------|--------|
| 2773359 | 14B GRPO mv2 g16 SNLI-P v2fix (train+resume) | `grpo_model_snli_premise_1ep_g16_multi_qwen25_14b_v2fix/` — 1 epoch; eval **2777871** → `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/` |
| 2771568 | eval-grpo-snli-h-v2fix | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix/` — see SNLI-H table (GRPO mv2 g16 v2fix) |
| 2771570 | eval-gdpo-snli-p-v2fix | `evaluation_snli_premise_200s_gdpo_v6_g16_v2fix/` — see SNLI-P table (GDPO v6 v2fix) |
| 2773355 | eval-dpo-snli-p-v2fix | `evaluation_snli_premise_200s_dpo_2pair_b4_2ep_v2fix/` — DPO 2pair v2fix; **+14.3%** ΔLFR (66.1% LFR); see SNLI-P table |
| 2773356 | eval-dpo-snli-h-v2fix | `evaluation_snli_hypothesis_200s_dpo_2pair_b4_2ep_v2fix/` — DPO 2pair v2fix; **+5.4%** ΔLFR; see SNLI-H table |
| 2773357 | eval-simpo-snli-p-v2fix | `evaluation_snli_premise_200s_simpo_2pair_v2fix/` — SimPO 2pair v2fix; **+30.5%** ΔLFR (83.4% LFR); see SNLI-P table |
| 2773358 | eval-simpo-snli-h-v2fix | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix/` — SimPO b3γ0.5 v2fix; **+29.4%** ΔLFR; tuned `total_cfs` **484** vs base 1162 (see Metrics — parsed CFs); see SNLI-H table |
| 2771532 | 14B BoolQ DPO eval (historical submit) | Superseded by **2777870** → `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/` (see 14B BoolQ table) |
| 2777870 | eval-14b-boolq-dpo-resume | `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/` — **+0.3%** ΔLFR, 55.2% LFR |
| 2777871 | eval-14b-snli-p-grpo | `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/` — **+15.0%** ΔLFR, 78.0% LFR |
| 2777872 | eval-14b-snli-h-gdpo | `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_14b_v2fix/` — **+8.8%** ΔLFR, 85.0% LFR |
| 2777899 | 14b-pairs-boolq-2p | BoolQ 2-pair prep for 14B SimPO (`dpo_pairs_boolq_*_qwen25_14b` tree); **COMPLETED** |
| 2777928 | 14b-simpo-boolq-b3g05 | `dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05_qwen25_14b/` — train **COMPLETED** |
| 2780617 | eval-14b-boolq-simpo | `evaluation_boolq_200s_simpo_b3g05_qwen25_14b/` — **+7.0%** ΔLFR, 62.5% LFR, NED 0.248, PPL 9.1 |
| 2781362 | eval-14b-snli-p-sft | `evaluation_snli_premise_200s_sft_2pair_b16_1ep_qwen25_14b/` — **-0.0%** ΔLFR, 63.1% LFR |
| 2781363 | eval-14b-snli-p-dpo | `evaluation_snli_premise_200s_dpo_2pair_b4_2ep_qwen25_14b/` — **+6.6%** ΔLFR, 69.4% LFR |
| 2781361 | eval-14b-boolq-sft | `evaluation_boolq_200s_sft_2pair_b4_ml2048_200step_qwen25_14b/` — **+0.8%** ΔLFR, 55.7% LFR |
| 2781364 | eval-14b-snli-h-dpo | `evaluation_snli_hypothesis_200s_dpo_2pair_b4_2ep_qwen25_14b/` — **+0.7%** ΔLFR, 76.7% LFR |
| 2781365 | eval-14b-snli-p-simpo | `evaluation_snli_premise_200s_simpo_b3g05_qwen25_14b/` — **+21.5%** ΔLFR, 84.0% LFR |
| 2781366 | eval-14b-snli-h-simpo | `evaluation_snli_hypothesis_200s_simpo_b3g05_qwen25_14b/` — **+13.7%** ΔLFR, 89.9% LFR |
| 2786425 | eval-14b-gdpo-snli-p-ckpt500 | `evaluation_snli_premise_200s_gdpo_v6_ckpt500_qwen25_14b/` — **+6.6%** ΔLFR (69.7% LFR, NED 0.316, PPL 93.1); GDPO v6 ckpt-500 (~0.25ep of timed-out train) |
| 2786426 | eval-14b-gdpo-boolq-ckpt200 | `evaluation_boolq_200s_gdpo_v6_ckpt200_qwen25_14b/` — **+0.5%** ΔLFR (55.3% LFR, NED 0.292, PPL 8.8); GDPO v6 ckpt-200 (early partial, OOM at ~0.22ep) |
| 2786445 | 14b-sft-snli-h (train) | `sft_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b/` — training complete (1:38h) |
| 2794799 | eval-sft-snlih-14b | `evaluation_snli_hypothesis_200s_sft_2pair_qwen25_14b/` — **+3.9%** ΔLFR (80.2% LFR, NED 0.482, PPL 186.9) |
| 2794877 | eval-14b-fair (GRPO BoolQ) | `evaluation_boolq_200s_grpo_mv2_g16_qwen25_14b/` — **+1.7%** ΔLFR (56.7% LFR, NED 0.264, PPL 8.5) |
| 2794878 | eval-14b-fair (GRPO SNLI-H) | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_qwen25_14b/` — **+9.2%** ΔLFR (85.5% LFR, NED 0.454, PPL 242.6) |
| 2794876 | 14b-gdpo-boolq-resume | `gdpo_model_boolq_1ep_g16_qwen25_14b_v6/` — resumed from ckpt-200 (bs=4, gen\_batch=64 on H200); reached ckpt-2000 + final adapter. Full-epoch eval submitted as **2805602** |
| 2794879 | 14b-gdpo-snlip-resume | `gdpo_model_snli_premise_1ep_g16_qwen25_14b_v6/` — resumed from ckpt-800; reached ckpt-4000 + final adapter. Full-epoch eval submitted as **2805603** |
| 2782111 | eval-grpo-boolq-v2fix-resume (7B) | `evaluation_boolq_200s_grpo_mv2_g16_v2fix/` — **−28.3%** ΔLFR (8.7% LFR); likely **policy collapse** after long GRPO; see BoolQ table row |
| 2771585–96 | SimPO SNLI-H **simpo_qual** (6 trains + 6 evals) | Extra β/γ/CPO-α cells; see SNLI-H table (`simpo_qual_*`); jobs **2771585**–**2771596** |
| 2735771 | SimPO BoolQ b3γ0.5 fair N=200 | **+21.8%** ΔLFR (59.0% LFR), NED 0.167, PPL 12.5; `dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05/eval_boolq_200/` |
| 2735772 | SimPO SNLI-P b3γ0.5 fair N=200 | **+23.5%** ΔLFR (76.1% LFR), NED 0.362, PPL 131.2 |
| 2735774 | SimPO SNLI-H b3γ0.5 CPO α=0.1 | **+25.2%** ΔLFR; PPL mean **6427** (skewed by outliers — check generations) |
| 2735775–79 | SFT SNLI-P fair N=200 grid | Best **+0.9%** (200step b4, b16 1ep); 0.5ep and 2ep b4 **−2%** to **−2.7%**; see SNLI-P table |
| 2733429–31 | GDPO v6 SNLI-H trick ablation (eval) | No cond: **-4.6%**; no clip: +13.0%; no KL: +37.1% but PPL=1360 (fluency collapse). Conditioned rewards are essential; KL acts as quality regularizer |
| 2733454/57 | SimPO SNLI-P / SNLI-H (train+eval) | SNLI-P **+27.1%** (near GRPO +27.5%); SNLI-H **+26.4%** (beats DPO +21% and GDPO v6 +18.7%; legacy report PPL `nan` — see PPL note in metrics + `evaluate_models.py` fix) |
| 2735485–96 | SimPO SNLI-H β×γ grid (6 train + 6 eval) | Best **b3g05** +29.7% ΔLFR, NED 0.428, finite-mean PPL ~1202; **γ=0.5** beats **γ=1.4** at same β on ΔLFR; duplicate config **b2g14** +24.3% vs first-run β2γ1.4 +26.4% (~2pp run variance) |
| 2733453/59/61 | SimPO / DPO β=0.05 / IPO lr=1e-6 BoolQ (eval) | SimPO **+9.8%** (46.7% LFR); β=0.05 **+5.8%** (worse PPL 29.2); IPO lr=1e-6 **+0.3%** (removes −11.7% blow-up, still below sigmoid +10.5%) |
| 2659188/89 | GRPO SFT warm-start BoolQ | +1.2% ΔLFR (no improvement over regular GRPO) |
| 2661976/77 | GDPO v5 SNLI-P | +1.7% ΔLFR (8 groups, lr=1e-6) |
| 2663697/98 | GDPO v6 SNLI-P | +15.7% ΔLFR (8 groups, lr=5e-6, conditioned rewards) |
| 2664934/64 | GDPO paper-match SNLI-P | +7.6% ΔLFR (4 GPU, 32 groups, TRL built-in) |
| 2664936/65 | GRPO paper-match SNLI-P | +25.9% ΔLFR (4 GPU, 32 groups, control) |
| 2663406 | SFT LR sweep lr=1e-5 | Negative across all datasets |
| 2663407 | SFT LR sweep lr=2e-5 | Negative/mixed, NaN PPL on BoolQ |
| 2674320/21 | GDPO v7 SNLI-P (4GPU, 32 groups, v6 innovations) | +15.7% ΔLFR (same as v6; more groups don't help) |
| 2674322 | GDPO v6 ckpt-2000 SNLI-P (~0.25ep) | **+18.2% ΔLFR** (best GDPO result) |
| 2674323 | GDPO v6 ckpt-4000 SNLI-P (~0.50ep) | +18.1% ΔLFR |
| 2674324 | GDPO v6 ckpt-6000 SNLI-P (~0.75ep) | +16.5% ΔLFR |
| 2674329 | SFT BoolQ ml2048 | +1.8% ΔLFR (improvement over -0.0% at ml1024) |
| 2676674 | GDPO v6 SNLI-H ckpt-2000 (~0.25ep) | +15.9% ΔLFR |
| 2679269 | GDPO v6 SNLI-H ckpt-4000 (~0.50ep) | +17.6% ΔLFR |
| 2681518 | GDPO v6 SNLI-H ckpt-6000 (~0.75ep) | **+18.7% ΔLFR** (beats GRPO mv2 +15.1%; best GDPO result) |
| 2674337 | GDPO v6 SNLI-H final (1.0ep) | +17.6% ΔLFR (decline from 0.75ep peak) |
| 2676665 | DPO BoolQ ml2048 (2pair b4) | +8.5% ΔLFR (below 1pair b4 +10.5%) |
| 2676666 | GDPO v6 BoolQ mcl1024 | +0.3% ΔLFR (mcl1024 didn't help) |
| 2674326 | GRPO mv2 BoolQ mcl1024 | TIMEOUT after 24h (reached ckpt-4400) |
| 2805602 | eval-14b-gdpo-boolq-full | `evaluation_boolq_200s_gdpo_v6_qwen25_14b/` — **+6.0%** ΔLFR (60.8% LFR, NED 0.186, PPL 8.3); best NED on BoolQ 14B |
| 2805603 | eval-14b-gdpo-snlip-full | `evaluation_snli_premise_200s_gdpo_v6_qwen25_14b/` — **+26.2%** ΔLFR (89.6% LFR, NED 0.361, PPL 79.2); best 14B result overall |
| 2811911 | 3b-eval-dpo-snlip | `evaluation_snli_premise_200s_dpo_1pair_qwen25_3b/` — **+7.9%** ΔLFR (40.4% LFR, NED 0.228, PPL 83.1) |
| 2811912 | 3b-eval-dpo-snlih | `evaluation_snli_hypothesis_200s_dpo_1pair_qwen25_3b/` — **+1.9%** ΔLFR (50.0% LFR, NED 0.374, PPL 131.6) |
| — | 3b-eval-simpo-snlip | `evaluation_snli_premise_200s_simpo_b3g05_qwen25_3b/` — **+25.5%** ΔLFR (59.9% LFR, NED 0.309, PPL 95.1) |
| — | 3b-eval-simpo-snlih | `evaluation_snli_hypothesis_200s_simpo_b3g05_qwen25_3b/` — **+6.8%** ΔLFR (54.9% LFR, NED 0.381, PPL 2672; excluded from visual summary) |
| — | 3b-eval-sft-snlip | `evaluation_snli_premise_200s_sft_qwen25_3b/` — **+5.2%** ΔLFR (37.7% LFR, NED 0.229, PPL 86.5) |
| — | 3b-eval-sft-snlih | `evaluation_snli_hypothesis_200s_sft_qwen25_3b/` — **+3.1%** ΔLFR (51.2% LFR, NED 0.396, PPL 174.5) |
| 2811909 | 3b-eval-gdpo-snlip | `evaluation_snli_premise_200s_gdpo_v6_qwen25_3b/` — **+17.5%** ΔLFR (50.6% LFR, NED 0.269, PPL 76.3); GDPO v6 1ep |
| 2811910 | 3b-eval-gdpo-snlih | `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_3b/` — **+13.6%** ΔLFR (61.9% LFR, NED 0.409, PPL 203.9); GDPO v6 1ep; best 3B SNLI-H result |
| 2819352 | 3b-eval-boolq-simpo | Submitted. BoolQ 3B SimPO b3γ0.5 eval (anchor: `evaluation_boolq_200s_dpo_1pair_qwen25_3b`) |
| 2819353 | 3b-eval-boolq-sft   | Submitted. BoolQ 3B SFT 200step eval |
| 2819354 | 3b-eval-boolq-grpo  | Submitted. BoolQ 3B GRPO mv2 g16 eval |
| 2819355 | 3b-eval-snlip-grpo  | Submitted. SNLI-P 3B GRPO mv2 g16 eval |
| 2819356 | 3b-eval-snlih-grpo  | Submitted. SNLI-H 3B GRPO mv2 g16 eval |
| 2819409 | 3b-eval-boolq-gdpo  | Submitted. BoolQ 3B GDPO v6 ckpt-100 (~1.6ep) interim eval |
| 2819410 | 3b-gdpo-boolq-5ep   | Submitted. Fresh GDPO BoolQ 3B training, 5 epochs (`gdpo_model_boolq_5ep_g16_qwen25_3b_v6`) |
| 2819412–14 | 3b-grpo-single | Training. GRPO v2 g16 (single reward) for 3B BoolQ/SNLI-P/SNLI-H |
| 2819415–17 | 14b-grpo-single | Training. GRPO v2 g16 (single reward) for 14B BoolQ/SNLI-P/SNLI-H on H200 |


---

## Training Data Statistics


| Dataset         | 1-pair Samples | 2-pair Samples | Entries with Pairs    |
| --------------- | -------------- | -------------- | --------------------- |
| BoolQ           | 1,647          | 3,191          | 1,647 / 2,000 (82.4%) |
| SNLI-Premise    | 1,955          | 3,810          | 1,955 / 2,000 (97.8%) |
| SNLI-Hypothesis | 1,735          | 3,244          | 1,735 / 2,000 (86.8%) |


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

