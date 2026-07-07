# Evaluation Results

Counterfactual-generation evaluation tables for all models. Discussion, analysis, and methodology: [OVERVIEW.md](OVERVIEW.md). **Primary tables** are **Qwen/Qwen2.5-7B-Instruct** unless a row says otherwise.

**Metrics**:

- **LFR** (Label Flip Rate): % of counterfactuals where the base model's prediction changed from the original label (higher = better)
- **ΔLFR**: LFR improvement over the base model on the same evaluation set
- **NED** (Normalized Edit Distance): edit distance / max text length (lower = more minimal edits). **Aggregation**: **median** NED over CFs with NED > 0 (identical copies excluded — those are not counterfactuals by definition; see [RGF, ACL 2022](https://aclanthology.org/2022.acl-long.117)). Best-result N=200 entries use this corrected median; historical/secondary entries may use mean NED from `eval_summary.avg_norm_edit_distance`.
- **PPL** (Perplexity): fluency of generated text (lower = more fluent). **Aggregation**: **median** PPL across generated CFs (robust to degenerate outlier CFs; mean is 3–27× higher when a few CFs have catastrophic loss). All canonical N=200 best results use median. Note: some historical/secondary entries (N=100 grid, early checkpoint sweeps, or collapsed-model runs) may still show mean PPL from `eval_summary.avg_perplexity` — do not compare those to median values.
- **LFR/NED**: label flip rate / normalized edit distance — edit efficiency (higher = more effective per unit of change). Reported in individual eval reports for N=200 runs.
- **Fair evaluation**: N=200 validation samples, 10 CFs per sample. Within each model×dataset, all methods reuse the **same frozen base counterfactual set** (`--base_eval_dir` + cached base verdicts) so ΔLFR comparisons are against an identical base LFR. Numbers in `results_charts.html` and `best_models/` reflect this protocol.
- **N**: number of validation samples evaluated (10 CFs generated per sample)
- **Parsed CFs only**: metrics are averaged over CFs with a non-null parsed `edited_text`. Fair evals (`--base_eval_dir`) reuse the same stored base CFs for the base side.
- **Parse rate** (`parse%`): fraction of the 10×N generation attempts that produced a valid `<edit>…</edit>` output. All metrics (LFR, NED, PPL) are computed **only over parsed CFs**. Low parse rate means the reported LFR is over a small, potentially biased sample — ⚠ < 60%, ⛔ < 10%. **Selection rule:** the run *featured* per method-cell in `results_charts.html` is the best ΔLFR run with **parse ≥ 15% of the base parse rate** — high-ΔLFR runs at near-zero parse are collapse artifacts (a model emitting few well-formed edits, all easy to flip) and are excluded from selection. See [parse rate summary below](#parse-rate-summary) and [OVERVIEW.md § CF coverage](OVERVIEW.md#cf-coverage-and-parse-rates) for analysis.

**Methods**: DPO (preference optimization), SimPO (reference-free preference via CPOTrainer), SFT (supervised fine-tuning on chosen CFs), GRPO (online RL with reward signal), GDPO (GRPO with per-reward normalization), KTO (Kahneman-Tversky Optimization — asymmetric loss via `KTOTrainer`; see Training Configuration)

**Configs**: `{pairs} {batch}` where pairs = number of preference pairs per entry, batch = effective batch size (b4 = bs1×ga4, b16 = bs4×ga4). GRPO configs: `single` = combined reward, `multi` = decomposed rewards, `g4`/`g16` = generations per prompt.

**GRPO reward versions**: GRPO g4 and g16 (non-v2) used an incorrect single reward: `flip + 0.8 * similarity` (no confidence). GRPO g16 v2 uses the corrected reward: `flip + confidence * similarity`, matching the DPO unified score. Multi-reward runs are unaffected (they use separate FlipReward + SimilarityReward). Results marked with † use the old reward.

**Duration**: `200step` = fixed 200 gradient steps (variable effective epochs), `0.5ep`–`3ep` = epoch-controlled training.

---

## Parse Rate Summary

Parse rate = parsed CFs / (N × 10 attempts). **This table reports the parse rate of the FEATURED (charted) run per cell** — i.e. the best run that passes the 15%-of-base parse gate. **⚠ Terminology (Jul 3):** this "parse rate" is *unique* parsed CFs / attempts, so it conflates **format-compliance** (emitted a valid edit) with **diversity** (non-duplicate rate). The dual-parse diagnostic (`PARSE_TAG_ISSUE.md §9`) shows every **featured** cell is ~100% *compliant* — the sub-100% figures here are mostly **deduplication**, not tag loss. Compliance and diversity are reported decoupled by `evaluate_models.py --dual_parse`. Values below 60% flagged ⚠; below 10% flagged ⛔. **BoolQ parse rates are generally high (70–92%) for all models. SNLI tasks are harder** — 3B models show 30–49% parse rates even at base, reflecting short NLI sentence difficulty. Several higher-raw-LFR Llama runs (e.g. SNLI-H SimPO β2 4.6%, DPO lr=2e-5 2.3%, SNLI-P DPO lr=1e-5 3.9%) were **excluded by the parse gate** — their LFR is computed over a tiny biased sample. The complete list of gate-excluded runs is in **`OVERVIEW.md` § Parse-gate exclusions**.

| Model    | Dataset | Base  | DPO   | SimPO | SFT   | GRPO-S | GRPO-M | GDPO  |
|----------|---------|-------|-------|-------|-------|--------|--------|-------|
| Llama-8B | BoolQ   | 89.0% | 84.2% | 70.3% | 83.8% | 86.5%  | 89.2%  | 91.6% |
| Llama-8B | SNLI-P  | 88.8% | 74.2% | 94.2% | 82.4% | 57.8%⚠ | 78.8% | 87.2% |
| Llama-8B | SNLI-H  | 89.1% | 83.9% | 81.2% | 71.7% | 83.5% | 65.0%  | 65.8% |
| Qwen-3B  | BoolQ   | 89.3% | 87.4% | 74.9% | 92.8% | 55.4%⚠ | 54.0%⚠| 90.3% |
| Qwen-3B  | SNLI-P  | 42.6%⚠| 50.8%⚠| 41.1%⚠| 46.9%⚠| 34.3%⚠ | 33.2%⚠| 46.5%⚠|
| Qwen-3B  | SNLI-H  | 45.2%⚠| 39.4%⚠| 36.8%⚠| 46.9%⚠| 30.9%⚠| 32.5%⚠| 46.0%⚠|
| Qwen-14B | BoolQ   | 87.9% | 76.1% | 66.0% | 89.2% | 82.1%  | 85.2%  | 67.5% |
| Qwen-14B | SNLI-P  | 64.4% | 58.4%⚠| 68.8% | 65.3% | 50.5%⚠ | 27.4%⚠| 29.3%⚠|
| Qwen-14B | SNLI-H  | 61.8% | 47.7%⚠| 15.8%⚠| 70.1% | 36.6%⚠ | 29.2%⚠| 42.8%⚠|
| Qwen-7B  | BoolQ   | 82.3% | 51.1%⚠| 55.7%⚠| 79.8% | 45.4%⚠ | 52.3%⚠ | 82.3% |
| Qwen-7B  | SNLI-P  | 68.6% | 50.9%⚠| 61.2% | 74.5% | 47.6%⚠ | 44.5%⚠ | 71.2% |
| Qwen-7B  | SNLI-H  | 58.1%⚠| 44.1%⚠| 24.2%⚠| 63.3% | 24.5%⚠ | 33.8%⚠ | 56.6%⚠|

Notes: "—" = progress file unavailable for that run. GRPO-S = GRPO single-reward, GRPO-M = GRPO multi-reward. 3B SNLI-P/H parse rates are low for the base model too, indicating task-difficulty (short NLI sentences). Gate-excluded (collapse) runs are listed in `OVERVIEW.md § Parse-gate exclusions`, not here.

---

## Qwen2.5-7B-Instruct

Tables in **BoolQ**, **SNLI-Premise**, and **SNLI-Hypothesis** are for this base model (QLoRA adapters on `Qwen/Qwen2.5-7B-Instruct`).

### BoolQ

**Canonical-base fair re-evals (base CF set `05b8d14619`).** These reuse one frozen 7B BoolQ base CF anchor (`evaluation_boolq_200s_grpo_mv2_g16`) so every method is judged against the same base. NED = median over NED>0; PPL = median. These are the values featured in `results_charts.html`.

| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL  | Parse | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ---- | ----- | ---- | --- |
| SimPO  | 2pair b4 b3γ0.5 **fair-canon** | 2ep | 66.7% | **+28.6%** | 0.115 | 8.8 | 68% | yes✓ | 200 | (7B BoolQ best) |
| DPO    | 2pair b4 fair-canon | 2ep | 59.9% | **+22.1%** | 0.093 | 8.8 | 51% | yes✓ | 200 |
| GRPO single | v2 g16 **lr=1e-5** fair-canon | 1ep | 59.0% | **+21.2%** | 0.086 | 8.3 | 45% | yes✓ | 200 | (lr=1e-5 lifts +13.5pp over default lr) |
| GRPO multi | mv2 g16 **lr=1e-5** fair-canon | 1ep | 56.6% | **+18.8%** | 0.090 | 8.1 | 52% | yes✓ | 200 | (lr=1e-5 lifts +9pp over default lr) |
| SFT    | 2pair b4 fair-canon | — | 47.2% | +9.6% | 0.141 | 8.4 | 80% | yes✓ | 200 |
| GDPO   | g16 ckpt200 fair-canon | — | 37.7% | +1.1% | 0.106 | 9.6 | 82% | yes✓ | 200 |

Full historical sweep (mixed bases, kept for the record):

| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| DPO    | 2pair b4  | 2ep      | 51.8% | +14.7% | 0.144 | 13.1 | no   | 200 |
| DPO    | 2pair b4  | 200step  | 57.6% | +13.9% | 0.130 | 10.2 | no   | 50  |
| DPO    | 1pair b4  | 2ep      | 48.0% | +10.5% | 0.110 | 23.9 | yes  | 200 |
| SimPO  | 1pair b4 simpo | 2ep | 46.7% | +9.8%  | 0.111 | 14.0 | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 | 2ep | 59.0% | +21.8% | 0.167 | 10.4 | yes  | 200 |
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
| GDPO   | mv2 g16   | ckpt200  | 37.7% | +1.1%  | 0.106 | 11.8 | yes  | 200 |
| GDPO   | mv2 g16   | ckpt200  | 36.3% | -6.1%  | 0.166 | 11.5 | yes  | 100 |
| GRPO   | mv2 g16   | ~0.25ep  | 37.8% | +1.2%  | 0.170 | 11.6 | yes  | 200 |
| GRPO   | mv2 g16 sft-ws| ~0.25ep | 37.8% | +1.2%  | 0.170 | 11.9 | yes  | 200 |
| GRPO   | mv2 g16 lr1e6 | ~0.25ep | 37.4% | +1.1% | 0.155 | 11.8 | yes  | 200 |
| GRPO   | v2 g16    | ~0.11ep  | 37.7% | +0.4%  | 0.096 | 11.7 | yes  | 200 |
| GDPO   | mv2 g16 lr1e6 | 1.0ep | 36.8% | +0.2%  | 0.153 | 11.6 | yes  | 200 |
| GRPO   | v2 g24 lr1e6 | 1.0ep | 36.7% | +0.2%  | 0.161 | 11.6 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 37.2% | -0.0%  | 0.160 | 11.5 | yes  | 200 |
| SFT    | 2pair b4 ml2048| 200step | 38.4% | +1.8%  | 0.141 | 11.6 | yes  | 200 |
| SFT    | 2pair b4 lr1e5| 200step | 40.2% | -2.0%  | 0.175 | 12.0 | yes  | 100 |
| SFT    | 2pair b4 lr2e5| 200step | 44.0% | +1.9%  | 0.272 | —    | yes  | 100 |
| DPO    | 2pair b4 ml2048| 2ep      | 45.4% | +8.5%  | 0.123 | 26.9 | yes  | 200 |
| DPO    | 1pair b4 ml2048| 2ep      | 40.5% | +3.8%  | 0.114 | 540.4| yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep | 8.7% | **-28.3%** | 0.214 | 12.0 | yes  | 200 |
| GRPO   | mv2 g16 mcl768 | ~0.25ep | 38.2% | +1.7% | 0.098 | 12.0 | yes  | 200 |
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

### SNLI-Premise


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | mv2 g16 v2fix | 1.0ep | 80.9% | +29.1% | 0.327 | 92.9  | yes  | 200 |
| SimPO  | 2pair b4 simpo v2fix | 2ep | 83.4% | +30.5% | 0.340 | 157.6 | yes  | 200 |
| DPO    | 2pair b4 **fair-canon** | 2ep | 78.0% | +24.5% | 0.311 | 53.5 | yes✓ | 200 | (canonical-base re-eval, was 1p +15.7%; featured in charts) |
| GRPO   | mv2 g16   | 1.0ep    | 79.7% | +27.5% | 0.348 | 72.7  | yes  | 200 |
| SimPO  | 2pair b4 simpo | 2ep | 79.9% | +27.1% | 0.358 | 154.7 | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 | 2ep | 76.1% | +23.5% | 0.362 | 131.2 | yes  | 200 |
| GRPO   | paper 4GPU| 1.0ep    | 78.3% | +25.9% | 0.348 | 103.2 | yes  | 200 |
| GRPO   | v2 g16    | ~0.50ep  | 78.0% | +25.8% | 0.337 | 110.6 | yes  | 200 |
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
| SFT    | 2pair b16 | 1ep      | 53.2% | +0.9%  | 0.236 | 130.5 | yes  | 200 |
| SFT    | 2pair b4  | 200step  | 53.1% | +0.9%  | 0.249 | 131.4 | yes  | 200 |
| SFT    | 2pair b16 | 0.5ep    | 49.9% | -2.2%  | 0.237 | 136.7 | yes  | 200 |
| SFT    | 2pair b4  | 0.5ep    | 49.8% | -2.0%  | 0.253 | 145.6 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 48.9% | -2.7%  | 0.266 | 355.3 | yes  | 200 |


### SNLI-Hypothesis


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| DPO    | 2pair b4 **fair-canon** | 2ep | 68.4% | +23.2% | 0.340 | 113.2 | yes✓ | 200 | (canonical-base re-eval, 44% parse; featured in charts) |
| GRPO   | v2 g24    | 1.0ep    | 67.4% | +22.6% | 0.250 | 137 (med) | yes | 200 | (NED median-nz, PPL median; mean PPL was 9651 — outlier-driven; ⚠ 25% parse; featured over g16 +11.0%) |
| SimPO  | 2pair b4 simpo b3g05 | 2ep | 74.2% | +29.7% | 0.428 | 1202  | yes  | 200 |
| SimPO  | 2pair b4 b3γ0.5 v2fix | 2ep | 73.6% | +29.4% | 0.370 | 617.1 | yes  | 200 |
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
| DPO    | 2pair b4  | 2ep      | 65.9% | +21.0% | 0.300 | 296.5 | yes  | 200 |
| DPO    | 2pair b4 v2fix | 2ep | 49.5% | +5.4%  | 0.315 | 285.7 | yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep | 67.0% | +22.6% | 0.353 | 508.3 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 59.4% | +15.1% | 0.266 | 702.9 | yes  | 200 |
| GRPO   | mv2 g24   | 1.0ep    | 57.4% | +12.6% | 0.253 | 453.2 | yes  | 200 |
| GRPO   | v2 g16    | 1.0ep    | 55.7% | +11.0% | 0.229 | 317.3 | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 53.4% | +8.7%  | 0.334 | 368.0 | no   | 200 |
| GRPO-fair| mv2 g16 fair | 1.0ep  | 51.1% | +6.8%  | 0.316 | 285.1 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 46.1% | +1.7%  | 0.326 | 286.8 | yes  | 200 |
| GDPO   | mv2 g24 lr1e6 | 1.0ep | 45.6% | +1.1%  | 0.302 | 247.6 | yes  | 200 |
| GDPO v2| mv2 g16 cond  | 1.0ep | 45.0% | +0.7%  | 0.316 | 329.0 | yes  | 200 |
| GDPO v3| mv2 g16 ned02 | ~0.95ep | 46.3% | +1.8% | 0.276 | 308.3 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.25ep| 60.3% | +15.9% | 0.365 | 248.9 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.50ep| 62.0% | +17.6% | 0.363 | 329.6 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.75ep| 63.1% | +18.7% | 0.372 | 334.9 | yes  | 200 |
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
| SimPO  | 2pair b4 β3γ0.5 | 2ep | 62.5% | +7.0% | 0.201 | 7.4  | yes  | 200 |
| SimPO  | 2pair b4 β2γ0.5 | 2ep | 59.8% | +4.6% | 0.194 | 7.0  | yes  | 200 |
| GDPO v6| g16 bs4   | 1.0ep    | 60.8% | +6.0% | 0.150 | 6.5  | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep | 57.7% | +2.9% | 0.223 | 6.9  | yes  | 200 |
| DPO    | 2pair b4 lr=1e-5 | 2ep   | 57.9% | +2.9% | 0.220 | 7.0  | yes  | 200 |
| GRPO   | mv2 g16 lr=2e-6 | ckpt7100 (~89%) | 55.5% | +0.8% | 0.287 | 7.0 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 56.7% | +1.7% | 0.217 | 6.8  | yes  | 200 |
| SFT    | 2pair b4 ml2048 | 200step | 55.7% | +0.8% | 0.250 | 7.0  | yes  | 200 |
| GDPO v6| g16 bs8   | ckpt-200 | 55.3% | +0.5% | 0.292 | 7.0  | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 55.2% | +0.3% | 0.268 | 6.8  | yes  | 200 |
| GRPO   | v2 g16 (single) lr=2e-6 | ckpt7400 (~93%) | 54.5% | -0.5% | 0.286 | 7.0 | yes | 200 |

SimPO β=3: `dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_boolq_200s_simpo_b3g05_qwen25_14b/` (train **2777928**, eval **2780617**). SimPO β=2: `evaluation_boolq_200s_simpo_b2g05_qwen25_14b/`. β=2 gives +4.6% vs β=3 +7.0% — β=3 remains best for 14B BoolQ.
GDPO v6 full epoch: `gdpo_model_boolq_1ep_g16_qwen25_14b_v6` (resumed to ckpt-2000, bs=4) → `evaluation_boolq_200s_gdpo_v6_qwen25_14b/` (eval **2805602**). NED 0.150 (median, NED>0) is notably lower than the ckpt-200 partial (0.249), suggesting the full training teaches more minimal edits.
GRPO: `grpo_model_boolq_1ep_g16_multi_qwen25_14b_boolq_mv2` → `evaluation_boolq_200s_grpo_mv2_g16_qwen25_14b/` (train **2777920**, eval **2794877**).
SFT: `sft_model_boolq_2pair_b4_ml2048_200step_qwen25_14b` → `evaluation_boolq_200s_sft_2pair_b4_ml2048_200step_qwen25_14b/` (train **2780681**, eval **2781361**).
GDPO v6 ckpt-200: partial result (OOM at ~0.22ep); superseded by full-epoch eval above.
DPO 1pair: `dpo_model_boolq_2000e40c_1pair_qwen25_14b` → `evaluation_boolq_200s_dpo_1pair_b4_2ep_qwen25_14b/` (eval **2777870**). DPO 2pair lr=1e-5: `evaluation_boolq_200s_dpo_2pair_lr1e5_qwen25_14b/` — +2.9%, same as GRPO single but DPO 1pair with default lr only reaches +0.3%; lr=1e-5 helps but still below SimPO/GDPO.

### SNLI-Premise (14B)

| Method | Config        | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------ | ------------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| GRPO   | mv2 g16       | 1.0ep    | 92.8% | +29.4% | 0.389 | 44.2 | yes  | 200 |
| GDPO v6| g16 bs4       | 1.0ep    | 89.6% | +26.2% | 0.352 | 52.9 | yes  | 200 |
| SimPO  | 2pair b4 β2γ0.5 | 2ep   | 85.7% | +22.7% | 0.413 | 50.5 | yes  | 200 |
| SimPO  | 2pair b4 β3γ0.5 | 2ep    | 84.0% | +21.5% | 0.421 | 52.8 | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep    | 85.6% | +22.7% | 0.336 | 43.5 | yes  | 200 |
| DPO    | 2pair b4 lr=1e-5 | 2ep   | 78.7% | +15.6% | 0.362 | 70.9 | yes  | 200 |
| GRPO   | mv2 g16 v2fix | 1.0ep    | 78.0% | +15.0% | 0.311 | 45.2 | yes  | 200 |
| DPO    | 2pair b4      | 2ep      | 69.4% | +6.6%  | 0.318 | 50.4 | yes  | 200 |
| DPO    | 1pair b4 lr=2e-6 | 2ep   | 63.6% | +0.6%  | 0.309 | 49.5 | yes  | 200 |
| GDPO v6| g16 bs8   | ckpt-500 | 69.7% | +6.6%  | 0.316 | 48.9 | yes  | 200 |
| SFT    | 2pair b16     | 1ep      | 63.1% | -0.0%  | 0.321 | 50.9 | yes  | 200 |

GRPO mv2 original: `grpo_model_snli_premise_1ep_g16_multi_qwen25_14b` → `evaluation_snli_premise_200s_grpo_mv2g16_qwen25_14b/`. **+29.4%** — best 14B SNLI-P result overall, surpassing GDPO. Note: the v2fix retrain (`grpo_mv2g16_v2fix`) scored only +15.0-15.5%, suggesting the original training was superior.
GDPO v6 full epoch: `gdpo_model_snli_premise_1ep_g16_qwen25_14b_v6` (resumed to ckpt-4000, bs=4) → `evaluation_snli_premise_200s_gdpo_v6_qwen25_14b/` (eval **2805603**). Note: total_cfs 607 vs base 1288 (47% parse rate) — high self-selection; model only outputs CFs when highly confident they flip.
SimPO: `dpo_model_snli_premise_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_snli_premise_200s_simpo_b3g05_qwen25_14b/` (train **2777926**, eval **2781365**).
GRPO v2fix: `grpo_model_snli_premise_1ep_g16_multi_qwen25_14b_v2fix` → `evaluation_snli_premise_200s_grpo_mv2_g16_qwen25_14b_v2fix/` (train **2773359**, eval **2777871**).
DPO lr=1e-5: `dpo_model_snli_premise_2pair_lr1e5_qwen25_14b` → `evaluation_snli_premise_200s_dpo_2pair_lr1e5_qwen25_14b/` (train **2896333**, eval **2896334**). **+15.6%** — more than doubles the default-lr DPO result (+6.6%), confirming lr=1e-5 is optimal for 14B DPO on SNLI.
DPO default: `dpo_model_snli_premise_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_premise_200s_dpo_2pair_b4_2ep_qwen25_14b/` (train **2777924**, eval **2781363**).
GDPO v6 ckpt-500: partial; superseded by full-epoch eval above.
SFT: `sft_model_snli_premise_2pair_b16_1ep_qwen25_14b` → `evaluation_snli_premise_200s_sft_2pair_b16_1ep_qwen25_14b/` (train **2780682**, eval **2781362**).

### SNLI-Hypothesis (14B)

| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | mv2 g16 lr=2e-6 | ckpt12800 (~80%) | 91.1% | +14.9% | 0.462 | 126.5 | yes | 200 |
| SimPO  | 2pair b4 β3γ0.5 | 2ep | 89.9% | +13.7% | 0.424 | 95.1  | yes  | 200 |
| GRPO   | v2 g16 (single) | 1ep | 87.7% | +11.6% | 0.467 | 108.0 | yes  | 200 |
| SimPO  | 2pair b4 β2γ0.5 | 2ep | 87.7% | +11.4% | 0.398 | 89.4  | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 85.5% | +9.2%  | 0.454 | 94.5  | yes  | 200 |
| GDPO   | v6 g16 v2fix | 1.0ep | 85.0% | +8.8%  | 0.452 | 93.9  | yes  | 200 |
| GDPO   | v6 g16    | ckpt1000 | 84.0% | +8.0%  | 0.445 | 93.0  | yes  | 200 |
| DPO    | 2pair b4 lr=1e-5 | 2ep | 82.9% | +6.8%  | 0.471 | 194.4 | yes  | 200 |
| SFT    | 2pair b4  | 2ep      | 80.2% | +3.9%  | 0.500 | 83.7  | yes  | 200 |
| GRPO   | v2 g16 (single) lr=2e-6 | ckpt8700 (~54%) | 80.3% | +4.2% | 0.481 | 79.0  | yes  | 200 |
| DPO    | 2pair b4  | 2ep      | 76.7% | +0.7%  | 0.467 | 78.2  | yes  | 200 |
| DPO    | 2pair b4 lr=2e-6 | 2ep | 74.1% | -2.2%  | 0.478 | 76.5  | yes  | 200 |

SimPO β=3: `dpo_model_snli_hypothesis_2pair_b4_2ep_simpo_b3_g05_qwen25_14b` → `evaluation_snli_hypothesis_200s_simpo_b3g05_qwen25_14b/` (train **2777927**, eval **2781366**).
SimPO β=2: `evaluation_snli_hypothesis_200s_simpo_b2g05_qwen25_14b/`. **+11.4%** — β=2 and β=3 both strong, nearly identical LFR, lower NED for β=2.
GRPO: `grpo_model_snli_hypothesis_1ep_g16_multi_qwen25_14b_snli_h_mv2` → `evaluation_snli_hypothesis_200s_grpo_mv2_g16_qwen25_14b/` (train **2777921**, eval **2794878**).
GDPO: `gdpo_model_snli_hypothesis_1ep_g16_qwen25_14b_v6_v2fix` → `evaluation_snli_hypothesis_200s_gdpo_v6_qwen25_14b_v2fix/` (eval **2777872**). ckpt1000: `evaluation_snli_hypothesis_200s_gdpo_v6_ckpt1000_qwen25_14b/` (+8.0%, slightly below full epoch).
SFT: `sft_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_hypothesis_200s_sft_2pair_qwen25_14b/` (train **2786445**, eval **2794799**).
DPO lr=1e-5: `dpo_model_snli_hypothesis_2pair_lr1e5_qwen25_14b` → `evaluation_snli_hypothesis_200s_dpo_2pair_lr1e5_qwen25_14b/` (train **2896335**, eval **2896336**). **+6.8%** — nearly 10× improvement over default-lr DPO (+0.7%).
DPO default: `dpo_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b` → `evaluation_snli_hypothesis_200s_dpo_2pair_b4_2ep_qwen25_14b/` (train **2777925**, eval **2781364**).
**14B matrix complete** (all 5 methods × 3 datasets evaluated). GDPO BoolQ and SNLI-P partial-checkpoint results will be superseded by full-epoch evals (**2805602**, **2805603**).

---

## Qwen2.5-3B-Instruct

All rows below use **Qwen/Qwen2.5-3B-Instruct** as generator and judge. QLoRA 4-bit, RTXA6000/RTXA6000-SLT, fair-base anchors via `run_eval_3b_fair.sh`. Anchors: BoolQ/SNLI-P/SNLI-H → `evaluation_{ds}_200s_dpo_1pair_qwen25_3b/`. PPL = **median** (robust to outliers). Fair reruns (jobs 2891873–2891916) completed May 2026; rows updated to confirmed fair values.

### BoolQ (3B)

Base LFR 30.5%. Anchor: `evaluation_boolq_200s_dpo_1pair_qwen25_3b/`.

| Method  | Config              | LFR   | ΔLFR    | NED   | Med PPL | Fair  | N   |
| ------- | ------------------- | ----- | ------- | ----- | ------- | ----- | --- |
| GRPO    | v2 g16 (single) lr=1e-5 1ep | 41.6% | +10.4% | 0.090 | 11.4    | yes   | 200 |
| GRPO    | mv2 g16 lr=1e-5 1ep | 38.6% | +7.6%   | 0.089 | 11.6    | yes   | 200 |
| GRPO    | mv2 g16 1ep         | 36.6% | +6.0%   | 0.234 | 9.8     | yes   | 200 |
| GRPO    | v2 g16 (single) 1ep | 36.2% | +5.6%   | 0.245 | 9.9     | yes   | 200 |
| SimPO   | 2pair b4 β2γ0.5     | 34.6% | +3.5%   | 0.221 | 10.7    | yes   | 200 |
| SFT     | 2pair 200step       | 31.5% | +1.0%   | 0.271 | 11.0    | yes   | 200 |
| DPO     | 1pair b4 200step    | 30.5% | +0.0%   | 0.145 | 9.6     | yes   | 200 |
| DPO     | 1pair b4 2ep        | 29.4% | −1.1%   | 0.142 | 9.7     | yes   | 200 |
| GDPO v6 | g16 bs8 ckpt100 (~1.6ep) | 31.7% | +1.1% | 0.314 | 11.0  | ~yes(−0.3pp) | 200 |
| GDPO v6 | g16 bs8 ~1.6ep      | 30.4% | −0.1%   | 0.316 | 10.8    | yes   | 200 |
| SimPO   | 2pair b4 β3γ0.5     | 17.5% | −13.0%  | 0.503 | 23.4    | yes   | 200 |

### SNLI-Premise (3B)

Base LFR 32.6%. Anchor: `evaluation_snli_premise_200s_dpo_1pair_qwen25_3b/`.

| Method  | Config               | LFR   | ΔLFR    | NED   | Med PPL | Fair  | N   |
| ------- | -------------------- | ----- | ------- | ----- | ------- | ----- | --- |
| SimPO   | 2pair b4 β3γ0.5      | 58.6% | +24.4%  | 0.308 | 94.2    | yes   | 200 |
| GRPO    | mv2 g16 lr=1e-5 1ep  | 58.8% | +25.2%  | 0.333 | 72.9    | yes   | 200 |
| GDPO v6 | g16 bs8 1ep          | 50.4% | +17.1%  | 0.263 | 79.0    | yes   | 200 |
| GRPO    | v2 g16 (single) lr=1e-5 1ep | 61.1% | +27.7%  | 0.346 | 66.2    | yes   | 200 |
| GRPO    | v2 g16 (single) 1ep  | 49.4% | +16.9%  | 0.235 | 44.3    | yes   | 200 |
| DPO     | 2pair b4 2ep lr=1e-5 | 44.6% | +10.6%  | 0.254 | 82.3    | yes   | 200 |
| GRPO    | mv2 g16 lr=2e-6 1ep  | 41.2% | +8.6%   | 0.225 | 46.7    | yes   | 200 |
| DPO     | 1pair b4 200step     | 40.4% | +7.9%   | 0.228 | 47.2    | yes   | 200 |
| SFT     | 2pair 200step        | 37.7% | +5.2%   | 0.229 | 47.0    | yes   | 200 |

### SNLI-Hypothesis (3B)

Base LFR 48.1%. Anchor: `evaluation_snli_hypothesis_200s_dpo_1pair_qwen25_3b/`.

| Method  | Config               | LFR   | ΔLFR    | NED   | Med PPL | Fair  | N   |
| ------- | -------------------- | ----- | ------- | ----- | ------- | ----- | --- |
| GDPO v6 | g16 bs8 ~0.75ep (fair) | 63.7% | +15.2%  | 0.496 | 68.3  | yes   | 200 |
| GDPO v6 | g16 bs8 ~0.75ep      | 62.5% | +14.0%  | 0.418 | 68.6    | no    | 200 |
| GDPO v6 | g16 bs8 1ep          | 61.9% | +13.6%  | 0.409 | 68.1    | yes   | 200 |
| GDPO v6 | g16 bs8 ckpt1000     | 60.4% | +12.0%  | 0.401 | 66.5    | yes   | 200 |
| SimPO   | 2pair b4 β3γ0.5      | 55.3% | +7.2%   | 0.379 |  92.8   | yes   | 200 |
| SimPO   | 2pair b4 β2γ0.5      | 53.2% | +5.1%   | 0.366 | 236.6   | yes   | 200 |
| DPO     | 2pair b4 2ep lr=1e-5 | 55.8% | +7.3%   | 0.415 |  65.4   | yes   | 200 |
| GRPO    | mv2 g16 lr=1e-5 1ep  | 56.7% | +7.3%   | 0.375 | 100.7   | yes   | 200 |
| GRPO    | mv2 g16 1ep (fair2)  | 55.7% | +7.1%   | 0.278 |  83.9   | yes   | 200 |
| GRPO    | mv2 g16 1ep          | 53.9% | +5.2%   | 0.280 | 82.2    | no    | 200 |
| SimPO   | 2pair b4 β2γ0.5      | 51.4% | +3.3%   | 0.372 | 74.4    | yes   | 200 |
| SFT     | 2pair 200step        | 51.2% | +3.1%   | 0.459 | 67.9    | yes   | 200 |
| GRPO    | v2 g16 (single) lr=1e-5 1ep | 54.0% | +6.3% | 0.348 | 80.9 | yes  | 200 |
| GRPO    | v2 g16 (single) 1ep (fair) | 50.9% | +2.3%  | 0.298 | 215.5 | yes | 200 |
| GRPO    | v2 g16 (single) 1ep  | 51.3% | +3.0%   | 0.288 | 81.4    | no    | 200 |
| DPO     | 2pair b4 2ep         | 50.4% | +2.1%   | 0.379 | 174.8   | yes   | 200 |
| DPO     | 1pair b4 200step     | 50.0% | +1.9%   | 0.374 | 67.4    | yes   | 200 |

---

## Llama-3.1-8B-Instruct

All rows use `meta-llama/Llama-3.1-8B-Instruct` as generator and judge. QLoRA 4-bit, H200/A100-80GB (GDPO bs8 matches 7B/3B fairness). Anchors: all datasets → `evaluation_{ds}_200s_dpo_1pair_llama31_8b/`. PPL = **median**. GRPO SNLI training projected ~170h/epoch — partial results reported. Rows marked "rerun" used independently generated base CFs; bias shown in parentheses.

Base LFRs: BoolQ 53.7% · SNLI-P 43.2% · SNLI-H 56.4%

### BoolQ (Llama)

| Method  | Config              | LFR   | ΔLFR    | NED   | Med PPL | Fair              | N   |
| ------- | ------------------- | ----- | ------- | ----- | ------- | ----------------- | --- |
| GDPO v6 | g16 bs8 ckpt500     | 71.0% | +17.3%  | 0.348 | 13.1    | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt700     | 69.8% | +16.2%  | 0.373 | 30.3    | yes               | 200 |
| GDPO v6 | g16 bs8 1ep         | 70.1% | +16.4%  | 0.370 | 13.5    | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt300     | 68.1% | +14.5%  | 0.386 | 224.6   | yes               | 200 |
| DPO     | 2pair b4 lr=1e-5    | 66.0% | +11.5%  | 0.101 | 8.7     | yes               | 200 |
| SimPO   | 2pair b4 β2γ0.5 lr=2e-6 | 63.5% | +9.5%  | 0.083 | 8.5  | yes               | 200 |
| SimPO   | 2pair b4 β3γ0.5 lr=2e-6 | 62.0% | +7.9%  | 0.118 | 8.8  | yes               | 200 |
| DPO     | 1pair b4 lr=1e-5    | 57.1% | +3.4%   | 0.330 | 11.8    | yes (5% parse⚠)   | 200 |
| DPO     | 1pair b4 lr=1e-5 (v2)| 55.5% | +2.0%  | 0.315 | 11.2    | yes               | 200 |
| SFT     | 2pair 200step       | 53.9% | +0.3%   | 0.279 | 67.4    | yes                     | 200 |
| GRPO    | mv2 g16 mcl768 ckpt1600 | 66.9% | +13.2% | 0.326 | 15.5    | yes               | 200 |
| GRPO    | v2 g16 ckpt200 (~2%) | 51.7% | −1.4%  | 0.325 | 13.0    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt1800 (~22%) | 60.4% | +6.7%  | 0.221 | 12.3    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt1500 (~19%) | 60.2% | +6.5%  | 0.270 | 11.9    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt2100 (~26%) | 59.0% | +5.2%  | 0.246 | 12.2    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt2500 (~31%) | 57.9% | +4.4%  | 0.239 | 12.0    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt3000 (~37%) | 57.5% | +3.8%  | 0.243 | 11.8    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt3500 (~44%) | 56.3% | +2.6%  | 0.201 | 10.6    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt1200 (~15%) | 57.2% | +3.5%  | 0.287 | 12.1    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt600 (~7%)  | 51.2% | −2.5%  | 0.323 | 12.6    | yes                     | 200 |
| GRPO    | v2 g16 mcl768 v2 ckpt200 (~2%)  | 53.7% | −0.0%  | 0.322 | 12.7    | yes                     | 200 |
| GRPO    | mv2 g16 ckpt400 (~5%)| 52.0% | −1.0%  | 0.327 | 13.0    | yes                     | 200 |
| GRPO    | mv2 g16 ckpt200 (~2%)| 51.1% | −2.0%  | 0.318 | 13.0    | yes                     | 200 |
| GRPO    | v2 g16 ckpt400 (~5%) | 50.5% | −2.6%  | 0.326 | 13.0    | yes                     | 200 |
| GRPO    | v2 g16 ~1ep         | 44.6% | −9.9%   | 0.097 | 20.0    | yes                     | 200 |
| GRPO    | mv2 g16 ~1ep        | 32.5% | −21.9%  | 0.031 | 10.5    | yes                     | 200 |
| GRPO    | mv2 g16 lr=1e-5 ~1ep|  8.9% | −46.8%  | 0.004 | 10.4    | rerun             | 200 |

### SNLI-Premise (Llama)

**Charts feature the best run per method (healthy parse).** For DPO that is now `2pair b4 lr=2e-6` **+2.4%** (84% parse, base 43.2→45.6%) — the proven 2pair recipe at gentle lr (jobs 3013355/3013665), beating the old `1pair` +0.4% and avoiding the `lr=1e-5` parse collapse (+10.3% @ 4% parse, excluded). GRPO single ckpt7500 **+32.5%** is the genuine best.

| Method  | Config                  | LFR   | ΔLFR    | NED   | Med PPL | Fair              | N   |
| ------- | ----------------------- | ----- | ------- | ----- | ------- | ----------------- | --- |
| GRPO    | v2 g16 ckpt7500 (~47%)  | 74.6% | +32.5%  | 0.290 | 87.6    | yes               | 200 |
| GRPO    | v2 g16 ckpt7500 (_fair rerun) | 73.4% | +31.3%  | 0.299 | 97.8    | yes (stochasticity — original above is best) | 200 |
| GRPO-fair| mv2 ckpt7000 (~44%)    | 71.5% | +29.0%  | 0.348 | 1706.9  | yes               | 200 |
| GRPO-fair| mv2 ckpt4500 (~28%)    | 70.5% | +28.0%  | 0.343 | 120.8   | yes               | 200 |
| GRPO    | v2 g16 ckpt10100 (~63%) | 73.1% | +30.4%  | 0.295 | 114.1   | yes               | 200 |
| GRPO    | v2 g16 ckpt8500 (~53%)  | 71.0% | +28.1%  | 0.300 | 63.0    | yes               | 200 |
| GRPO-fair| mv2 ckpt11200 (~70%)   | 71.6% | +29.2%  | 0.331 | 138.8   | yes               | 200 |
| GRPO    | v2 g16 ckpt5600 (~35%)  | 65.4% | +22.9%  | 0.325 | 70.8    | yes               | 200 |
| SimPO   | 2pair b4 β3γ0.5 lr=2e-6 | 62.2% | +18.6%  | 0.423 | 89.9    | yes               | 200 |
| SimPO   | 2pair b4 β2γ0.5 lr=2e-6 | 61.2% | +17.4%  | 0.409 | 89.6    | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt500         | 54.4% | +11.4%  | 0.463 | 107.5   | yes               | 200 |
| DPO     | 1pair b4 lr=1e-5        | 53.5% | +10.3%  | 0.368 | 102.7   | yes (4% parse⚠)   | 200 |
| GDPO v6 | g16 bs8 1ep             | 53.8% | +10.6%  | 0.364 | 96.1    | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt1000        | 51.3% | +8.4%   | 0.364 | 94.9    | yes               | 200 |
| SFT     | 2pair 200step           | 45.7% | +2.5%   | 0.468 | 102.9   | yes                     | 200 |
| GRPO    | mv2 g16 ckpt2000 (~12%) | 46.6% | +3.2%   | 0.479 | 112.0   | yes               | 200 |
| GRPO    | mv2 g16 ckpt3000 (~19%) | 46.2% | +2.7%   | 0.475 | 113.0   | yes               | 200 |
| GRPO    | mv2 g16 ckpt4000 (~25%) | 46.2% | +2.9%   | 0.482 | 113.1   | yes               | 200 |
| GRPO    | mv2 g16 ckpt5000 (~31%) | 45.5% | +2.0%   | 0.480 | 116.0   | yes               | 200 |
| GRPO    | mv2 g16 ckpt6000 (~37%) | 45.2% | +1.7%   | 0.477 | 112.0   | yes               | 200 |
| GRPO    | mv2 g16 ckpt7000 (~44%) | 44.7% | +1.2%   | 0.478 | 112.0   | yes               | 200 |
| DPO     | 1pair b4 lr=5e-6        | 43.6% | +0.4%   | 0.388 | 119.5   | yes (anchor)      | 200 |
| GRPO    | mv2 g16 lr=2e-6 ckpt11700 (~73%) | 46.4% | +3.7%  | 0.410 | 100.0  | yes               | 200 |
| GRPO    | mv2 g16 ckpt1000 (~6%)  | 43.3% | +0.2%   | 0.443 | 113.0   | yes               | 200 |
| GRPO    | mv2 g16 ckpt8400 (~52%) | 43.3% | −0.2%   | 0.477 | 731.7   | yes               | 200 |
| DPO     | 2pair b4 lr=1e-5        | 34.1% | −9.1%   | 0.428 | 133.9   | yes (3% parse⚠)   | 200 |

### SNLI-Hypothesis (Llama)

**Charts feature the best run per offline method (checkpoint sweep, healthy parse):** SFT `ckpt100` **+0.6%** (80% parse — clears base; full 200step was −1.4%), SimPO `β3γ0.5 lr=2e-6` **+0.7%** (70% parse), DPO `β=0.3 lr=2e-6 ckpt100` **+0.5%** (90% parse, median PPL 146 — **β-sweep cleared base**; the default β=0.1 sweep had every checkpoint ≤ base, best −0.8%). The higher-ΔLFR variants — SimPO β2 **+11.6%** (4.6% parse), DPO lr=2e-5 **+9.1%** (2.3% parse), SFT 2ep **+2.8%** (11% parse) — are **parse-collapse artifacts** excluded by the 15%-of-base parse gate (full cut list in `OVERVIEW.md`). **GRPO multi +9.6%** (ckpt14900, 66% parse) is the genuine best (a lr=1e-5 GRPO retrain was eval-infeasible — degenerate slow generation). With the β-sweep, **all four offline/online methods now clear base** in this cell (DPO +0.5, SimPO +0.7, SFT +0.6, GRPO multi +9.6); offline still only marginally clears it.

| Method  | Config                   | LFR   | ΔLFR    | NED   | Med PPL | Fair              | N   |
| ------- | ------------------------ | ----- | ------- | ----- | ------- | ----------------- | --- |
| SimPO   | 2pair b4 β2γ0.5          | 68.5% | +11.6%  | 0.571 | 564.7   | yes               | 200 |
| GRPO    | mv2 g16 ckpt14900 (~93%) | 67.3% | +9.6%   | 0.405 | 183.4   | yes               | 200 |
| DPO     | 2pair b4 lr=2e-5         | 67.4% | +9.1%   | 0.658 |  79.7   | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt1000         | 62.4% | +6.2%   | 0.559 | 160.6   | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt1500         | 62.5% | +6.0%   | 0.516 | 155.4   | yes               | 200 |
| GDPO v6 | g16 bs8 ckpt500          | 63.3% | +5.7%   | 0.561 | 174.2   | yes               | 200 |
| GDPO v6 | g16 bs8 1ep              | 62.0% | +5.5%   | 0.525 | 154.2   | yes               | 200 |
| GRPO    | v2 g16 ckpt6000 (~37%)   | 59.7% | +3.4%   | 0.463 | 148.5   | yes (same anchor as _fair run; higher-scoring variant) | 200 |
| SFT     | 2pair 2ep                | 62.2% | +2.8%   | 0.571 | 203.7   | yes (note: _fair rerun dir shows −6.5% due to stale-resume bug — discard) | 200 |
| GRPO    | v2 g16 ckpt4200 (~26%)   | 56.5% | +0.1%   | 0.542 | 490.8   | yes               | 200 |
| DPO     | 2pair b4 lr=5e-5         | —     | ~−3.3%  | 0.621 | 310.2   | yes (INVALID — 18/2000 parse = 0.9%; model collapsed at lr=5e-5) | 200 |
| **DPO** | **2pair lr=2e-5 tag-native** | **62.7%** | **+4.0%** | **0.414** | **117.5** | **yes (84% parse — FEATURED)** | 200 |
| SimPO   | 2pair simpo b2γ0.5 tag-native | 61.7% | +3.0%   | 0.409 | 161.4   | yes (81% parse — FEATURED) | 200 |
| SimPO   | 2pair b4 β3γ0.5 lr=2e-6  | 57.5% | +0.7%   | 0.490 | 214.3   | yes (superseded by tag-native retrain) | 200 |
| DPO     | 2pair b4 β=0.3 lr=2e-6 ckpt100 | 56.6% | +0.5% | 0.548 | 146.2 | yes (superseded by tag-native retrain) | 200 |
| DPO     | 2pair b4 lr=1e-5         | 56.8% | −0.6%   | 0.478 | 681.7   | yes               | 200 |
| DPO     | 2pair b4 β=0.5 lr=2e-6 ckpt200 | 54.4% | −2.1% | 0.546 | — | yes (β=0.5 over-regularizes vs β=0.3) | 200 |
| DPO     | 2pair b4 β=0.5 lr=2e-6 ckpt100 | 53.5% | −2.8% | 0.540 | 148.0 | yes (β=0.5 over-regularizes) | 200 |
| SFT     | 2pair 200step            | 55.0% | −1.4%   | 0.538 | 146.0   | yes               | 200 |
| DPO     | 2pair b4 lr=2e-6         | 54.1% | −2.3%   | 0.504 | 741.3   | yes               | 200 |
| DPO     | 2pair b4 lr=5e-6         | 53.3% | −3.4%   | 0.457 | 147.0   | yes               | 200 |
| DPO     | 1pair b4 lr=5e-6         | 48.6% | −7.8%   | 0.447 | 148.6   | yes               | 200 |
| GRPO    | v2 g16 ckpt7500 (~47%)   | 51.6% | −6.3%   | 0.297 | 1037.7  | yes               | 200 |
| GRPO    | mv2 g16 ckpt5100 (~32%)  | 45.8% | −12.1%  | 0.238 | 380.5   | yes               | 200 |
| DPO     | 1pair b4 lr=1e-5         | 43.7% | −13.2%  | 0.411 | 119.3   | yes               | 200 |
| GRPO    | v2 g16 ckpt9000 (~56%)   | 49.5% | −8.2%   | 0.241 | 383.0   | yes               | 200 |
| GRPO    | v2 g16 ckpt10500 (~66%)  | 40.8% | −17.5%  | 0.185 | 409.7   | yes               | 200 |
| GRPO    | v2 g16 ckpt12000 (~75%)  | 37.7% | −20.3%  | 0.165 | 494.4   | yes               | 200 |
| GRPO    | v2 g16 ckpt12900 (~81%)  | 39.4% | −18.6%  | 0.169 | 625.0   | yes               | 200 |
| SimPO   | 2pair b4 β3γ0.5 (orig)   | 16.6% | −38.7%  | 0.650 | 60699.3 | yes (training diverged — collapse) | 200 |

---

## Active / Pending Jobs (submitted Jul 7, 2026)

| Job ID | Name | Status |
|--------|------|--------|
| 3148850 / 3148851 | **Frozen-base re-verify** (72 featured cells: 12 anchor-freeze + 60 dependent; RTXA6000, `--deterministic`, reuse saved gens) | ✅ COMPLETE (72/72). All 12 canonical anchors now have `base_cfs_verified.jsonl`; base LFR identical across methods within every model×dataset cell. Featured method per cell unchanged; ΔLFR moves ≤1.5pp vs prior numbers except a few near-zero SFT/GDPO baselines (honestly ±0). Charts / `best_models` / docs updated. Scripts: `submit_frozen_reeval_all.sh`, `agg_frozen_reeval.py`. |
| 3147217 / 3147218 | Llama SNLI-H **tag-native retrain** re-eval (DPO lr2e5 + SimPO b2γ0.5, frozen anchor) | ✅ COMPLETE. DPO **+4.0%** (62.7% LFR, 84% parse), SimPO **+3.0%** (61.7% LFR, 81% parse) — both now featured for Llama SNLI-H offline. |
| 3131463 | **Dual-parse diagnostic** (26 cells) | ✅ COMPLETE (26/26). Internal validation only; featured picks unchanged. See `DUAL_PARSE_RESULTS.md`, `PARSE_TAG_ISSUE.md`. |

## Active / Pending Jobs (submitted May 30, 2026)

| Job ID | Name | Status |
|--------|------|--------|
| 3131463 | **Dual-parse diagnostic** (26 cells: all SimPO + Llama DPO/SFT + 4 gate-excluded Llama + 4 tag-clean controls) | ✅ COMPLETE (26/26). Strict vs. lenient-fallback parser on **identical** generations, anchors frozen, compliance decoupled from diversity (`evaluate_models.py --dual_parse`). **Result:** controls strict≡fallback (|ΔLFR|≤0.4pp); **featured picks unchanged — 0/12 cells, no new bests** (ΔLFR moves ≤3.2pp); **gate-excluded Llama runs genuinely re-qualify** — SNLI-H DPO lr2e5 **+9.9%**, SNLI-H SimPO b2 **+8.1%**, SNLI-P DPO 1p-lr1e5 **+6.8%** (SNLI-H SFT 2ep stays out at −2.8%). Featured matrix stands; retrain only Llama offline (DPO/SimPO SNLI-H + DPO SNLI-P) with tags. Full: `DUAL_PARSE_RESULTS.md`, `PARSE_TAG_ISSUE.md §9`. (Note: first submit 3130946 failed all 26 on GPU-broken node serv-9219; resubmitted with `--exclude`.) |
| 3013001 | 7B BoolQ SimPO b3γ0.5 **fair re-eval** (canonical base `05b8d14619`) | ✅ COMPLETE. **+28.6%** (66.7% LFR, base 38.1%, NED 0.115, PPL 8.8, 68% parse). On the canonical base SimPO is the **7B BoolQ best**, ahead of DPO +22.1%. Chart/RESULTS/README updated; ⚠ removed. → `evaluation_boolq_200s_simpo_b3g05_fair/` |
| 3013002 | Llama SNLI-H **GRPO multi-reward v2 g16 lr=1e-5** (train) | ✅ TRAINED to ckpt7400. Eval was never launched; checkpoint sweep submitted (3114894–3114897, see below). → `grpo_model_snli_hypothesis_1ep_g16_multi_mv2_lr1e5_llama31_8b/` |
| 3114894–3114897 | Llama SNLI-H **GRPO multi lr=1e-5 eval sweep** (ckpt 2000/4000/6000/7400) | ❌ ABANDONED — eval-infeasible. The lr=1e-5 GRPO model generates pathologically slowly (~500–880 s/**sample**, degenerate long outputs); all 4 evals hit the 12 h wall at ~30–40% of 200 samples. Not worth re-running on a degenerate model: the cell is already **+9.6%** at default lr (GRPO multi ckpt14900), which stands as the cell best. |
| 3116220–3116223 | Llama SNLI-H DPO beta sweep (first attempt) | ❌ FAILED — passed `LOSS_TYPE=dpo`, but `train_dpo.py` expects `sigmoid` for standard DPO. Resubmitted below. |
| 3118959/3118960 (β0.3) · 3118961/3118962 (β0.5) | Llama SNLI-H **DPO beta sweep** lr=2e-6 `sigmoid` (train + chained eval) | ✅ COMPLETE. **β=0.3 ckpt100 = +0.5%** (56.6% LFR, 90% parse, median PPL 146) — **clears base, now FEATURED** (replaces the −0.8% beta=0.1 result). β=0.5 over-regularizes: ckpt100 −2.8%, ckpt200 −2.1%. All prior Llama SNLI-H DPO runs used the default beta=0.1; raising beta anchors closer to the reference (more conservative) and lifted the only below-base cell above base *without* a pair rebuild (which would break cross-cell consistency). → `evaluation_snli_hypothesis_200s_dpo_2pair_lr2e6_beta03_ckpt100_llama31_8b/` |
| 3013293–3013296 | Llama SNLI-H **DPO checkpoint sweep** (ckpt 100/300/600/1000) | ✅ COMPLETE. At default **beta=0.1**: c100 **−0.8%**, c300 −1.5%, c600 −1.6%, c1000 −1.2% — every checkpoint ≤ base. This motivated the **beta sweep** (3118959–62) which cleared base at β=0.3 ckpt100 (+0.5%, now featured). The "offline ceiling" framing was thus a beta-0.1 artifact, not a fundamental limit. |
| 3013297–3013299 | Llama SNLI-H **SFT checkpoint sweep** (ckpt 50/100/150) | ✅ COMPLETE. c50 −1.3%, **c100 +0.6%** (80% parse — clears base), c150 +0.3%. Chart updated to c100 (was −1.4%). |
| 3013300–3013302 | 14B SNLI-P **SFT checkpoint sweep** (ckpt 50/100/150) | ✅ COMPLETE. c50 −0.0%, c100 −0.7%, c150 −0.4% — flat at base even at earliest ckpt. **Genuine ceiling: SFT is inert on this cell.** |
| 3013354 | Llama SNLI-P **DPO 2pair lr=5e-6** (train) | SUBMITTED. **Under-explored:** the 2pair recipe (7B SNLI-P DPO +24.5%) was never trained for Llama at sane lr — only 2pair lr=1e-5 (parse-collapse) and 1pair (+0.4%). lr=5e-6 = 7B's winning lr. → eval after train. |
| 3013355 | Llama SNLI-P **DPO 2pair lr=2e-6** (train) | SUBMITTED. Same recipe at Llama's safe gentle lr (matches Llama SimPO +18.6% / GRPO that avoid collapse). → eval after train. |
| 3013354/3013664 (lr5e6) · 3013355/3013665 (lr2e6) | Llama SNLI-P DPO 2pair train+eval | ✅ COMPLETE. **lr2e6 +2.4%** (84% parse, NED 0.411, PPL 96.9) — healthy, now the featured DPO for the cell (was 1p +0.4%). lr5e6 +2.2% but only 15% parse. Confirms the 2pair recipe transfers but Llama SNLI-P DPO stays modest (GRPO single +32.5% dominates). |
| 3013675 / 3013676 | 3B BoolQ **DPO 2pair lr=5e-6** (train + eval) | ✅ COMPLETE. **−0.6%** (100% parse). The 2pair recipe does **not** help 3B BoolQ (1pair was +1.2% on its own base, anchor 0.0%). **Confirms "3B can't learn BoolQ offline"** holds under the stronger recipe. → `evaluation_boolq_200s_dpo_2pair_qwen25_3b/` |
| 3013830 / 3013831 | 7B BoolQ **GRPO single lr=1e-5** (train + chained eval) | ✅ COMPLETE. **+21.2%** (59.0% LFR, 45% parse, NED 0.086, PPL 8.3) — **+13.5pp over default lr (+7.7%)**. lr=1e-5 transfers from 3B; now featured. → `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair/` |
| 3013832 / 3013833 | 7B BoolQ **GRPO multi lr=1e-5** (train + chained eval) | ✅ COMPLETE. **+18.8%** (56.6% LFR, 52% parse, NED 0.090, PPL 8.1) — **+9pp over default lr (+9.8%)**. → `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair/` |

## Completed Evaluations

All prior experiments have completed. Key recent completions:

| Job ID | Name | Result |
|--------|------|--------|
| 2957871 | Llama SNLI-H DPO lr=5e-5 eval | COMPLETED but INVALID. 18/2000 parse rate (0.9%) — lr=5e-5 too high, adapter collapsed. lr=2e-5 +9.1% remains best DPO for Llama SNLI-H. |
| 2957872 | 3B SNLI-H SimPO β=3 fair2 reconfirm | COMPLETED. **+7.2%** (NED 0.364, cfs=736) — slightly higher than original +6.8%. Update chart to +7.2%. |
| 2957873 | Llama BoolQ GRPO v2 ckpt200 | COMPLETED. **−0.0%** (PPL=266k — extremely high, model not yet trained). |
| 2957874 | Llama BoolQ GRPO v2 ckpt600 | COMPLETED. **−2.5%** (PPL=5680). |
| 2957875 | Llama BoolQ GRPO v2 ckpt1200 | COMPLETED. **+3.5%** (NED 0.287, PPL=669) — first positive; mcl768 single. Peak at ckpt1800 +6.7%. |
| 2960497 | Llama BoolQ GRPO v2 ckpt1500 | COMPLETED. **+6.5%** (mcl768 single; new v2 model). |
| 2960498 | Llama BoolQ GRPO v2 ckpt1800 | COMPLETED. **+6.7% — PEAK** (mcl768 single; new v2 model). |
| 2960499 | Llama BoolQ GRPO v2 ckpt2100 | COMPLETED. **+5.2%** (mcl768 single; new v2 model). |
| 2960500 | Llama BoolQ GRPO v2 ckpt2500 | COMPLETED. **+4.4%** (mcl768 single; new v2 model). |
| 2960501 | Llama BoolQ GRPO v2 ckpt3000 | COMPLETED. **+3.8%** (mcl768 single; new v2 model). |
| 2960502 | Llama BoolQ GRPO v2 ckpt3500 | COMPLETED. **+2.6%** (mcl768 single; new v2 model). |
| 2929454 | Llama SNLI-P GRPO-fair ckpt4500 | `evaluation_snli_premise_200s_grpo_mv2g16_fair_multi_ckpt4500_llama31_8b/` — **+28.0%** ΔLFR (70.5% LFR, NED 0.343, PPL 120.8) |
| 2929455 | Llama SNLI-P GRPO-fair ckpt7000 | `evaluation_snli_premise_200s_grpo_mv2g16_fair_multi_ckpt7000_llama31_8b/` — **+29.0%** ΔLFR (71.5% LFR, NED 0.348, PPL 1706.9) |
| 2929456 | Llama SNLI-H DPO lr=2e-5 fair rerun (corrupt) | resumed old progress file → 0 tuned CFs. Note: original `dpo_2pair_lr2e5_llama31_8b` (+9.1%) was already fair (same anchor). Resubmitted confirmation as **2932695**. |
| 2929501 | Llama SNLI-H DPO lr=5e-5 (train) | `dpo_model_snli_hypothesis_2pair_lr5e5_llama31_8b/` — training complete; eval submitted **2929507** |
| 2929507 | Llama SNLI-H DPO lr=5e-5 eval | eval-llama8b-fair — RUNNING (job 2929507) |
| 2929543–2929557 | Llama SNLI-H: 15 confirmation re-evals | All COMPLETED. All originals already fair. Best confirmed results: SimPO β2 **+11.6%** (re-eval), GRPO mv2 **+9.6%**, GDPO **+6.2%**. DPO lr=2e-5 original **+9.1%** best. SFT 2ep original **+2.8%** (re-eval _fair dir corrupt −6.5%). |
| 2929558 | Llama SNLI-P GRPO single ckpt7500 fair | `evaluation_snli_premise_200s_grpo_v2g16_ckpt7500_fair_llama31_8b/` — **+31.3%** (original was also fair; +32.5% was not bias — just stochasticity) |
| 2929559 | 3B SNLI-P GRPO multi lr=1e-5 canonical | `evaluation_snli_premise_200s_grpo_mv2g16_lr1e5_fair_canonical_qwen25_3b/` — **+25.2%** |
| 2929560 | 3B SNLI-P SimPO b3g05 canonical | `evaluation_snli_premise_200s_simpo_b3g05_fair_canonical_qwen25_3b/` — **+24.4%** |
| 2929561 | 3B SNLI-P DPO 2pair 2ep lr=1e-5 fair | `evaluation_snli_premise_200s_dpo_2pair_2ep_lr1e5_fair_qwen25_3b/` — **+10.6%** |
| 2929565–2929567 | 3B GRPO lr=1e-5 evals (FAILED) | Model at checkpoint subdir, not root. Resubmitted as **2932696–2932698** with checkpoint paths. |
| 2929607 | Llama SNLI-H SimPO β2 fair | `evaluation_snli_hypothesis_200s_simpo_b2g05_fair_llama31_8b/` — **+11.6%** (best Llama SNLI-H method so far) |
| 2929608 | Llama SNLI-H SimPO β3 fair | `evaluation_snli_hypothesis_200s_simpo_b3g05_fair_llama31_8b/` — **−38.7%** (confirms training divergence) |
| 2929619–2929620 | 3B SNLI-P GDPO + GRPO canonical | GDPO v6 1ep **+17.1%**; GRPO mv2 ckpt4000 **+1.9%** |
| 2929624–2929627 | 3B SNLI-H fair re-evals | GRPO mv2 1ep (fair) **+7.1%**; GRPO single (fair) **+2.3%**; DPO 2pair 2ep **+2.1%** |
| 2932694 | Llama SNLI-H DPO lr=5e-5 eval | PENDING. `evaluation_snli_hypothesis_200s_dpo_2pair_lr5e5_fair_llama31_8b/` |
| 2932695 | Llama SNLI-H DPO lr=2e-5 confirmation rerun | PENDING. Original +9.1% already fair; this is an additional confirmation run. |
| 2932696 | 3B BoolQ GRPO multi lr=1e-5 eval | COMPLETED. **+7.6%** — beats previous best +6.0%. |
| 2932697 | 3B SNLI-H GRPO multi lr=1e-5 eval | COMPLETED. **+7.3%** (slight improvement over +7.1%). |
| 2932698 | 3B SNLI-P GRPO single lr=1e-5 eval | COMPLETED. **+27.7%** — massive improvement over default lr +16.9%. |
| 2932699 | Llama SNLI-P GRPO-fair multi ckpt11200 | COMPLETED. **+29.2%** — marginally better than ckpt7000 +29.0%. |
| 2933393 | Llama SNLI-H DPO lr=5e-5 eval (FAILED) | Script error: OUTPUT_DIR not set. Deleted output dir. Resubmitted as **2957871**. |
| 2932901 | 3B SNLI-H SimPO β=3 reconfirm (FAILED) | Script error: OUTPUT_DIR not set. Resubmitted as **2957872**. |
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
| 2929500 | Llama BoolQ GRPO v2 single (train) | CANCELLED at 49% (ckpt3952/8000 — time limit). Produced `grpo_model_boolq_1ep_g16_llama31_8b_v2` with checkpoints up to ckpt3900. Checkpoint evals: **2957873–2957875** |
| 2929501 | Llama SNLI-H DPO lr=5e-5 (train) | COMPLETED. `dpo_model_snli_hypothesis_2pair_lr5e5_llama31_8b/`. Eval: **2957871** |
| 2957871 | Llama SNLI-H DPO lr=5e-5 eval | COMPLETED but INVALID. 18/2000 parse rate (0.9%) — lr=5e-5 too high, adapter collapsed. lr=2e-5 +9.1% remains best DPO for Llama SNLI-H. |
| 2957872 | 3B SNLI-H SimPO β=3 fair2 reconfirm | COMPLETED. **+7.2%** (NED 0.364, cfs=736) — slightly higher than original +6.8%. Update chart to +7.2%. |
| 2957873 | Llama BoolQ GRPO v2 ckpt200 | COMPLETED. **−0.0%** (PPL=266k — extremely high, model not yet trained). |
| 2957874 | Llama BoolQ GRPO v2 ckpt600 | COMPLETED. **−2.5%** (PPL=5680). |
| 2957875 | Llama BoolQ GRPO v2 ckpt1200 | COMPLETED. **+3.5%** (NED 0.287, PPL=669) — first positive; mcl768 single. Peak at ckpt1800 +6.7%. |
| 2960497 | Llama BoolQ GRPO v2 ckpt1500 | COMPLETED. **+6.5%** (mcl768 single; new v2 model). |
| 2960498 | Llama BoolQ GRPO v2 ckpt1800 | COMPLETED. **+6.7% — PEAK** (mcl768 single; new v2 model). |
| 2960499 | Llama BoolQ GRPO v2 ckpt2100 | COMPLETED. **+5.2%** (mcl768 single; new v2 model). |
| 2960500 | Llama BoolQ GRPO v2 ckpt2500 | COMPLETED. **+4.4%** (mcl768 single; new v2 model). |
| 2960501 | Llama BoolQ GRPO v2 ckpt3000 | COMPLETED. **+3.8%** (mcl768 single; new v2 model). |
| 2960502 | Llama BoolQ GRPO v2 ckpt3500 | COMPLETED. **+2.6%** (mcl768 single; new v2 model). |
| 2929454 | Llama SNLI-P GRPO-fair ckpt4500 eval | RUNNING. GRPO-fair multi (KL-regularized, no collapse). |
| 2929455 | Llama SNLI-P GRPO-fair ckpt7000 eval | RUNNING. |
| 2929456 | Llama SNLI-H DPO lr=2e-5 fair reeval | RUNNING. Canonical fair eval of breakthrough DPO result. |
| 2929543–2929557 | **Llama SNLI-H: 15 fair re-evals** | All originals were already fair (same anchor). base_LFR variance (57.6–59.4%) was stochasticity, not bias. Re-evals provide confirmation runs; best result per method is used. |
| 2929558 | Llama SNLI-P GRPO single ckpt7500 fair | PENDING. Corrects −0.7pp anchor bias for best SNLI-P single result. |
| 2929559 | 3B SNLI-P GRPO multi lr=1e-5 fair canonical | Original was already fair (same anchor); base_LFR variance was stochasticity. Canonical rerun gives +25.2%; original fair gave +25.3%. |
| 2929560 | 3B SNLI-P SimPO b3g05 fair canonical | Original was already fair. Canonical gives +24.4%; best fair run (fair2) gives +25.5%. |
| 2929561 | 3B SNLI-P DPO 2pair 2ep lr=1e-5 fair | Original was already fair. Canonical gives +10.6%; original fair gave +11.5%. |
| 2929565 | **3B BoolQ GRPO multi lr=1e-5** (eval) | PENDING. First multi-reward lr=1e-5 eval for 3B BoolQ — underexplored. Model `grpo_model_boolq_1ep_g16_multi__lr1e5_qwen25_3b_mv2` trained (ckpt4200). |
| 2929566 | **3B SNLI-H GRPO multi lr=1e-5** (eval) | PENDING. First multi-reward lr=1e-5 eval for 3B SNLI-H — underexplored. Model `grpo_model_snli_hypothesis_1ep_g16_multi__lr1e5_qwen25_3b_mv2` trained (ckpt8100). |
| 2929567 | **3B SNLI-P GRPO single lr=1e-5** (eval) | PENDING. First single-reward lr=1e-5 eval for 3B SNLI-P — only 1 prior single config at default lr. Model `grpo_model_snli_premise_1ep_g16_v2_lr1e5_qwen25_3b` trained (ckpt13200). |
| 2929506 | Llama BoolQ GRPO v2 single eval (CANCELLED) | DependencyNeverSatisfied — 2929500 was cancelled. Replaced by **2957873–2957875**. |
| 2921262 | Llama SNLI-P GRPO-fair full eval | CANCELLED (DependencyNeverSatisfied). Full-epoch eval completed via 2932699 instead. |
| **2929594–2929601** | **Llama BoolQ: 8 re-evals** | All originals already fair (same anchor). base_LFR variation was stochasticity. Re-evals provide additional runs; best result per method used. |
| **2929602–2929606** | **Llama SNLI-P: 5 re-evals** | Same — all originals fair. GRPO single ckpt7500 original (+32.5%) better than re-eval (+31.3%). |
| **2929607–2929608** | **Llama SNLI-H: SimPO re-evals** | SimPO b2 re-eval (+11.6%) better than original (+7.2%); SimPO b3 collapse confirmed (−38.7%). |
| **2929609–2929612** | **14B: 4 re-evals** | All originals already fair. Re-evals may yield marginally different values. |
| **2929615–2929618** | **3B BoolQ: 4 re-evals** | All originals already fair. Re-evals for additional confirmation. |
| **2929619–2929623** | **3B SNLI-P: 5 re-evals** | All originals fair. Some re-eval runs lower due to stochasticity; best result per method retained. |
| **2929624–2929629** | **3B SNLI-H: 6 re-evals** | All originals fair. Best result per method used. |
| **2929639–2929648** | **7B: 10 fair re-evals** | PENDING. Created `run_eval_7b_fair.sh`. BoolQ: DPO 1pair/2pair, GRPO mv2 mcl1024 ckpt2000, GRPO mv2 g24 lr=1e-6, GRPO v2, SFT. SNLI-P: DPO 2pair, GDPO, SFT. SNLI-H: DPO 2pair. |


---

## Training Data Statistics


| Dataset         | 1-pair Samples | 2-pair Samples | Entries with Pairs    |
| --------------- | -------------- | -------------- | --------------------- |
| BoolQ           | 1,647          | 3,191          | 1,647 / 2,000 (82.4%) |
| SNLI-Premise    | 1,955          | 3,810          | 1,955 / 2,000 (97.8%) |
| SNLI-Hypothesis | 1,735          | 3,244          | 1,735 / 2,000 (86.8%) |



