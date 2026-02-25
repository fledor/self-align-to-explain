# Evaluation Results

All evaluation results for counterfactual generation models trained on Qwen/Qwen2.5-7B-Instruct.

**Metrics**:

- **LFR** (Label Flip Rate): % of counterfactuals where the base model's prediction changed from the original label (higher = better)
- **ΔLFR**: LFR improvement over the base model on the same evaluation set
- **NED** (Normalized Edit Distance): edit distance / max text length (lower = more minimal edits)
- **PPL** (Perplexity): fluency of generated text (lower = more fluent)
- **Fair**: whether base model counterfactuals were reused across evaluations (`--base_eval_dir`) for consistent comparison
- **N**: number of validation samples evaluated (10 CFs generated per sample)

**Methods**: DPO (preference optimization), SFT (supervised fine-tuning on chosen CFs), GRPO (online RL with reward signal)

**Configs**: `{pairs} {batch}` where pairs = number of preference pairs per entry, batch = effective batch size (b4 = bs1×ga4, b16 = bs4×ga4). GRPO configs: `single` = combined reward, `multi` = decomposed rewards, `g4`/`g16` = generations per prompt.

**GRPO reward versions**: GRPO g4 and g16 (non-v2) used an incorrect single reward: `flip + 0.8 * similarity` (no confidence). GRPO g16 v2 uses the corrected reward: `flip + confidence * similarity`, matching the DPO unified score. Multi-reward runs are unaffected (they use separate FlipReward + SimilarityReward). Results marked with † use the old reward.

**Duration**: `200step` = fixed 200 gradient steps (variable effective epochs), `0.5ep`–`3ep` = epoch-controlled training.

---

## BoolQ


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL  | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ---- | ---- | --- |
| DPO    | 2pair b4  | 2ep      | 51.8% | +14.7% | 0.144 | 13.1 | no   | 200 |
| DPO    | 2pair b4  | 200step  | 57.6% | +13.9% | 0.130 | 10.2 | no   | 50  |
| DPO    | 1pair b4  | 2ep      | 48.0% | +10.5% | 0.110 | 23.9 | yes  | 200 |
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


## SNLI-Premise


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| DPO    | 2pair b4  | 2ep      | 75.0% | +22.0% | 0.327 | 142.3 | no   | 200 |
| DPO    | 2pair b4  | 2ep      | 71.5% | +17.3% | 0.326 | —     | yes  | 100 |
| DPO    | 2pair b16 | 2ep      | 70.5% | +17.2% | 0.321 | 83.6  | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 68.1% | +16.6% | 0.296 | 89.8  | yes  | 100 |
| DPO    | 2pair b16 | 3ep      | 69.9% | +16.5% | 0.323 | 79.6  | yes  | 100 |
| DPO    | 1pair b4  | 2ep      | 68.0% | +15.7% | 0.301 | 133.3 | yes  | 200 |
| DPO    | 2pair b4  | 200step  | 64.2% | +12.7% | 0.275 | 93.8  | yes  | 100 |
| DPO    | 1pair b4  | 2ep      | 65.7% | +12.2% | 0.310 | 94.8  | yes  | 100 |
| DPO    | 2pair b16 | 200step  | 65.5% | +12.2% | 0.311 | 90.1  | yes  | 100 |
| DPO    | 1pair b16 | 2ep      | 63.8% | +10.5% | 0.303 | 85.9  | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 61.2% | +9.7%  | 0.276 | 94.8  | no   | 100 |
| DPO    | 1pair b16 | 3ep      | 63.0% | +9.3%  | 0.297 | 94.1  | yes  | 100 |
| DPO    | 1pair b4  | 200step  | 60.2% | +7.9%  | 0.257 | 75.0  | no   | 50  |
| GRPO   | multi g4  | 2ep      | 60.2% | +7.2%  | 0.275 | 85.1  | yes  | 100 |
| GRPO † | single g4 | 2ep      | 58.7% | +5.4%  | 0.266 | 92.4  | yes  | 100 |
| DPO    | 2pair b4  | 200step  | 56.4% | +4.6%  | 0.244 | 69.8  | no   | 50  |
| DPO    | 1pair b4  | 200step  | 57.6% | +4.5%  | 0.274 | 96.5  | yes  | 100 |
| SFT    | 1pair b4  | 200step  | 54.5% | +2.7%  | 0.248 | 109.9 | yes  | 100 |
| SFT    | 2pair b16 | 1ep      | 55.2% | +1.8%  | 0.249 | 105.9 | yes  | 100 |
| SFT    | 2pair b4  | 200step  | 53.3% | +1.5%  | 0.243 | 110.6 | yes  | 100 |
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


## SNLI-Hypothesis


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| DPO    | 2pair b4  | 2ep      | 65.9% | +21.0% | 0.378 | 296.5 | yes  | 200 |
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
| SFT    | 2pair b16 | 1ep      | 43.9% | -3.5%  | 0.330 | 208.3 | yes  | 100 |
| SFT    | 2pair b4  | 1ep      | 43.3% | -3.5%  | 0.347 | 272.5 | yes  | 100 |
| SFT    | 1pair b4  | 2ep      | 42.5% | -4.2%  | 0.346 | 233.3 | yes  | 100 |
| SFT    | 1pair b16 | 200step  | 42.4% | -4.6%  | 0.325 | 254.0 | yes  | 100 |
| SFT    | 2pair b16 | 200step  | 42.3% | -5.1%  | 0.329 | 194.3 | yes  | 100 |
| SFT    | 1pair b16 | 2ep      | 41.9% | -5.1%  | 0.332 | 218.5 | yes  | 100 |
| SFT    | 2pair b16 | 0.5ep    | 41.9% | -5.1%  | 0.326 | 385.0 | yes  | 100 |
| SFT    | 1pair b4  | 0.5ep    | 41.7% | -5.3%  | 0.328 | 271.4 | yes  | 100 |
| SFT    | 1pair b4  | 1ep      | 40.7% | -6.1%  | 0.343 | 211.0 | yes  | 100 |
| GRPO † | single g4 | 2ep      | 36.3% | -10.6% | 0.167 | 227.9 | yes  | 100 |
| GRPO   | multi g4  | 2ep      | 35.6% | -11.3% | 0.151 | 359.5 | yes  | 100 |


---

## Pending Evaluations


| Model               | Dataset | Status                     |
| ------------------- | ------- | -------------------------- |
| DPO 2pair b4 2ep    | SNLI-H  | eval submitted (fair base) |
| DPO 2pair b16 2ep   | SNLI-H  | eval submitted (fair base) |
| GRPO single g4      | BoolQ   | training ~80% done         |
| GRPO multi g4       | BoolQ   | training ~80% done         |
| GRPO † single g16   | all     | training in progress (old reward)  |
| GRPO multi g16      | all     | training in progress       |
| GRPO single g16 v2  | all     | training queued (corrected reward) |
| SFT 1pair b16 0.5ep | BoolQ   | eval running               |
| SFT 2pair b4 0.5ep  | BoolQ   | eval running               |
| SFT 2pair b16 0.5ep | BoolQ   | eval running               |


---

## Training Data Statistics


| Dataset         | 1-pair Samples | 2-pair Samples | Entries with Pairs    |
| --------------- | -------------- | -------------- | --------------------- |
| BoolQ           | 1,647          | 3,191          | 1,647 / 2,000 (82.4%) |
| SNLI-Premise    | 1,955          | 3,810          | 1,955 / 2,000 (97.8%) |
| SNLI-Hypothesis | 1,735          | 3,244          | 1,735 / 2,000 (86.8%) |


## Training Configuration

All models use QLoRA (4-bit quantization), LoRA r=32, alpha=16, lr=5e-6 on Qwen/Qwen2.5-7B-Instruct.

- **b4**: batch size 1, gradient accumulation 4 (effective batch 4)
- **b16**: batch size 4, gradient accumulation 4 (effective batch 16, requires A100-80GB)
- **200step**: `max_steps=200` (effective epochs vary: 1pair b4 ~0.4-0.5ep, 1pair b16 ~1.6-1.9ep, 2pair b4 ~0.2-0.3ep, 2pair b16 ~0.8-1.0ep)
- **DPO pairs**: chosen = label-flipping CFs, rejected = non-flipping CFs; 2pair = best+worst and 2nd-best+2nd-worst; 1pair = best+worst only
- **SFT**: trains on chosen examples only (no rejected/contrast data)
- **GRPO g4/g16**: online RL, 4 or 16 generations per prompt, temperature=1.2, combined or decomposed reward
- **GRPO † (old reward)**: single reward = `flip + 0.8 * similarity` (no confidence weighting)
- **GRPO v2 (corrected reward)**: single reward = `flip + confidence * similarity` (matches DPO unified score)

