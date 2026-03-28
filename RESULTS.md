# Evaluation Results

All evaluation results for counterfactual generation models trained on Qwen/Qwen2.5-7B-Instruct.

**Metrics**:

- **LFR** (Label Flip Rate): % of counterfactuals where the base model's prediction changed from the original label (higher = better)
- **ΔLFR**: LFR improvement over the base model on the same evaluation set
- **NED** (Normalized Edit Distance): edit distance / max text length (lower = more minimal edits)
- **PPL** (Perplexity): fluency of generated text (lower = more fluent)
- **LFR/NED**: label flip rate / normalized edit distance — edit efficiency (higher = more effective per unit of change). Reported in individual eval reports for N=200 runs.
- **Fair**: whether base model counterfactuals were reused across evaluations (`--base_eval_dir`) for consistent comparison
- **N**: number of validation samples evaluated (10 CFs generated per sample)

**Methods**: DPO (preference optimization), SFT (supervised fine-tuning on chosen CFs), GRPO (online RL with reward signal), GDPO (GRPO with per-reward normalization)

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
| GRPO   | mv2 g16 mcl768 | ~0.25ep | 38.2% | +1.7% | 0.159 | 12.0 | yes  | 200 |
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


## SNLI-Premise


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | mv2 g16   | 1.0ep    | 79.7% | +27.5% | 0.348 | 72.7  | yes  | 200 |
| GRPO   | paper 4GPU| 1.0ep    | 78.3% | +25.9% | 0.348 | 103.2 | yes  | 200 |
| GRPO   | v2 g16    | ~0.50ep  | 78.0% | +25.8% | 0.345 | 110.6 | yes  | 200 |
| GRPO   | mv2 g24   | 1.0ep    | 74.2% | +22.2% | 0.387 | 146.6 | yes  | 200 |
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
| GDPO v7| 4GPU lr5e6    | 1.0ep  | 67.9% | +15.7% | 0.285 | 117.3 | yes  | 200 |
| GDPO   | paper 4GPU    | 1.0ep  | 59.4% | +7.6%  | 0.258 | 121.6 | yes  | 200 |
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


## SNLI-Hypothesis


| Method | Config    | Duration | LFR   | ΔLFR   | NED   | PPL   | Fair | N   |
| ------ | --------- | -------- | ----- | ------ | ----- | ----- | ---- | --- |
| GRPO   | v2 g24    | 1.0ep    | 67.4% | +22.6% | 0.302 | 9651  | yes  | 200 |
| DPO    | 2pair b4  | 2ep      | 65.9% | +21.0% | 0.378 | 296.5 | yes  | 200 |
| GRPO   | mv2 g16   | 1.0ep    | 59.4% | +15.1% | 0.266 | 702.9 | yes  | 200 |
| GRPO   | mv2 g24   | 1.0ep    | 57.4% | +12.6% | 0.253 | 453.2 | yes  | 200 |
| GRPO   | v2 g16    | 1.0ep    | 55.7% | +11.0% | 0.269 | 317.3 | yes  | 200 |
| DPO    | 1pair b4  | 2ep      | 53.4% | +8.7%  | 0.334 | 368.0 | no   | 200 |
| SFT    | 2pair b4  | 2ep      | 46.1% | +1.7%  | 0.329 | 286.8 | yes  | 200 |
| GDPO   | mv2 g24 lr1e6 | 1.0ep | 45.6% | +1.1%  | 0.302 | 247.6 | yes  | 200 |
| GDPO v2| mv2 g16 cond  | 1.0ep | 45.0% | +0.7%  | 0.316 | 329.0 | yes  | 200 |
| GDPO v3| mv2 g16 ned02 | ~0.95ep | 46.3% | +1.8% | 0.276 | 308.3 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.25ep| 60.3% | +15.9% | 0.365 | 248.9 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.50ep| 62.0% | +17.6% | 0.363 | 329.6 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | ~0.75ep| 63.1% | +18.7% | 0.361 | 334.9 | yes  | 200 |
| GDPO v6| mv2 g16 lr5e6 | 1.0ep  | 61.8% | +17.6% | 0.356 | 281.1 | yes  | 200 |
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


---

## Completed Evaluations

All prior experiments have completed. Key recent completions:

| Job ID | Name | Result |
|--------|------|--------|
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
- **GRPO g4/g16**: online RL, 4 or 16 generations per prompt, temperature=1.2, combined or decomposed reward. g4 trained for 2ep; g16 trains for 1ep (see [GRPO_ANALYSIS.md](GRPO_ANALYSIS.md) for epoch analysis)
- **GRPO † (old reward)**: single reward = `flip + 0.8 * similarity` (no confidence weighting)
- **GRPO v2 (corrected reward)**: single reward = `flip + confidence * similarity` (matches DPO unified score)
- **GRPO mv2 (multi-reward v2)**: 4 decomposed rewards: FlipReward + SimilarityReward + GatedConfidenceReward + FormatReward
- **GRPO lr1e6**: same reward as parent config but trained at lr=1e-6 (vs default 5e-6) to prevent BoolQ instability
- **GDPO**: GRPO with per-reward normalization (normalize each reward independently per group, then sum). Uses mv2 reward functions. See `gdpo_trainer.py`
- **g24**: experiments with 24 generations per prompt (vs 16) for better group diversity and reward signal. SNLI g24 requires ~24000 optimizer steps (2x BoolQ due to 2 prompts/entry), exceeding the 24h SLURM limit; resumed from checkpoints
- **GDPO lr1e6**: GDPO with lr=1e-6 (vs default 5e-6) and FormatReward weight 3x to prevent entropy collapse
- **GDPO v2 (conditioned)**: GDPO with conditioned rewards (similarity gated on flip), beta=0.001 KL penalty, dynamic sampling (zero-variance group filtering), lr=1e-6. See [GDPO_BOOLQ_IMPROVEMENTS.md](GDPO_BOOLQ_IMPROVEMENTS.md)
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

