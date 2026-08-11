# Best Configurations

The exact winning configuration for every featured **model × dataset × method** cell
(72 in total) — the runs behind the results tables in `README.md` and the thesis.
Metrics come from the frozen fair-base evaluation (`frozen_metrics.json`,
N=200 prompts × 10 counterfactuals, shared frozen base verdicts per model×dataset).

**Fixed settings shared by all runs** (only the settings listed per cell vary):
QLoRA with 4-bit NF4 double quantization and bf16 compute; LoRA r=32, α=16,
dropout 0.05 on all attention + MLP projections; AdamW; cosine schedule with
warmup ratio 0.1; generation at temperature 1.2, top-p 0.99, top-k 100.
Offline methods train on pairs built from ~40 base-model counterfactuals for each
of 2,000 training entries (`generate_counterfactuals.py` → `evaluate_counterfactuals.py`
→ `construct_dpo_pairs.py`, run per base model); online methods generate during
training with group size G=16.

**Early stopping.** Where the featured run is an intermediate checkpoint, the
`checkpoint-N` is stated with the command; the adapter released for that cell *is*
that checkpoint. All other cells use the final checkpoint of the run.

**Evaluation.** Each adapter was evaluated with:

```bash
python evaluate_models.py \
    --model_path <adapter_dir> --datasets <dataset> --split validation \
    --num_samples 200 --cfs_per_entry 10 \
    --base_eval_dir <frozen base eval dir for this model×dataset> --resume
```

The released adapters (one per row) are on Hugging Face — see `README.md`.

## Summary

| Model | Dataset | Method | LR | Tuned settings | Stop | ΔLFR (pp) | Parse% | Adapter |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | BoolQ | SFT | 5e-6 | max_steps=200 | step 200 | +1.2 | 92.8 | `qwen3b_boolq_sft` |
| Qwen2.5-3B-Instruct | BoolQ | DPO | 5e-6 | β=0.1, 1 pair/entry | 2 ep | +1.7 | 87.4 | `qwen3b_boolq_dpo` |
| Qwen2.5-3B-Instruct | BoolQ | SimPO | 5e-6 | β=2.0, γ=0.5, 2 pair/entry | 2 ep | +7.5 | 74.9 | `qwen3b_boolq_simpo` |
| Qwen2.5-3B-Instruct | BoolQ | GRPO (composite reward) | 1e-5 | composite reward | step 4300 | +11.0 | 55.4 | `qwen3b_boolq_grpo_single` |
| Qwen2.5-3B-Instruct | BoolQ | GRPO (decomposed reward) | 1e-5 | decomposed reward | step 4200 | +8.2 | 54.0 | `qwen3b_boolq_grpo_multi` |
| Qwen2.5-3B-Instruct | BoolQ | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 100 | +1.3 | 90.3 | `qwen3b_boolq_gdpo` |
| Qwen2.5-3B-Instruct | SNLI-Premise | SFT | 5e-6 | max_steps=200 | step 200 | +4.4 | 46.9 | `qwen3b_snli_premise_sft` |
| Qwen2.5-3B-Instruct | SNLI-Premise | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +11.0 | 50.8 | `qwen3b_snli_premise_dpo` |
| Qwen2.5-3B-Instruct | SNLI-Premise | SimPO | 5e-6 | β=0.3, γ=0.5, 2 pair/entry | 2 ep | +25.8 | 41.1 | `qwen3b_snli_premise_simpo` |
| Qwen2.5-3B-Instruct | SNLI-Premise | GRPO (composite reward) | 1e-5 | composite reward | step 13200 | +27.1 | 34.3 | `qwen3b_snli_premise_grpo_single` |
| Qwen2.5-3B-Instruct | SNLI-Premise | GRPO (decomposed reward) | 1e-5 | decomposed reward | 1 ep | +25.6 | 33.2 | `qwen3b_snli_premise_grpo_multi` |
| Qwen2.5-3B-Instruct | SNLI-Premise | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | 1 ep | +17.7 | 46.5 | `qwen3b_snli_premise_gdpo` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | SFT | 5e-6 | max_steps=200 | step 200 | +3.1 | 46.9 | `qwen3b_snli_hypothesis_sft` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +7.3 | 39.4 | `qwen3b_snli_hypothesis_dpo` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | SimPO | 5e-6 | β=0.3, γ=0.5, 2 pair/entry | 2 ep | +6.7 | 36.8 | `qwen3b_snli_hypothesis_simpo` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | GRPO (composite reward) | 1e-5 | composite reward | step 7200 | +5.1 | 30.9 | `qwen3b_snli_hypothesis_grpo_single` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | GRPO (decomposed reward) | 1e-5 | decomposed reward | step 8100 | +8.2 | 32.5 | `qwen3b_snli_hypothesis_grpo_multi` |
| Qwen2.5-3B-Instruct | SNLI-Hypothesis | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 1500 | +15.1 | 46.0 | `qwen3b_snli_hypothesis_gdpo` |
| Qwen2.5-7B-Instruct | BoolQ | SFT | 5e-6 | max_steps=200 | step 200 | +9.4 | 79.8 | `qwen7b_boolq_sft` |
| Qwen2.5-7B-Instruct | BoolQ | DPO | 5e-6 | β=0.1, 2 pair/entry | 2 ep | +22.1 | 51.1 | `qwen7b_boolq_dpo` |
| Qwen2.5-7B-Instruct | BoolQ | SimPO | 5e-6 | β=3, γ=0.5, 2 pair/entry | 2 ep | +28.9 | 55.7 | `qwen7b_boolq_simpo` |
| Qwen2.5-7B-Instruct | BoolQ | GRPO (composite reward) | 1e-5 | composite reward | 1 ep | +21.5 | 45.4 | `qwen7b_boolq_grpo_single` |
| Qwen2.5-7B-Instruct | BoolQ | GRPO (decomposed reward) | 1e-5 | decomposed reward | 1 ep | +19.2 | 52.3 | `qwen7b_boolq_grpo_multi` |
| Qwen2.5-7B-Instruct | BoolQ | GDPO | 5e-6 | KL β=0.001, gen_batch=16 | step 200 | -0.2 | 82.3 | `qwen7b_boolq_gdpo` |
| Qwen2.5-7B-Instruct | SNLI-Premise | SFT | 5e-6 | epochs=2 | 2 ep | +0.4 | 74.5 | `qwen7b_snli_premise_sft` |
| Qwen2.5-7B-Instruct | SNLI-Premise | DPO | 5e-6 | β=0.1, 2 pair/entry | 2 ep | +24.5 | 50.9 | `qwen7b_snli_premise_dpo` |
| Qwen2.5-7B-Instruct | SNLI-Premise | SimPO | 5e-6 | β=2.0, γ=1.4, 2 pair/entry | 2 ep | +31.5 | 61.2 | `qwen7b_snli_premise_simpo` |
| Qwen2.5-7B-Instruct | SNLI-Premise | GRPO (composite reward) | 5e-6 | composite reward | step 8000 | +26.0 | 47.6 | `qwen7b_snli_premise_grpo_single` |
| Qwen2.5-7B-Instruct | SNLI-Premise | GRPO (decomposed reward) | 5e-6 | decomposed reward | 1 ep | +27.4 | 44.5 | `qwen7b_snli_premise_grpo_multi` |
| Qwen2.5-7B-Instruct | SNLI-Premise | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 2000 | +16.5 | 71.2 | `qwen7b_snli_premise_gdpo` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | SFT | 5e-6 | epochs=2 | 2 ep | +1.8 | 63.3 | `qwen7b_snli_hypothesis_sft` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | DPO | 5e-6 | β=0.1, 2 pair/entry | 2 ep | +23.2 | 44.1 | `qwen7b_snli_hypothesis_dpo` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | SimPO | 5e-6 | β=3.0, γ=0.5, 2 pair/entry | 2 ep | +31.1 | 24.2 | `qwen7b_snli_hypothesis_simpo` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | GRPO (composite reward) | 5e-6 | composite reward | 1 ep | +20.7 | 24.5 | `qwen7b_snli_hypothesis_grpo_single` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | GRPO (decomposed reward) | 5e-6 | decomposed reward | 1 ep | +22.1 | 33.8 | `qwen7b_snli_hypothesis_grpo_multi` |
| Qwen2.5-7B-Instruct | SNLI-Hypothesis | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 6000 | +17.7 | 56.6 | `qwen7b_snli_hypothesis_gdpo` |
| Qwen2.5-14B-Instruct | BoolQ | SFT | 5e-6 | max_steps=200 | step 200 | +0.5 | 89.2 | `qwen14b_boolq_sft` |
| Qwen2.5-14B-Instruct | BoolQ | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +2.7 | 76.1 | `qwen14b_boolq_dpo` |
| Qwen2.5-14B-Instruct | BoolQ | SimPO | 5e-6 | β=3, γ=0.5, 2 pair/entry | 2 ep | +7.4 | 66.0 | `qwen14b_boolq_simpo` |
| Qwen2.5-14B-Instruct | BoolQ | GRPO (composite reward) | 5e-6 | composite reward | 1 ep | +2.5 | 82.1 | `qwen14b_boolq_grpo_single` |
| Qwen2.5-14B-Instruct | BoolQ | GRPO (decomposed reward) | 5e-6 | decomposed reward | 1 ep | +2.1 | 85.2 | `qwen14b_boolq_grpo_multi` |
| Qwen2.5-14B-Instruct | BoolQ | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=64 | 1 ep | +5.7 | 67.5 | `qwen14b_boolq_gdpo` |
| Qwen2.5-14B-Instruct | SNLI-Premise | SFT | 5e-6 | epochs=1 | 1 ep | -0.0 | 65.3 | `qwen14b_snli_premise_sft` |
| Qwen2.5-14B-Instruct | SNLI-Premise | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +15.5 | 58.4 | `qwen14b_snli_premise_dpo` |
| Qwen2.5-14B-Instruct | SNLI-Premise | SimPO | 5e-6 | β=2, γ=0.5 | 2 ep | +22.3 | 68.8 | `qwen14b_snli_premise_simpo` |
| Qwen2.5-14B-Instruct | SNLI-Premise | GRPO (composite reward) | 5e-6 | composite reward | 1 ep | +22.4 | 50.5 | `qwen14b_snli_premise_grpo_single` |
| Qwen2.5-14B-Instruct | SNLI-Premise | GRPO (decomposed reward) | 5e-6 | decomposed reward | 1 ep | +30.1 | 27.4 | `qwen14b_snli_premise_grpo_multi` |
| Qwen2.5-14B-Instruct | SNLI-Premise | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=64 | 1 ep | +26.5 | 29.3 | `qwen14b_snli_premise_gdpo` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | SFT | 5e-6 | epochs=2 | 2 ep | +3.8 | 70.1 | `qwen14b_snli_hypothesis_sft` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +6.7 | 47.7 | `qwen14b_snli_hypothesis_dpo` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | SimPO | 5e-6 | β=3, γ=0.5, 2 pair/entry | 2 ep | +13.7 | 15.8 | `qwen14b_snli_hypothesis_simpo` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | GRPO (composite reward) | 5e-6 | composite reward | 1 ep | +11.6 | 36.6 | `qwen14b_snli_hypothesis_grpo_single` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | GRPO (decomposed reward) | 2e-6 | decomposed reward | step 12800 | +15.1 | 29.2 | `qwen14b_snli_hypothesis_grpo_multi` |
| Qwen2.5-14B-Instruct | SNLI-Hypothesis | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | 1 ep | +8.7 | 42.8 | `qwen14b_snli_hypothesis_gdpo` |
| Llama-3.1-8B-Instruct | BoolQ | SFT | 5e-6 | max_steps=200 | step 200 | -0.9 | 83.8 | `llama8b_boolq_sft` |
| Llama-3.1-8B-Instruct | BoolQ | DPO | 1e-5 | β=0.1, 2 pair/entry | 2 ep | +11.5 | 84.2 | `llama8b_boolq_dpo` |
| Llama-3.1-8B-Instruct | BoolQ | SimPO | 2e-6 | β=2, γ=0.5, 2 pair/entry | 2 ep | +9.0 | 70.3 | `llama8b_boolq_simpo` |
| Llama-3.1-8B-Instruct | BoolQ | GRPO (composite reward) | 5e-6 | composite reward, compl_len=768 | step 1800 | +6.1 | 86.5 | `llama8b_boolq_grpo_single` |
| Llama-3.1-8B-Instruct | BoolQ | GRPO (decomposed reward) | 5e-6 | decomposed reward, compl_len=768 | step 1600 | +12.5 | 89.2 | `llama8b_boolq_grpo_multi` |
| Llama-3.1-8B-Instruct | BoolQ | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 500 | +16.3 | 91.6 | `llama8b_boolq_gdpo` |
| Llama-3.1-8B-Instruct | SNLI-Premise | SFT | 5e-6 | max_steps=200 | step 200 | +1.9 | 82.4 | `llama8b_snli_premise_sft` |
| Llama-3.1-8B-Instruct | SNLI-Premise | DPO | 2e-6 | β=0.1, 2 pair/entry | 2 ep | +2.4 | 74.2 | `llama8b_snli_premise_dpo` |
| Llama-3.1-8B-Instruct | SNLI-Premise | SimPO | 2e-6 | β=3, γ=0.5, 2 pair/entry | 2 ep | +19.0 | 94.2 | `llama8b_snli_premise_simpo` |
| Llama-3.1-8B-Instruct | SNLI-Premise | GRPO (composite reward) | 5e-6 | composite reward | step 7500 | +31.4 | 57.8 | `llama8b_snli_premise_grpo_single` |
| Llama-3.1-8B-Instruct | SNLI-Premise | GRPO (decomposed reward) | 5e-6 | decomposed reward, KL β=0.0005, ε_high=0.28, conditioned | step 11200 | +28.4 | 78.8 | `llama8b_snli_premise_grpo_multi` |
| Llama-3.1-8B-Instruct | SNLI-Premise | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 500 | +11.2 | 87.2 | `llama8b_snli_premise_gdpo` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | SFT | 5e-6 | max_steps=200 | step 100 | -1.7 | 71.7 | `llama8b_snli_hypothesis_sft` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | DPO | 2e-6 | β=0.3, 2 pair/entry | 2 ep | +4.0 | 83.9 | `llama8b_snli_hypothesis_dpo` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | SimPO | 2e-6 | β=3, γ=0.5, 2 pair/entry | 2 ep | +3.0 | 81.2 | `llama8b_snli_hypothesis_simpo` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | GRPO (composite reward) | 5e-6 | composite reward | step 6000 | +2.6 | 83.5 | `llama8b_snli_hypothesis_grpo_single` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | GRPO (decomposed reward) | 5e-6 | decomposed reward | step 14900 | +8.6 | 65.0 | `llama8b_snli_hypothesis_grpo_multi` |
| Llama-3.1-8B-Instruct | SNLI-Hypothesis | GDPO | 5e-6 | KL β=0.0005, ε_high=0.28, conditioned, gen_batch=128 | step 1000 | +6.0 | 65.8 | `llama8b_snli_hypothesis_gdpo` |

---

## Exact training commands

### Qwen2.5-3B-Instruct — BoolQ

**SFT** — ΔLFR +1.2 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq_2000e40c_qwen25_3b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +1.7 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_1pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_1pair_b4_2ep_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +7.5 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2pair_b4_2ep_simpo_b2_g05_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 2.0 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +11.0 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16__lr1e5_qwen25_3b_v2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-4300` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +8.2 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_multi__lr1e5_qwen25_3b_mv2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

*featured checkpoint: `checkpoint-4200` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +1.3 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_boolq_1ep_g16_qwen25_3b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-100` (early stop by checkpoint sweep)*


### Qwen2.5-3B-Instruct — SNLI-Premise

**SFT** — ΔLFR +4.4 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_premise_2000e40c_qwen25_3b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +11.0 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2pair_b4_2ep_lr1e5_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +25.8 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2000e40c_2pair_simpo_b3_g05_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 0.3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +27.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_v2_lr1e5_qwen25_3b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-13200` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +25.6 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_multi_qwen25_3b_mv2_lr1e5 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward \
    --resume_from_checkpoint true
```

**GDPO** — ΔLFR +17.7 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_premise_1ep_g16_qwen25_3b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --resume_from_checkpoint true
```


### Qwen2.5-3B-Instruct — SNLI-Hypothesis

**SFT** — ΔLFR +3.1 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_hypothesis_2000e40c_qwen25_3b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +7.3 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2pair_b4_2ep_lr1e5_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +6.7 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_2pair_qwen25_3b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2000e40c_2pair_simpo_b3_g05_qwen25_3b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 0.3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +5.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16__lr1e5_qwen25_3b_v2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-7200` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +8.2 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_multi__lr1e5_qwen25_3b_mv2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

*featured checkpoint: `checkpoint-8100` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +15.1 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_hypothesis_1ep_g16_qwen25_3b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-1500` (early stop by checkpoint sweep)*


### Qwen2.5-7B-Instruct — BoolQ

**SFT** — ΔLFR +9.4 pp

```bash
python train_sft.py \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq_2000e40c \
    --learning_rate 5e-06 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --max_steps 200 \
    --use_4bit --bf16 --gradient_checkpointing
```

*command reconstructed from `training_args.bin` (no wandb log for this run)*

**DPO** — ΔLFR +22.1 pp

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2pair_b4_2ep \
    --num_train_epochs 2 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +28.9 pp

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05 \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +21.5 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_v2_lr1e5 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

**GRPO (decomposed reward)** — ΔLFR +19.2 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_multi_mv2_lr1e5 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 1e-5 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

**GDPO** — ΔLFR -0.2 pp

```bash
python train_gdpo.py \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_boolq_1ep_g16 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-200` (early stop by checkpoint sweep)*


### Qwen2.5-7B-Instruct — SNLI-Premise

**SFT** — ΔLFR +0.4 pp

```bash
python train_sft.py \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_premise_2pair_b4_2ep \
    --num_train_epochs 2 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**DPO** — ΔLFR +24.5 pp

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2pair_b4_2ep \
    --num_train_epochs 2 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +31.5 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_v2fix/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2pair_b4_2ep_simpo_v2fix \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 2.0 \
    --simpo_gamma 1.4 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +26.0 pp

```bash
python train_grpo.py \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_v2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-8000` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +27.4 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_multi_v2_fix \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

**GDPO** — ΔLFR +16.5 pp

```bash
python train_gdpo.py \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_premise_1ep_g16_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```

*featured checkpoint: `checkpoint-2000` (early stop by checkpoint sweep)*


### Qwen2.5-7B-Instruct — SNLI-Hypothesis

**SFT** — ΔLFR +1.8 pp

```bash
python train_sft.py \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_hypothesis_2pair_b4_2ep \
    --num_train_epochs 2 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**DPO** — ΔLFR +23.2 pp

```bash
python train_dpo.py \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2pair_b4_2ep \
    --num_train_epochs 2 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 200
```

**SimPO** — ΔLFR +31.1 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_v2fix/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2pair_b4_2ep_simpo_b3_g05_v2fix \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 3.0 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +20.7 pp

```bash
python train_grpo.py \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g24_v2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 24 \
    --generation_batch_size 24 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

**GRPO (decomposed reward)** — ΔLFR +22.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_multi_v2fix \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

**GDPO** — ΔLFR +17.7 pp

```bash
python train_gdpo.py \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_hypothesis_1ep_g16_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```

*featured checkpoint: `checkpoint-6000` (early stop by checkpoint sweep)*


### Qwen2.5-14B-Instruct — BoolQ

**SFT** — ΔLFR +0.5 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq_2pair_b4_ml2048_200step_qwen25_14b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 2048 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +2.7 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2pair_b4_2ep_lr1e5_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +7.4 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2pair_b4_2ep_simpo_b3_g05_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +2.5 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_1ep_g16_qwen25_14b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

**GRPO (decomposed reward)** — ΔLFR +2.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_multi_qwen25_14b_boolq_mv2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

**GDPO** — ΔLFR +5.7 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_boolq_1ep_g16_qwen25_14b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 64 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --resume_from_checkpoint true
```


### Qwen2.5-14B-Instruct — SNLI-Premise

**SFT** — ΔLFR -0.0 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_premise_2pair_b16_1ep_qwen25_14b \
    --max_steps -1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +15.5 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2pair_lr1e5_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +22.3 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_simpo_b2g05_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 2 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +22.4 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_1ep_g16_qwen25_14b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

**GRPO (decomposed reward)** — ΔLFR +30.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_multi_qwen25_14b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward \
    --resume_from_checkpoint true
```

**GDPO** — ΔLFR +26.5 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_premise_1ep_g16_qwen25_14b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 64 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --resume_from_checkpoint true
```


### Qwen2.5-14B-Instruct — SNLI-Hypothesis

**SFT** — ΔLFR +3.8 pp

```bash
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_hypothesis_2pair_b4_2ep_qwen25_14b \
    --max_steps -1 \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +6.7 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2pair_lr1e5_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +13.7 pp

```bash
python train_dpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_qwen25_14b/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2pair_b4_2ep_simpo_b3_g05_qwen25_14b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 5e-6 \
    --beta 3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +11.6 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_1ep_g16_qwen25_14b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

**GRPO (decomposed reward)** — ΔLFR +15.1 pp

```bash
python train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_multi_qwen25_14b_mv2_lr2e6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 2e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-12800` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +8.7 pp

```bash
python train_gdpo.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_hypothesis_1ep_g16_qwen25_14b_v6_v2fix \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```


### Llama-3.1-8B-Instruct — BoolQ

**SFT** — ΔLFR -0.9 pp

```bash
python train_sft.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c_2pair_llama31_8b/dpo_training.jsonl \
    --output_dir ./results/sft_model_boolq_2000e40c_llama31_8b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +11.5 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2000e40c_2pair_lr1e5_llama31_8b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +9.0 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_boolq_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_boolq_2000e40c_2pair_simpo_b2_g05_lr2e6_llama31_8b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 2e-6 \
    --max_grad_norm 1.0 \
    --beta 2 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +6.1 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_llama31_8b_v2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 768 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100
```

*featured checkpoint: `checkpoint-1800` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +12.5 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_boolq_1ep_g16_multi__mcl768_llama31_8b_mv2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 768 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

*featured checkpoint: `checkpoint-1600` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +16.3 pp

```bash
python train_gdpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name boolq \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_boolq_1ep_g16_llama31_8b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```

*featured checkpoint: `checkpoint-500` (early stop by checkpoint sweep)*


### Llama-3.1-8B-Instruct — SNLI-Premise

**SFT** — ΔLFR +1.9 pp

```bash
python train_sft.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c_2pair_llama31_8b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_premise_2000e40c_llama31_8b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

**DPO** — ΔLFR +2.4 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path results/dpo_pairs_snli_premise_2000e40c_2pair_llama31_8b/dpo_training.jsonl \
    --output_dir results/dpo_model_snli_premise_2000e40c_2pair_lr2e6_llama31_8b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 2e-6 \
    --beta 0.1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +19.0 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_premise_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_premise_2000e40c_2pair_simpo_b3_g05_lr2e6_llama31_8b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 2e-6 \
    --max_grad_norm 1.0 \
    --beta 3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +31.4 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_llama31_8b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-7500` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +28.4 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_premise_1ep_g16_multi_fair_multi_llama31_8b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --epsilon_high 0.28 \
    --conditioned_rewards \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward
```

*featured checkpoint: `checkpoint-11200` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +11.2 pp

```bash
python train_gdpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_premise \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_premise_1ep_g16_llama31_8b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```

*featured checkpoint: `checkpoint-500` (early stop by checkpoint sweep)*


### Llama-3.1-8B-Instruct — SNLI-Hypothesis

**SFT** — ΔLFR -1.7 pp

```bash
python train_sft.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c_2pair_llama31_8b/dpo_training.jsonl \
    --output_dir ./results/sft_model_snli_hypothesis_2000e40c_llama31_8b \
    --max_steps 200 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 50
```

*featured checkpoint: `checkpoint-100` (early stop by checkpoint sweep)*

**DPO** — ΔLFR +4.0 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c/dpo_training.jsonl \
    --output_dir results/dpo_model_snli_hypothesis_2000e40c_2pair_lr2e6_beta03_llama31_8b \
    --num_train_epochs 2 \
    --loss_type sigmoid \
    --learning_rate 2e-6 \
    --beta 0.3 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**SimPO** — ΔLFR +3.0 pp

```bash
python train_dpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./results/dpo_pairs_snli_hypothesis_2000e40c/dpo_training.jsonl \
    --output_dir ./results/dpo_model_snli_hypothesis_2000e40c_2pair_simpo_b3_g05_lr2e6_llama31_8b \
    --num_train_epochs 2 \
    --loss_type simpo \
    --learning_rate 2e-6 \
    --max_grad_norm 1.0 \
    --beta 3 \
    --simpo_gamma 0.5 \
    --cpo_alpha 0.0 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 100
```

**GRPO (composite reward)** — ΔLFR +2.6 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_llama31_8b \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-6000` (early stop by checkpoint sweep)*

**GRPO (decomposed reward)** — ΔLFR +8.6 pp

```bash
python train_grpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/grpo_model_snli_hypothesis_1ep_g16_multi_llama31_8b_mv2 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --multi_reward \
    --resume_from_checkpoint true
```

*featured checkpoint: `checkpoint-14900` (early stop by checkpoint sweep)*

**GDPO** — ΔLFR +6.0 pp

```bash
python train_gdpo.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name snli_hypothesis \
    --max_entries 2000 \
    --output_dir ./results/gdpo_model_snli_hypothesis_1ep_g16_llama31_8b_v6 \
    --num_train_epochs 1 \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_generations 16 \
    --generation_batch_size 128 \
    --max_completion_length 512 \
    --learning_rate 5e-6 \
    --beta 0.0005 \
    --temperature 1.2 \
    --logging_steps 1 \
    --save_steps 100 \
    --ned_penalty_alpha 0.0 \
    --epsilon_high 0.28 \
    --conditioned_rewards
```

*featured checkpoint: `checkpoint-1000` (early stop by checkpoint sweep)*

