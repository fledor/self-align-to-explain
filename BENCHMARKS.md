# General-Capability Benchmarks (MMLU + ANLI)

Zero-shot `lm-eval` (mmlu, anli_r1/r2/r3), `acc`. Each adapter from `best_models/` applied to its base; Δ columns are vs the **unmodified base model**. ANLI = mean of r1/r2/r3. Purpose: check whether CF fine-tuning degrades general capability. 76 models (4 base + 72 adapters).

## Base models (reference)

| Base | MMLU | ANLI-r1 | ANLI-r2 | ANLI-r3 | ANLI avg |
|------|-----:|--------:|--------:|--------:|---------:|
| Qwen-3B | 65.2 | 56.6 | 45.8 | 49.1 | 50.5 |
| Qwen-7B | 71.0 | 64.3 | 54.9 | 53.2 | 57.5 |
| Qwen-14B | 78.8 | 71.9 | 64.0 | 61.3 | 65.7 |
| Llama-8B | 67.8 | 48.7 | 46.7 | 44.4 | 46.6 |

## Adapters — MMLU acc and Δ vs base

| Model | Dataset | Method | MMLU | ΔMMLU | ANLI avg | ΔANLI |
|-------|---------|--------|-----:|------:|---------:|------:|
| Qwen-3B | BoolQ | DPO | 65.2 | -0.0 | 50.8 | +0.3 |
| Qwen-3B | BoolQ | SimPO | 65.1 | -0.1 | 50.7 | +0.3 |
| Qwen-3B | BoolQ | SFT | 65.3 | +0.2 | 50.9 | +0.4 |
| Qwen-3B | BoolQ | GRPO-S | 65.3 | +0.1 | 50.6 | +0.1 |
| Qwen-3B | BoolQ | GRPO-M | 65.3 | +0.1 | 51.0 | +0.5 |
| Qwen-3B | BoolQ | GDPO | 65.4 | +0.2 | 50.4 | -0.1 |
| Qwen-3B | SNLI-P | DPO | 65.2 | +0.0 | 50.7 | +0.2 |
| Qwen-3B | SNLI-P | SimPO | 65.3 | +0.1 | 51.1 | +0.6 |
| Qwen-3B | SNLI-P | SFT | 65.4 | +0.2 | 51.5 | +1.0 |
| Qwen-3B | SNLI-P | GRPO-S | 65.2 | +0.1 | 50.7 | +0.2 |
| Qwen-3B | SNLI-P | GRPO-M | 65.4 | +0.2 | 50.3 | -0.2 |
| Qwen-3B | SNLI-P | GDPO | 65.3 | +0.1 | 50.6 | +0.1 |
| Qwen-3B | SNLI-H | DPO | 65.4 | +0.2 | 50.7 | +0.2 |
| Qwen-3B | SNLI-H | SimPO | 65.2 | +0.0 | 51.7 | +1.2 |
| Qwen-3B | SNLI-H | SFT | 65.4 | +0.2 | 50.7 | +0.2 |
| Qwen-3B | SNLI-H | GRPO-S | 65.3 | +0.1 | 51.2 | +0.7 |
| Qwen-3B | SNLI-H | GRPO-M | 65.4 | +0.2 | 50.6 | +0.1 |
| Qwen-3B | SNLI-H | GDPO | 65.2 | +0.1 | 50.2 | -0.2 |
| Qwen-7B | BoolQ | DPO | 71.0 | -0.0 | 58.2 | +0.8 |
| Qwen-7B | BoolQ | SimPO | 71.1 | +0.0 | 58.2 | +0.7 |
| Qwen-7B | BoolQ | SFT | 71.0 | -0.1 | 57.6 | +0.2 |
| Qwen-7B | BoolQ | GRPO-S | 71.0 | -0.1 | 57.9 | +0.4 |
| Qwen-7B | BoolQ | GRPO-M | 71.0 | -0.1 | 58.0 | +0.5 |
| Qwen-7B | BoolQ | GDPO | 71.1 | +0.1 | 58.0 | +0.5 |
| Qwen-7B | SNLI-P | DPO | 71.1 | +0.0 | 57.8 | +0.3 |
| Qwen-7B | SNLI-P | SimPO | 71.1 | +0.1 | 57.5 | +0.0 |
| Qwen-7B | SNLI-P | SFT | 71.0 | -0.0 | 57.2 | -0.2 |
| Qwen-7B | SNLI-P | GRPO-S | 71.1 | +0.1 | 57.7 | +0.2 |
| Qwen-7B | SNLI-P | GRPO-M | 70.9 | -0.1 | 57.5 | -0.0 |
| Qwen-7B | SNLI-P | GDPO | 70.9 | -0.1 | 57.7 | +0.3 |
| Qwen-7B | SNLI-H | DPO | 71.1 | +0.0 | 57.9 | +0.4 |
| Qwen-7B | SNLI-H | SimPO | 71.0 | +0.0 | 57.4 | -0.1 |
| Qwen-7B | SNLI-H | SFT | 71.0 | -0.0 | 57.4 | -0.1 |
| Qwen-7B | SNLI-H | GRPO-S | 71.0 | +0.0 | 57.5 | +0.0 |
| Qwen-7B | SNLI-H | GRPO-M | 71.0 | -0.0 | 57.8 | +0.3 |
| Qwen-7B | SNLI-H | GDPO | 71.0 | -0.1 | 57.5 | +0.1 |
| Qwen-14B | BoolQ | DPO | 78.9 | +0.1 | 65.4 | -0.3 |
| Qwen-14B | BoolQ | SimPO | 78.7 | -0.1 | 65.6 | -0.1 |
| Qwen-14B | BoolQ | SFT | 78.9 | +0.1 | 65.7 | -0.1 |
| Qwen-14B | BoolQ | GRPO-S | 78.8 | +0.0 | 65.4 | -0.3 |
| Qwen-14B | BoolQ | GRPO-M | 78.8 | -0.0 | 65.7 | +0.0 |
| Qwen-14B | BoolQ | GDPO | 78.9 | +0.1 | 65.3 | -0.4 |
| Qwen-14B | SNLI-P | DPO | 78.9 | +0.0 | 65.4 | -0.3 |
| Qwen-14B | SNLI-P | SimPO | 78.9 | +0.1 | 65.5 | -0.3 |
| Qwen-14B | SNLI-P | SFT | 78.9 | +0.1 | 65.5 | -0.3 |
| Qwen-14B | SNLI-P | GRPO-S | 78.9 | +0.1 | 65.3 | -0.4 |
| Qwen-14B | SNLI-P | GRPO-M | 78.9 | +0.0 | 65.3 | -0.4 |
| Qwen-14B | SNLI-P | GDPO | 79.0 | +0.2 | 65.3 | -0.4 |
| Qwen-14B | SNLI-H | DPO | 78.9 | +0.1 | 65.0 | -0.7 |
| Qwen-14B | SNLI-H | SimPO | 78.9 | +0.1 | 65.2 | -0.5 |
| Qwen-14B | SNLI-H | SFT | 78.9 | +0.1 | 65.1 | -0.7 |
| Qwen-14B | SNLI-H | GRPO-S | 78.9 | +0.1 | 65.6 | -0.1 |
| Qwen-14B | SNLI-H | GRPO-M | 78.9 | +0.0 | 65.1 | -0.6 |
| Qwen-14B | SNLI-H | GDPO | 78.9 | +0.0 | 65.3 | -0.4 |
| Llama-8B | BoolQ | DPO | 67.9 | +0.1 | 46.8 | +0.2 |
| Llama-8B | BoolQ | SimPO | 67.8 | -0.0 | 47.0 | +0.4 |
| Llama-8B | BoolQ | SFT | 67.8 | -0.0 | 46.4 | -0.2 |
| Llama-8B | BoolQ | GRPO-S | 67.8 | +0.0 | 46.8 | +0.2 |
| Llama-8B | BoolQ | GRPO-M | 67.8 | +0.0 | 46.5 | -0.1 |
| Llama-8B | BoolQ | GDPO | 67.8 | +0.0 | 46.4 | -0.2 |
| Llama-8B | SNLI-P | DPO | 67.7 | -0.1 | 46.5 | -0.1 |
| Llama-8B | SNLI-P | SimPO | 67.9 | +0.1 | 46.4 | -0.2 |
| Llama-8B | SNLI-P | SFT | 67.7 | -0.1 | 46.2 | -0.4 |
| Llama-8B | SNLI-P | GRPO-S | 67.4 | -0.4 | 46.8 | +0.2 |
| Llama-8B | SNLI-P | GRPO-M | 67.8 | -0.0 | 46.7 | +0.1 |
| Llama-8B | SNLI-P | GDPO | 67.8 | +0.0 | 46.8 | +0.2 |
| Llama-8B | SNLI-H | DPO | 67.8 | -0.0 | 46.3 | -0.3 |
| Llama-8B | SNLI-H | SimPO | 68.0 | +0.2 | 46.6 | -0.0 |
| Llama-8B | SNLI-H | SFT | 67.8 | +0.0 | 46.4 | -0.2 |
| Llama-8B | SNLI-H | GRPO-S | 67.9 | +0.1 | 47.1 | +0.5 |
| Llama-8B | SNLI-H | GRPO-M | 67.8 | +0.0 | 47.5 | +0.9 |
| Llama-8B | SNLI-H | GDPO | 67.7 | -0.1 | 47.0 | +0.4 |

## Summary

- Adapters evaluated: 72/72
- ΔMMLU vs base: mean +0.04pp, min -0.37, max +0.21
- ΔANLI vs base: mean +0.07pp, min -0.68, max +1.19
- (ΔMMLU and ΔANLI are independent metrics — the MMLU min and ANLI min are different cells.)
- Largest MMLU drops:
    - llama8b_snli_premise_grpo_single: ΔMMLU -0.4pp (MMLU 67.4)
    - qwen7b_snli_premise_grpo_multi: ΔMMLU -0.1pp (MMLU 70.9)
    - qwen7b_snli_premise_gdpo: ΔMMLU -0.1pp (MMLU 70.9)
    - llama8b_snli_premise_dpo: ΔMMLU -0.1pp (MMLU 67.7)
    - qwen14b_boolq_simpo: ΔMMLU -0.1pp (MMLU 78.7)
- Largest ANLI drops:
    - qwen14b_snli_hypothesis_dpo: ΔANLI -0.7pp (ANLI 65.0)
    - qwen14b_snli_hypothesis_sft: ΔANLI -0.7pp (ANLI 65.1)
    - qwen14b_snli_hypothesis_grpo_multi: ΔANLI -0.6pp (ANLI 65.1)
    - qwen14b_snli_hypothesis_simpo: ΔANLI -0.5pp (ANLI 65.2)
    - qwen14b_snli_premise_grpo_multi: ΔANLI -0.4pp (ANLI 65.3)
