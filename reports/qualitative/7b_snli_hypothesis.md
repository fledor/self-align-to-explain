# Qualitative Counterfactual Comparison — Qwen2.5-7B-Instruct, SNLI-Hypothesis

Each method edits the **hypothesis** to flip the NLI label (judged by the base model). For every sample below, all six methods produced a *successful* flip, so edits are directly comparable.

**Source runs (current best per method, as featured in `results_charts.html`):**

- **DPO** — `results/evaluation_snli_hypothesis_200s_dpo_2pair_fair/`
- **SimPO** — `results/evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix/`
- **SFT** — `results/evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep/`
- **GRPO single** — `results/evaluation_snli_hypothesis_200s_grpo_v2_g24/`
- **GRPO multi** — `results/evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix/`
- **GDPO** — `results/evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000/`

Examples are a **representative cross-section** (stratified across label transitions, readable length, fluent), *not* cherry-picked minimal edits. For each method we show a **typical (median-NED) flipping counterfactual**, so the comparison reflects what each method actually produces. Metrics per CF: **NED** = normalized edit distance; **PPL** = perplexity of the edit (lower = more fluent).

---

## Example 1  (dataset idx 9445)

| | |
|---|---|
| **Hypothesis (original)** | A group of people are gathered to cook chicken on the grill. |
| **Premise (fixed)** | A group of people in white sports outfits are gathered together while one of them performs a dance. |
| **Original label** | contradiction |
| **Target label** | entailment |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A group of people are gathered together. | 0.350 | 26 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A group of people are gather together. | 0.383 | 50 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A group of people are gathering for an event. | 0.417 | 16 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A group of people are not gathering to cook chicken on the grill. | 0.108 | 71 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A group of people are gather together while one of them performs a dance. | 0.425 | 31 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A group of people are gather[ed] to perform a dance. | 0.400 | 44 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 2  (dataset idx 1912)

| | |
|---|---|
| **Hypothesis (original)** | Three females are on an airplane. |
| **Premise (fixed)** | Three ladies wearing dresses are bent down in a muddy creek gathering green plants. |
| **Original label** | contradiction |
| **Target label** | neutral |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Three females are on an airplane or in a muddy creek. | 0.377 | 347 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | Three females are near green plants. | 0.306 | 1037 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | Three females are not on an airplane but they are in a muddy creek gathering green plants. | 0.633 | 191 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | Three females are not on an airplane. | 0.108 | 150 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | Three females are bending down to gather plants. | 0.396 | 145 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Three females are gathered together. | 0.444 | 247 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 3  (dataset idx 2849)

| | |
|---|---|
| **Hypothesis (original)** | A man was pulling a rickshaw behind him. |
| **Premise (fixed)** | A man wearing a brown shirt and green pans is pulling a rickshaw down a street |
| **Original label** | entailment |
| **Target label** | contradiction |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A man was not pulling a rickshaw behind him. | 0.091 | 70 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A man was pushing a rickshaw in front of him. | 0.267 | 32 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A man was pushing a rickshaw ahead of him. | 0.214 | 39 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A man was not pulling a rickshaw behind him. | 0.091 | 68 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A man was not pulling a rickshaw behind him. | 0.091 | 72 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A man was not pulling a rickshaw. | 0.375 | 76 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 4  (dataset idx 4128)

| | |
|---|---|
| **Hypothesis (original)** | A child jumps high, assisted by a spotter and a harness. |
| **Premise (fixed)** | A young girl jumps high with the help of her bungie harness device and a safety spotter. |
| **Original label** | entailment |
| **Target label** | neutral |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A child jumps high, assisted only by a spotter. | 0.339 | 116 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A child jumps high unassisted. | 0.536 | 245 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A child jumps high, assisted by a spotter and a harness, which makes the jump easier but not necessarily high. | 0.491 | 41 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A child does not jumps high, assisted by a spotter and a harness. | 0.138 | 127 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A child does not jumps high, assisted by a spotter and a harness. | 0.138 | 124 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A child jumps high, assisted only by a harness. | 0.250 | 135 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 5  (dataset idx 9448)

| | |
|---|---|
| **Hypothesis (original)** | A small dark-haired boy wearing a blue shirt is climbing in a large tree with a tree house in it. |
| **Premise (fixed)** | A small dark-haired boy climbing in a large tree. |
| **Original label** | neutral |
| **Target label** | contradiction |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A small dark-haired boy is not climbing in a large tree. | 0.454 | 110 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A small dark-haired boy climbing in a small bush. | 0.557 | 168 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A small dark-haired boy climbing in a small shrub. | 0.557 | 131 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A small dark-haired boy is not climbing in a large tree | 0.464 | 152 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A small dark-haired boy is not climbing in a large tree. | 0.454 | 113 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A small dark-haired boy is climbing in a large tree. | 0.464 | 43 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 6  (dataset idx 3351)

| | |
|---|---|
| **Hypothesis (original)** | A man is climbing the grand canyon. |
| **Premise (fixed)** | A man is rock climbing with protective rope. |
| **Original label** | neutral |
| **Target label** | entailment |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A man is climbing with protective rope. | 0.410 | 632 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A man is not climbing anywhere. | 0.514 | 329 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A man is swimming in a pool. | 0.429 | 23 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A man is not climbing. | 0.600 | 418 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A man is not climbing. | 0.600 | 403 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A man is rock climbing with protective rope. | 0.477 | 165 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 7  (dataset idx 6030)

| | |
|---|---|
| **Hypothesis (original)** | Three dogs run through the arctic. |
| **Premise (fixed)** | Three dogs drinking water at a public spigot. |
| **Original label** | contradiction |
| **Target label** | entailment |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Three dogs are drinking water. | 0.529 | 110 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | Three dogs are near a water source. | 0.543 | 75 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | Three dogs run through the snow near a public spigot. | 0.396 | 81 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | [Three dogs do not run through the arctic.] | 0.209 | 365 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | Three dogs drink water at a public spigot. | 0.548 | 143 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Three dogs drink water at a public spigot. | 0.548 | 143 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 8  (dataset idx 3812)

| | |
|---|---|
| **Hypothesis (original)** | The woman is alone in her bathroom. |
| **Premise (fixed)** | Two women dressed for cold weather in jackets and gloves look at something on a cellphone screen. |
| **Original label** | contradiction |
| **Target label** | neutral |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | The woman is alone in her bathroom or she is looking at something on a cellphone screen. | 0.602 | 53 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | or with others | 0.800 | 2322 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | The two women are together looking at something on a cellphone screen. | 0.700 | 59 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | The women is not alone in her bathroom. | 0.128 | 125 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | The women are not alone | 0.543 | 493 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | The women look at something on a cellphone screen. | 0.600 | 127 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 9  (dataset idx 1863)

| | |
|---|---|
| **Hypothesis (original)** | A person in a green knit cap stands ready to push a sled, with a little boy behind her. |
| **Premise (fixed)** | A girl in a green knit cap stands ready to push a sled, with a little boy behind her. |
| **Original label** | entailment |
| **Target label** | contradiction |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A person in a green knit cap sits ready to push a sled, without a little boy behind her. | 0.080 | 120 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | A person in no cap stands ready to push a sled, with a little boy behind. | 0.172 | 121 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | A person in a red knit cap stands ready to push a sled, with a little boy in front of her. | 0.144 | 28 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | A person is not standing ready to push a sled, with a little boy behind her. | 0.207 | 86 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | A person in a green knit cap does not stand ready to push a sled, with a little boy behind her. | 0.105 | 100 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A person in a green knit cap stands ready to push a sled, without the little boy is away. | 0.169 | 108 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |

## Example 10  (dataset idx 3489)

| | |
|---|---|
| **Hypothesis (original)** | The man likes the hot tub. |
| **Premise (fixed)** | A guy in a robe sitting on the edge of a hot tub smiling. |
| **Original label** | neutral |
| **Target label** | contradiction |

| Method | Edited hypothesis | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | The man dislikes the hot tub. | 0.103 | 952 | `evaluation_snli_hypothesis_200s_dpo_2pair_fair` |
| **SimPO** | The man dislikes the hot tub. | 0.103 | 1193 | `evaluation_snli_hypothesis_200s_simpo_b3g05_v2fix` |
| **SFT** | The man hates the hot tub. | 0.115 | 897 | `evaluation_snli_hypothesis_200s_sft_2pair_b4_2ep` |
| **GRPO single** | The man is in the hot tub. | 0.192 | 36 | `evaluation_snli_hypothesis_200s_grpo_v2_g24` |
| **GRPO multi** | The man is sitting on the hot tub. | 0.353 | 42 | `evaluation_snli_hypothesis_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | The man enjoying the hot tub. | 0.276 | 440 | `evaluation_snli_hypothesis_200s_gdpo_v6_g16_ckpt6000` |
