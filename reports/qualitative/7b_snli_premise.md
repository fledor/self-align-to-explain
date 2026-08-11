# Qualitative Counterfactual Comparison — Qwen2.5-7B-Instruct, SNLI-Premise

Each method edits the **premise** to flip the NLI label (judged by the base model). For every sample below, all six methods produced a *successful* flip, so edits are directly comparable.

**Source runs (current best per method, as featured in `results_charts.html`):**

- **DPO** — `results/evaluation_snli_premise_200s_dpo_2pair_fair/`
- **SimPO** — `results/evaluation_snli_premise_200s_simpo_2pair_v2fix/`
- **SFT** — `results/evaluation_snli_premise_200s_sft_fair/`
- **GRPO single** — `results/evaluation_snli_premise_200s_grpo_v2_g16/`
- **GRPO multi** — `results/evaluation_snli_premise_200s_grpo_mv2_g16_v2fix/`
- **GDPO** — `results/evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000/`

Examples are a **representative cross-section** (stratified across label transitions, readable length, fluent), *not* cherry-picked minimal edits. For each method we show a **typical (median-NED) flipping counterfactual**, so the comparison reflects what each method actually produces. Metrics per CF: **NED** = normalized edit distance; **PPL** = perplexity of the edit (lower = more fluent).

---

## Example 1  (dataset idx 1521)

| | |
|---|---|
| **Premise (original)** | Man walking in front of Queen and Portland Lofts and Condominiums. |
| **Hypothesis (fixed)** | A man is lying down in the park. |
| **Original label** | contradiction |
| **Target label** | entailment |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A man is lying down in the park in front of Queen and Portland Lofts and Condominiums. | 0.279 | 86 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A man walking in front of Queen and Portland Lofts and Condominiums, and also lying down in the park. | 0.356 | 78 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | Man walking in front of Queen and Portland Lofts and Condominiums. The man could be lying down in the park. | 0.383 | 66 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | Man walking in front of Queen and Portland Lofts and Condominiums and also lying down in the park | 0.330 | 84 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | Man walking in front of Queen and Portland Lofts and Condominiums and he is lying down in the park. | 0.333 | 75 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Man walking in front of Queen and Portland Lofts and Condominiums; he could be lying down in the park. | 0.353 | 91 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 2  (dataset idx 5257)

| | |
|---|---|
| **Premise (original)** | Two men look down at another man standing on an unfinished walkway. |
| **Hypothesis (fixed)** | Two women look down at a finished walkway. |
| **Original label** | contradiction |
| **Target label** | neutral |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Two men look down at another man standing on a finished walkway. | 0.045 | 53 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | Two men look down at another man standing on a finished walkway while two women are also looking down at a finished walkway. | 0.468 | 22 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | Two men look down at another man standing on a walkway. | 0.179 | 29 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | Two men look down at another man standing on an unfinished walkway and also two women look down at a finished walkway. | 0.432 | 31 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | Two men look down at another man standing on an unfinished walkway and a finished walkway. | 0.256 | 32 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Two men look down at another man standing on a finished walkway. | 0.045 | 61 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 3  (dataset idx 9792)

| | |
|---|---|
| **Premise (original)** | Rival female basketball teams are playing a game in front of an audience at an indoor arena. |
| **Hypothesis (fixed)** | Two female basketball teams are playing a game inside of a building. |
| **Original label** | entailment |
| **Target label** | contradiction |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Rival female basketball teams are playing a game in front of an audience outside at an outdoor arena. | 0.109 | 56 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | Rival female basketball teams are playing a game in front of an audience outdoors. | 0.152 | 85 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | Rival female basketball teams are playing a game in front of an audience outdoors. | 0.152 | 78 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | Rival female basketball teams are playing a game in front of an audience at an indoor arena but not inside of a building | 0.242 | 53 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | Rival female basketball teams are playing a game in front of an audience at an indoor arena and it is not necessarily inside a building. | 0.324 | 58 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Rival female basketball teams are playing a game in front of an audience outside an indoor arena. | 0.062 | 60 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 4  (dataset idx 1568)

| | |
|---|---|
| **Premise (original)** | Four females wearing helments are riding on an ATV. |
| **Hypothesis (fixed)** | Four women are driving an all terrain vehicle. |
| **Original label** | entailment |
| **Target label** | neutral |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Four females wearing helmets are riding on an all terrain vehicle. | 0.303 | 49 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | Four females wearing helmets are watching others ride on an ATV. | 0.281 | 130 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | Four females wearing helmets are laying down. | 0.235 | 432 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | Four females wearing helment are riding on an ATV but are not driving an all terrain vehicle | 0.478 | 86 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | Four females wearing helments are riding on an ATV and are not driving an all terrain vehicle. | 0.457 | 59 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Four females wearing helmets are near an ATV. | 0.176 | 283 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 5  (dataset idx 6218)

| | |
|---|---|
| **Premise (original)** | A group of people plays a game on the floor of a living room while a TV plays in the background. |
| **Hypothesis (fixed)** | A group of friends are playing a board game at their friend's birthday party. |
| **Original label** | neutral |
| **Target label** | contradiction |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A group of people plays on the floor of a living room while a TV plays, but they are not playing a game and it is not a friend's birthday party. | 0.479 | 20 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A group of people plays on the floor of a living room while a TV plays in the background, and no friends are playing a board game at a birthday party. | 0.453 | 27 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | A group of people plays a game on the floor of a living room while a TV plays in the background; however, they are not playing a board game at a birthday party. | 0.400 | 23 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | A group of people plays a game on the floor of a living room while a TV plays in the background and they are not playing a board game at a birthday party | 0.379 | 29 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | A group of people plays a game on the floor of a living room while a TV plays in the background, and they are not at朋友的生日派对并且没有下棋。 | 0.269 | 37 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A group of people plays a board game on the floor of a living room while a TV plays in the background, celebrating their friend's birthday. | 0.309 | 19 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 6  (dataset idx 7028)

| | |
|---|---|
| **Premise (original)** | A woman in a tan dress is tending to her wares in the market. |
| **Hypothesis (fixed)** | The woman is selling pies. |
| **Original label** | neutral |
| **Target label** | entailment |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A woman in a tan dress is tending to her wares and selling pies in the market. | 0.218 | 47 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A woman in a tan dress is tending to her wares in the market and selling pies. | 0.218 | 42 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | A woman in a tan dress is tending to her wares and selling pies in the market. | 0.218 | 46 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | A woman in a tan dress is tending to her wares in the market and is selling pies. | 0.247 | 42 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | A woman in a tan dress is tending to her wares in the market and is selling pies. | 0.247 | 43 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A woman in a tan dress is tending to her wares in the market, and she is selling pies. | 0.291 | 32 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 7  (dataset idx 2097)

| | |
|---|---|
| **Premise (original)** | A black dog runs along the green grass carrying a toy in its mouth. |
| **Hypothesis (fixed)** | A white dog is sitting under a tree. |
| **Original label** | contradiction |
| **Target label** | entailment |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A black dog runs along the green grass carrying a toy in its mouth while a white dog is sitting under a tree. | 0.385 | 14 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A black dog runs along the green grass carrying a toy in its mouth while a white dog is sitting under a tree. | 0.385 | 13 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | A black dog runs along the green grass carrying a toy in its mouth while a white dog is sitting under a tree. | 0.385 | 13 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | A black dog runs along the green grass carrying a toy in its mouth and a white dog is sitting under a tree simultaneously | 0.455 | 20 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | A black dog runs along the green grass carrying a toy in its mouth and a white dog is sitting under a tree. | 0.374 | 14 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A black dog runs along the green grass carrying a toy in its mouth, while a white dog is sitting under a tree. | 0.391 | 11 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 8  (dataset idx 7651)

| | |
|---|---|
| **Premise (original)** | Woman with headscarf and girl in red costume dress walk through urban area. |
| **Hypothesis (fixed)** | The woman is out in a field. |
| **Original label** | contradiction |
| **Target label** | neutral |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Woman with headscarf and girl in red costume dress walk through an urban area or a field. | 0.157 | 55 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | Woman with headscarf and girl in red costume dress walk through urban area or in a field. | 0.157 | 57 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | Woman with headscarf and girl in red costume dress walk through field. | 0.133 | 50 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | Woman with headscarf and girl in red costume dress walk through urban area and they are in a field | 0.245 | 50 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | Woman with headscarf and girl in red costume dress walk through urban area and they are out in a field. | 0.272 | 49 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | Woman with headscarf and girl in red costume dress walking through an urban area or a field. | 0.185 | 54 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 9  (dataset idx 4416)

| | |
|---|---|
| **Premise (original)** | A lady looks to her right and holds a mini camera. |
| **Hypothesis (fixed)** | A lady is holding a device. |
| **Original label** | entailment |
| **Target label** | contradiction |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A lady looks to her right with no device present. | 0.380 | 221 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A lady looks to her right without no device. | 0.340 | 319 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | A lady looks to her right. | 0.480 | 116 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | A lady looks to her right and holds a mini camera but a lady is not holding a device | 0.417 | 83 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | A lady looks to her right and holds a mini camera, and she is not holding a device. | 0.398 | 53 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A lady looks to her left and does not hold any device. | 0.426 | 120 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |

## Example 10  (dataset idx 5633)

| | |
|---|---|
| **Premise (original)** | A man in a t-shirt is stirring something in a pot which is on a counter near a microwave oven. |
| **Hypothesis (fixed)** | A man is stirring something near a microwave |
| **Original label** | entailment |
| **Target label** | neutral |

| Method | Edited premise | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | A man in a t-shirt is stirring something in a pot which is on a counter. | 0.234 | 19 | `evaluation_snli_premise_200s_dpo_2pair_fair` |
| **SimPO** | A man in a t-shirt is stirring something in a pot on a counter away from a microwave oven. | 0.170 | 28 | `evaluation_snli_premise_200s_simpo_2pair_v2fix` |
| **SFT** | A man in a t-shirt is stirring something in a pot which is on a counter. | 0.234 | 19 | `evaluation_snli_premise_200s_sft_fair` |
| **GRPO single** | A man in a t-shirt is stirring something in a pot which is on a counter near a microwave and not near a microwave | 0.195 | 35 | `evaluation_snli_premise_200s_grpo_v2_g16` |
| **GRPO multi** | A man in a t-shirt is stirring something in a pot which is on a counter near a microwave oven and not near a microwave. | 0.210 | 35 | `evaluation_snli_premise_200s_grpo_mv2_g16_v2fix` |
| **GDPO** | A man in a t-shirt is not stirring anything in a pot near a microwave oven. | 0.319 | 43 | `evaluation_snli_premise_200s_gdpo_v6_g16_ckpt2000` |
