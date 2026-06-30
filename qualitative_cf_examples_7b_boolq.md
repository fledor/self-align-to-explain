# Qualitative Counterfactual Comparison — Qwen2.5-7B-Instruct, BoolQ

Each method edits the **passage** to flip the yes/no answer (judged by the base model). Edits are shown as a compact word-level diff (`~~removed~~`, `**+added**`, `~~old~~ **→ new**`) against the original passage, with long unchanged spans elided (`…`). For every sample, all six methods produced a *successful* flip.

**Source runs (current best per method, as featured in `results_charts.html`):**

- **DPO** — `results/evaluation_boolq_200s_dpo_2pair_fair/`
- **SimPO** — `results/evaluation_boolq_200s_simpo_b3g05_fair/`
- **SFT** — `results/evaluation_boolq_200s_sft_fair/`
- **GRPO single** — `results/evaluation_boolq_200s_grpo_v2g16_lr1e5_fair/`
- **GRPO multi** — `results/evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair/`
- **GDPO** — `results/evaluation_boolq_200s_gdpo_g16_ckpt200/`

Examples are a **representative cross-section** (stratified across label transitions, readable length, fluent), *not* cherry-picked minimal edits. For each method we show a **typical (median-NED) flipping counterfactual**, so the comparison reflects what each method actually produces. Metrics per CF: **NED** = normalized edit distance; **PPL** = perplexity of the edit (lower = more fluent).

---

## Example 1  (dataset idx 585)

| | |
|---|---|
| **Question** | is there a season 4 for cedar cove |
| **Original answer** | false |
| **Target answer** | true |
| **Passage (original)** | On December 1, 2015, it was announced that the series was cancelled and would not be renewed for a fourth season. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | On December 1, 2015, it was announced that the series was cancelled ~~and would not be~~ **→ but was later** renewed for a fourth ~~season.~~ **→ season and it did happen.** | 0.234 | 12 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | On December 1, 2015, it was announced that the series was cancelled ~~and would not be~~ **→ but was later** renewed for a fourth ~~season.~~ **→ season and it occurred.** | 0.222 | 13 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | On December 1, 2015, it was announced that the series ~~was~~ **→ would not be** cancelled and would ~~not~~ be renewed for a fourth season. | 0.127 | 5 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | On December 1, 2015, it was announced that the series was cancelled ~~and would not be~~ **→ but was later** renewed for a fourth season. | 0.106 | 7 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | On December 1, 2015, it was announced that the series was cancelled ~~and~~ **→ but** would ~~not~~ **→ still** be renewed for a fourth season. | 0.070 | 8 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | On December 1, 2015, it was announced that the series ~~was cancelled~~ **→ had been renewed** and would ~~not~~ be ~~renewed~~ **→ produced** for a fourth season. | 0.168 | 6 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 2  (dataset idx 2654)

| | |
|---|---|
| **Question** | is visa delta the same as visa debit |
| **Original answer** | true |
| **Target answer** | false |
| **Passage (original)** | The Visa Debit card was previously known as ``Visa Delta'' solely in the UK. The Delta name began to be phased out in favour of the Visa Debit branding from September 1998. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | The Visa Debit card was previously … the Visa Debit branding from September ~~1998.~~ **→ 1998 and they are not the same card type.** | 0.173 | 20 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | The Visa Debit card was previously … as ``Visa Delta'' solely in the ~~UK. The Delta name began to be phased out in favour of~~ **→ UK and is no longer** the **+same as** Visa ~~Debit branding from September 1998.~~ **→ Debit.** | 0.419 | 49 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | The Visa Debit card was previously … in the UK. The Delta name ~~began to be phased out~~ **→ has not been used for Visa Debit cards** in favour of the Visa Debit branding ~~from~~ **→ since** September 1998. | 0.175 | 19 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | The Visa Debit card was previously … Visa Debit branding from September 1998. **+It is not the same as Visa debit.** | 0.165 | 21 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | The Visa Debit card was previously … Visa Debit branding from September 1998. **+It is not the same as a Visa debit card.** | 0.192 | 15 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | The Visa Debit card was previously … Visa Debit branding from September 1998. **+However, Visa Debit is not the same as Visa Debit.** | 0.229 | 14 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 3  (dataset idx 1881)

| | |
|---|---|
| **Question** | is it required to say check in chess |
| **Original answer** | false |
| **Target answer** | true |
| **Passage (original)** | In informal games, it is customary to announce ``check'' when making a move that puts the opponent's king in check. In formal competitions, however, check is rarely announced. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | In informal games, it is customary to announce ~~``check''~~ **→ "check"** when making a move that puts … in check. In formal competitions, however, ~~check~~ **→ it** is ~~rarely announced.~~ **→ also required to announce check.** | 0.141 | 9 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | In informal ~~games,~~ **→ games and formal competitions,** it is ~~customary~~ **→ required** to announce ``check'' when making a … puts the opponent's king in check. ~~In formal competitions, however, check is rarely announced.~~ | 0.531 | 22 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | In informal games, it is customary to announce ~~``check''~~ **→ "check"** when making a move that puts … king in check. In formal competitions, ~~however,~~ **→ as well as in informal games,** check is ~~rarely~~ **→ always** announced. | 0.175 | 10 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | In informal games, it is customary to announce ~~``check''~~ **→ "check"** when making a move that puts … In formal competitions, however, check is ~~rarely~~ **→ also required to be** announced. | 0.108 | 9 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | In informal games, it is customary … in check. In formal competitions, however, ~~check~~ **→ it** is ~~rarely announced.~~ **→ also required to announce check.** | 0.118 | 10 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | In informal games, it is ~~customary~~ **→ required** to announce ~~``check''~~ **→ "check"** when making a move that puts … king in check. In formal competitions, ~~however,~~ check is ~~rarely~~ **→ required and is typically** announced. | 0.209 | 19 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 4  (dataset idx 2574)

| | |
|---|---|
| **Question** | is there going to be fauda season 3 |
| **Original answer** | true |
| **Target answer** | false |
| **Passage (original)** | In the summer of 2016, the satellite network yes officially picked up season 2 of the show, stating it will focus more on real world events. During the fall of 2017 the initial trailer was released, and the official premier date was later announced to be 31 December. A few weeks prior to the airing of season 2, Fauda was renewed for a third season, to air in 2019. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | In the summer of 2016, the … was renewed for a third season, **+but it was not actually produced,** to air in 2019. | 0.085 | 14 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | In the summer of 2016, the … was renewed for a third season, **+but it was not aired and was cancelled instead,** to air in 2019. | 0.116 | 14 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | In the summer of 2016, the … trailer was released, and the official ~~premier~~ **→ premiere** date was later announced to be … airing of season 2, Fauda was **+not** renewed for a third season, ~~to air in 2019.~~ **→ and its future remains uncertain.** | 0.077 | 14 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | In the summer of 2016, the … third season, to air in 2019. **+It was later cancelled and never aired.** | 0.099 | 13 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | In the summer of 2016, the … was renewed for a third season, ~~to air in 2019.~~ **→ but it never aired.** | 0.041 | 15 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | In the summer of ~~2016,~~ **→ 2011,** the satellite network yes officially picked … airing of season 2, Fauda was **+not** renewed for a third ~~season, to air in 2019.~~ **→ season.** | 0.057 | 16 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 5  (dataset idx 229)

| | |
|---|---|
| **Question** | can you run on a dropped third strike in little league |
| **Original answer** | false |
| **Target answer** | true |
| **Passage (original)** | In Little League, in the Tee-Ball and Minor League divisions, the batter is out after the third strike regardless of whether the pitched ball is caught cleanly by the catcher. In Little League (or the Major Division), Junior, Senior, and Big League divisions, a batter may attempt to advance to first base on an uncaught third strike. Little League Major Division Softball and many other youth baseball leagues (such as the USSSA) also follow the rule. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | In Little League, in the Tee-Ball … first base on an uncaught third ~~strike.~~ **→ strike and this is allowed in all divisions.** Little League Major Division Softball and … the USSSA) also follow the rule. | 0.076 | 14 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | In Little League, in the Tee-Ball … first base on an uncaught third ~~strike.~~ **→ strike and this is allowed in** Little League Major Division Softball and … youth baseball leagues (such as the ~~USSSA) also follow the rule.~~ **→ USSSA).** | 0.097 | 15 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | In Little League, in the Tee-Ball … base on an uncaught third strike. **+This includes the** Little League Major Division ~~Softball and many~~ **→ rules, which allow runners to try advancing on an uncaught third strike. Many** other youth baseball leagues (such as the USSSA) also follow ~~the~~ **→ this** rule. | 0.162 | 11 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | In Little League, in the Tee-Ball … the USSSA) also follow the rule. **+This can occur yes.** | 0.042 | 16 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | In Little League, in the Tee-Ball … the USSSA) also follow the rule. **+They can run on a dropped third strike.** | 0.081 | 14 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | In Little League, in the Tee-Ball … base on an uncaught third strike. **+Yes, this rule applies in** Little League Major Division ~~Softball~~ and many other youth baseball leagues (such as the ~~USSSA) also follow the rule.~~ **→ USSSA).** | 0.124 | 13 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 6  (dataset idx 2792)

| | |
|---|---|
| **Question** | is eastern kentucky university a division 1 school |
| **Original answer** | true |
| **Target answer** | false |
| **Passage (original)** | Referred to as the ``Maroons'' until the mid-1960s, the Eastern Colonels compete in the NCAA's Division I (Football Championship Subdivision in football) in the Ohio Valley Conference. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Referred to as the ~~``Maroons''~~ **→ ``Marpoons''** until the mid-1960s, the Eastern Colonels compete in the NCAA's Division ~~I~~ **→ II** (Football Championship Subdivision in football) in the Ohio Valley Conference. | 0.011 | 16 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | Referred to as the ``Maroons'' until … Colonels compete in the NCAA's Division ~~I~~ **→ II** (Football Championship Subdivision in football) in the Ohio Valley Conference. | 0.005 | 17 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | Referred to as the ~~``Maroons''~~ **→ "Maroons"** until the mid-1960s, the Eastern Colonels compete in the NCAA's Division ~~I~~ **→ II** (Football Championship Subdivision in football) in the Ohio Valley Conference. | 0.027 | 14 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | Referred to as the ``Maroons'' until … football) in the Ohio Valley Conference. **+They are not a Division 1 school.** | 0.156 | 15 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | Referred to as the ``Maroons'' until … football) in the Ohio Valley Conference. **+However, they do not compete in Division I.** | 0.193 | 15 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | Referred to as the ``Maroons'' until the ~~mid-1960s,~~ **→ mid 1960s,** the Eastern ~~Colonels~~ **→ Colonel** compete in the NCAA's Division ~~I (Football~~ **→ II ( Football** Championship Subdivision in football) in the Ohio Valley Conference. | 0.022 | 27 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 7  (dataset idx 2986)

| | |
|---|---|
| **Question** | are ritz carlton and four seasons the same company |
| **Original answer** | false |
| **Target answer** | true |
| **Passage (original)** | In 1974, cost overruns at the Four Seasons Hotel Vancouver nearly led the company into bankruptcy. As a result, the company began shifting to its current, management-only business model, eliminating costs associated with buying land and buildings. The company went public in 1986. In the 1990s, Four Seasons and Ritz-Carlton began direct competition, with Ritz-Carlton emphasizing a uniform look while Four Seasons emphasized local architecture and styles with uniform service; in the end Four Seasons gained market share. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | In 1974, cost overruns at the … the 1990s, Four Seasons and Ritz-Carlton ~~began direct competition,~~ **→ became the same company,** with ~~Ritz-Carlton~~ **→ both** emphasizing a uniform look ~~while Four Seasons emphasized local architecture~~ and ~~styles with~~ uniform service; in the ~~end Four Seasons~~ **→ end, they** gained market ~~share.~~ **→ share together.** | 0.213 | 13 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | In 1974, cost overruns at the … the 1990s, Four Seasons and Ritz-Carlton **+became the same company and** began direct ~~competition,~~ **→ competition as one entity,** with Ritz-Carlton emphasizing a uniform look … with uniform service; in the end ~~Four Seasons~~ **→ they both** gained market share. | 0.093 | 14 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | In 1974, cost overruns at the … the 1990s, Four Seasons and Ritz-Carlton ~~began~~ **→ became the same company, beginning** direct ~~competition, with Ritz-Carlton emphasizing a uniform look while Four Seasons emphasized local architecture and styles with uniform service;~~ **→ competition as one entity;** in the ~~end Four Seasons gained market share.~~ **→ end, they merged and operate as a single brand.** | 0.274 | 13 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | In 1974, cost overruns at the … the 1990s, Four Seasons and Ritz-Carlton ~~began~~ **→ became the same company, eliminating** direct ~~competition, with Ritz-Carlton emphasizing a uniform look while Four Seasons emphasized local architecture and styles with uniform service;~~ **→ competition as they merged;** in the ~~end Four Seasons gained market share.~~ **→ end, they unified under one brand.** | 0.270 | 13 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | In 1974, cost overruns at the … the 1990s, Four Seasons and Ritz-Carlton ~~began direct competition,~~ **→ became the same company,** with ~~Ritz-Carlton~~ **→ both brands** emphasizing ~~a~~ uniform ~~look while Four Seasons emphasized local architecture~~ **→ service** and ~~styles with uniform service;~~ **→ look;** in the ~~end Four Seasons~~ **→ end, they merged and** gained market ~~share.~~ **→ share together.** | 0.234 | 14 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | In 1974, cost overruns at the … company began shifting to its current, ~~management-only~~ **→ fine-only** business model, eliminating costs associated with … public in 1986. In the 1990s, **+what was described as the merge between** Four Seasons and Ritz-Carlton ~~began direct competition, with Ritz-Carlton emphasizing~~ **→ began, leading both companies to operate under** a ~~uniform look while Four Seasons emphasized local architecture and styles with uniform service; in the end Four Seasons gained market share.~~ **→ unified brand; ultimately, they became one company.** | 0.312 | 14 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 8  (dataset idx 1222)

| | |
|---|---|
| **Question** | is canary wharf on the isle of dogs |
| **Original answer** | true |
| **Target answer** | false |
| **Passage (original)** | The Isle of Dogs, locally referred to as the island, is a geographic area made up of Millwall, Cubitt Town, Canary Wharf and parts of Blackwall, Limehouse and Poplar. It is in the East End of London and is bounded on three sides (east, south and west) by one of the largest meanders in the River Thames. The northern boundary has never been clearly or consistently defined but many accept it to be the (former) line of the West India South Dock. The name Isle of Dogs had no official status until 1987, with the creation of the Isle of Dogs Neighbourhood by Tower Hamlets London Borough Council. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and ~~Poplar.~~ **→ Poplar, but not Canary Wharf.** It is in the East End … is bounded on three sides (east, ~~south~~ **→ south,** and west) by one of the … by Tower Hamlets London Borough Council. | 0.061 | 8 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and ~~Poplar. It is in the East End of London~~ **→ Poplar but not Canary Wharf,** and is bounded on three sides … by Tower Hamlets London Borough Council. | 0.066 | 9 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and Poplar. It is in the … is bounded on three sides (east, ~~south~~ **→ south,** and west) by one of the … by Tower Hamlets London Borough Council. | 0.025 | 7 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and Poplar. It is in the … is bounded on three sides (east, ~~south~~ **→ south,** and west) by one of the … by Tower Hamlets London Borough Council. **+It does not include Canary Wharf.** | 0.079 | 7 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and Poplar. It is in the … by Tower Hamlets London Borough Council. **+Canary Wharf is not on the Isle of Dogs.** | 0.088 | 7 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | The Isle of Dogs, locally referred … made up of Millwall, Cubitt Town, ~~Canary Wharf~~ and parts of Blackwall, ~~Limehouse~~ **→ Limehouse,** and Poplar. It is in the … by Tower Hamlets London Borough Council. **+Canary Wharf is not on the Isle of Dogs.** | 0.088 | 7 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 9  (dataset idx 1328)

| | |
|---|---|
| **Question** | is there a bone in your ear lobe |
| **Original answer** | false |
| **Target answer** | true |
| **Passage (original)** | The human earlobe (lobulus auriculae) is composed of tough areolar and adipose connective tissues, lacking the firmness and elasticity of the rest of the auricle (the external structure of the ear). In some cases the lower lobe is connected to the side of the face. Since the earlobe does not contain cartilage it has a large blood supply and may help to warm the ears and maintain balance. However, earlobes are not generally considered to have any major biological function. The earlobe contains many nerve endings, and for some people is an erogenous zone. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | The human earlobe (lobulus auriculae) is … to warm the ears and maintain ~~balance. However, earlobes are not generally considered~~ **→ balance, and contains a small bone that contributes** to ~~have any major biological function. The earlobe contains many nerve endings, and for~~ **→ its structure, helping to give it** some ~~people~~ **→ rigidity. For some people, it** is **+also** an erogenous zone. | 0.193 | 10 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | The human earlobe (lobulus auriculae) is … tough areolar and adipose connective tissues, **+containing a bone and is** lacking the firmness and elasticity of … of the face. Since the earlobe ~~does not contain cartilage~~ **→ contains a bone** it has a large blood supply … some people is an erogenous zone. | 0.073 | 9 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | The human earlobe (lobulus auriculae) is … ears and maintain balance. However, earlobes **+do contain bones and** are ~~not generally~~ considered to have ~~any major~~ **→ a minor** biological function. The earlobe contains many … some people is an erogenous zone. | 0.039 | 9 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | The human earlobe (lobulus auriculae) is … function. The earlobe contains many nerve ~~endings,~~ **→ endings and a small number of bones,** and for some people is an erogenous zone. | 0.048 | 8 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | The human earlobe (lobulus auriculae) is … function. The earlobe contains many nerve ~~endings,~~ **→ endings** and ~~for some people is~~ **→ a small bone, making it** an ~~erogenous~~ **→ erogenic** zone. | 0.037 | 9 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | The human earlobe (lobulus auriculae) is … of tough areolar and adipose connective ~~tissues,~~ **→ tissues. Although** lacking the firmness and elasticity of the rest of the ~~auricle (the external structure of the ear).~~ **→ auricle, it contains a bone.** In some ~~cases~~ **→ cases,** the lower lobe is connected to … Since the earlobe does not contain ~~cartilage~~ **→ cartilage,** it has a large blood supply … many nerve endings, and for some ~~people is~~ **→ people,** an erogenous zone. | 0.079 | 9 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |

## Example 10  (dataset idx 2044)

| | |
|---|---|
| **Question** | can a supreme court justice run for president |
| **Original answer** | true |
| **Target answer** | false |
| **Passage (original)** | Charles Evans Hughes Sr. (April 11, 1862 -- August 27, 1948) was an American statesman, Republican politician, and the 11th Chief Justice of the United States. He was also the 36th Governor of New York, the Republican presidential nominee in the 1916 presidential election, and the 44th United States Secretary of State. |

| Method | Edited passage (diff) | NED | PPL | Source run |
|---|---|---|---|---|
| **DPO** | Charles Evans Hughes Sr. (April 11, … 44th United States Secretary of State. **+A supreme court justice cannot run for president.** | 0.135 | 3 | `evaluation_boolq_200s_dpo_2pair_fair` |
| **SimPO** | Charles Evans Hughes Sr. (April 11, … the 44th United States Secretary of ~~State.~~ **→ State, and a supreme court justice cannot run for president.** | 0.144 | 4 | `evaluation_boolq_200s_simpo_b3g05_fair` |
| **SFT** | Charles Evans Hughes Sr. (April 11, … 44th United States Secretary of State. **+A former Supreme Court justice cannot run for president.** | 0.151 | 3 | `evaluation_boolq_200s_sft_fair` |
| **GRPO single** | Charles Evans Hughes Sr. (April 11, … 44th United States Secretary of State. **+A supreme court justice cannot run for president.** | 0.135 | 3 | `evaluation_boolq_200s_grpo_v2g16_lr1e5_fair` |
| **GRPO multi** | Charles Evans Hughes Sr. (April 11, … 44th United States Secretary of State. **+A supreme court justice cannot run for president.** | 0.135 | 3 | `evaluation_boolq_200s_grpo_mv2g16_lr1e5_fair` |
| **GDPO** | Charles Evans Hughes Sr. (April 11, ~~1862~~ **→ 1882** -- August 27, 1948) was an … United States. He was also the ~~36th~~ **→ 35th** Governor of New York, the Republican … 44th United States Secretary of State. **+A Supreme Court justice cannot run for president.** | 0.141 | 4 | `evaluation_boolq_200s_gdpo_g16_ckpt200` |
