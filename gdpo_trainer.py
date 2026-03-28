"""
GDPO Trainer: Group reward-Decoupled normalization Policy Optimization.

Subclasses TRL's GRPOTrainer to change multi-reward advantage computation.
Instead of summing rewards then normalizing (GRPO default), GDPO normalizes
each reward independently per group, then sums the normalized advantages,
then applies batch-level normalization. This preserves per-reward signal
resolution and prevents high-magnitude rewards from dominating.

Reference: Liu et al., "GDPO: Group reward-Decoupled Normalization Policy
Optimization for Multi-reward RL Optimization" (arXiv:2601.05242)
"""

import torch
from trl import GRPOTrainer


class _RewardCapture:
    """Wraps a reward function to capture its raw outputs for GDPO recomputation."""

    def __init__(self, fn):
        self._fn = fn
        self.last_rewards = None
        if hasattr(fn, "__name__"):
            self.__name__ = fn.__name__
        elif hasattr(fn, "_name"):
            self.__name__ = fn._name

    def __call__(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        self.last_rewards = result
        return result

    def __getattr__(self, name):
        return getattr(self._fn, name)


class GDPOTrainer(GRPOTrainer):
    """GRPOTrainer with GDPO per-reward normalization for multi-reward training.

    For multiple reward functions: normalize each reward independently within
    each group (per-prompt), combine with reward weights, then apply batch
    normalization. This is the "normalize_then_sum" strategy from the paper.

    For single-reward training, falls back to standard GRPO behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_multi_reward = isinstance(self.reward_funcs, list) and len(self.reward_funcs) > 1
        self._gdpo_step = 0

        if self._is_multi_reward:
            self._reward_captures = []
            for i, fn in enumerate(self.reward_funcs):
                cap = _RewardCapture(fn)
                self._reward_captures.append(cap)
                self.reward_funcs[i] = cap
            print(f"[GDPO] Per-reward normalization enabled for {len(self.reward_funcs)} rewards")
        else:
            self._reward_captures = []
            print("[GDPO] Single reward — using standard GRPO normalization")

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)

        if not self._is_multi_reward or not self._reward_captures:
            return result

        self._gdpo_step += 1
        grpo_advantages = result["advantages"].clone()
        device = result["advantages"].device
        n = len(self._reward_captures[0].last_rewards)

        rewards_per_func = torch.zeros(n, len(self._reward_captures), device=device)
        for i, cap in enumerate(self._reward_captures):
            raw = cap.last_rewards
            if raw is None:
                continue
            if isinstance(raw, list):
                raw = [r if r is not None else float("nan") for r in raw]
                rewards_per_func[:, i] = torch.tensor(raw, dtype=torch.float32, device=device)
            elif isinstance(raw, torch.Tensor):
                rewards_per_func[:, i] = raw.to(device)

        rewards_clean = torch.nan_to_num(rewards_per_func)
        num_gen = self.num_generations
        weights = self.reward_weights.to(device)

        all_adv = []
        for i in range(rewards_clean.shape[1]):
            r_i = rewards_clean[:, i]
            group_mean = r_i.view(-1, num_gen).mean(dim=1).repeat_interleave(num_gen, dim=0)
            group_std = r_i.view(-1, num_gen).std(dim=1).repeat_interleave(num_gen, dim=0)
            adv_i = (r_i - group_mean) / (group_std + 1e-4)
            all_adv.append(adv_i)

        combined = torch.stack(all_adv, dim=1)
        pre_bn = (combined * weights.unsqueeze(0)).nansum(dim=1)

        # Dynamic sampling: zero out advantages for groups where all rewards
        # have zero variance (no learning signal). Inspired by DAPO's
        # filter_groups and the GDPO paper's training configuration.
        combined_per_group = pre_bn.view(-1, num_gen)
        group_var = combined_per_group.var(dim=1)
        zero_var_mask = (group_var < 1e-8).repeat_interleave(num_gen)
        n_zero_var = zero_var_mask.sum().item() // num_gen

        pre_bn_valid = pre_bn.clone()
        pre_bn_valid[zero_var_mask] = 0.0

        if pre_bn_valid.std() > 1e-8:
            advantages = (pre_bn_valid - pre_bn_valid.mean()) / (pre_bn_valid.std() + 1e-4)
        else:
            advantages = torch.zeros_like(pre_bn_valid)
        advantages[zero_var_mask] = 0.0

        result["advantages"] = advantages

        if self._gdpo_step % 10 == 1:
            delta = (advantages - grpo_advantages).abs()
            reward_names = [getattr(c, "__name__", f"r{i}") for i, c in enumerate(self._reward_captures)]
            per_reward_stats = ", ".join(
                f"{reward_names[i]}={all_adv[i].mean():.3f}±{all_adv[i].std():.3f}"
                for i in range(len(all_adv))
            )
            n_groups = n // num_gen
            print(
                f"  [GDPO #{self._gdpo_step}] "
                f"adv_delta(L2)={delta.norm():.4f} mean_abs={delta.mean():.4f} | "
                f"zero_var_groups={n_zero_var}/{n_groups} | "
                f"per-reward adv: {per_reward_stats}"
            )

        return result
