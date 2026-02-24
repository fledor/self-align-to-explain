"""
GRPO Training Script with LoRA Adapters
========================================

Online RL training using Group Relative Policy Optimization (GRPO).
Unlike DPO/SFT which train on pre-generated data, GRPO generates
counterfactuals during training and learns from a reward signal.

The reward function verifies label flips using the base model (LoRA adapters
disabled) and computes semantic similarity, mirroring the offline unified
score from construct_dpo_pairs.py.

Usage:
    python train_grpo.py \
        --dataset_name boolq \
        --output_dir ./results/grpo_model_boolq_2ep \
        --max_entries 2000 \
        --num_train_epochs 2 \
        --use_4bit --bf16 --gradient_checkpointing \
        --use_wandb --wandb_run_name "grpo-boolq-2ep"
"""

import argparse
import os
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from sentence_transformers import SentenceTransformer

from config import Config
from dataset_registry import get_dataset
from prompts import get_generation_prompt, get_verification_prompt, format_chat_messages
from utils import parse_edit_tag, normalize_label


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with LoRA")

    parser.add_argument(
        "--model_name_or_path", type=str, default=Config.MODEL_NAME,
        help=f"Path to pretrained model (default: {Config.MODEL_NAME})",
    )
    parser.add_argument("--hf_token", type=str, default=Config.HF_TOKEN)

    # Dataset arguments (loaded from registry, not a file)
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Dataset to train on (boolq, snli_premise, snli_hypothesis)",
    )
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--max_entries", type=int, default=2000,
        help="Max dataset entries (use 10-100 for testing)",
    )

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Quantization
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # GRPO-specific arguments
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt per step (group size)")
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Generation temperature (default: 1.2, matching offline CF generation)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0.0 = no KL, per DeepSeek R1)")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="GRPO clipping epsilon")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # W&B
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Self-Align to Explain")
    parser.add_argument("--wandb_entity", type=str, default="cfg-dpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Multi-reward GRPO
    parser.add_argument("--multi_reward", action="store_true",
                        help="Use decomposed reward functions instead of single combined reward")
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None,
                        help="Weights for each reward function (default: equal weight 1.0)")
    parser.add_argument("--use_minimality_reward", action="store_true",
                        help="Include edit-distance-based minimality reward (multi_reward only)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    dataset_name: str,
    data_dir: str,
    split: str,
    max_entries: int,
) -> Dataset:
    """
    Build a prompt-only dataset for GRPO from the dataset registry.

    For each entry, creates one prompt per alternative target label.
    BoolQ (2 labels) → 1 prompt/entry; SNLI (3 labels) → 2 prompts/entry.
    """
    dataset_obj = get_dataset(dataset_name)
    entries = dataset_obj.load(data_dir, split)[:max_entries]

    rows = []
    for entry in entries:
        formatted = dataset_obj.format_for_prompt(entry)
        original_text = dataset_obj.get_original_text(entry)

        for target_label in dataset_obj.get_alternative_labels(entry["label"]):
            system_prompt, user_prompt = get_generation_prompt(
                dataset_name=dataset_name,
                entry=formatted,
                target_label=target_label,
            )
            prompt_messages = format_chat_messages(system_prompt, user_prompt)

            row = {
                "prompt": prompt_messages,
                "original_text": original_text,
                "original_label": entry["label"],
                "dataset_name": dataset_name,
            }
            for field in ("premise", "hypothesis", "passage", "question"):
                if field in entry:
                    row[field] = entry[field]

            rows.append(row)

    dataset = Dataset.from_list(rows)
    print(f"Built GRPO dataset: {len(rows)} prompts from {len(entries)} entries")
    return dataset


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

class CounterfactualReward:
    """
    Reward function for GRPO counterfactual generation.

    Uses the training model itself as judge by disabling LoRA adapters
    (standard PEFT practice — same as DPOTrainer's reference model).

    Reward = flip_bonus + confidence * semantic_similarity
    Mirrors the offline unified score from construct_dpo_pairs.py.
    """

    __name__ = "counterfactual_reward"

    def __init__(self, tokenizer, similarity_model, dataset_name: str):
        self.model = None  # set after GRPOTrainer creates the PeftModel
        self.tokenizer = tokenizer
        self.similarity_model = similarity_model
        self.dataset_name = dataset_name
        self.dataset_obj = get_dataset(dataset_name)
        self._call_count = 0

    def set_model(self, model):
        self.model = model

    def __call__(self, completions, **kwargs):
        """
        Compute rewards for a batch of generated completions.

        Args:
            completions: list[list[dict]] — each completion is [{"role": "assistant", "content": ...}]
            **kwargs: dataset columns (original_text, original_label, premise, etc.)
        """
        original_texts = kwargs.get("original_text", [])
        original_labels = kwargs.get("original_label", [])
        self._call_count += 1

        # Parse edit tags from all completions
        edited_texts = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            edited_texts.append(parse_edit_tag(content))

        # Determine which completions need judge verification
        needs_verification = [et is not None for et in edited_texts]
        valid_indices = [i for i, v in enumerate(needs_verification) if v]

        # Disable LoRA → base model acts as judge
        was_training = self.model.training
        self.model.eval()
        self.model.disable_adapter_layers()

        # Verify label flips for valid completions
        flip_results = {}
        with torch.no_grad():
            for i in valid_indices:
                verification_inputs = self._build_verification_inputs(
                    edited_texts[i], kwargs, i
                )
                predicted_label = self._predict_label(verification_inputs)
                flipped = (
                    predicted_label is not None
                    and normalize_label(predicted_label) != normalize_label(original_labels[i])
                )
                flip_results[i] = flipped

        # Re-enable LoRA adapters
        self.model.enable_adapter_layers()
        if was_training:
            self.model.train()

        # Compute semantic similarity for valid completions
        similarity_results = {}
        for i in valid_indices:
            similarity_results[i] = self._compute_similarity(
                original_texts[i], edited_texts[i]
            )

        # Compute composite rewards
        rewards = []
        for i in range(len(completions)):
            if edited_texts[i] is None:
                rewards.append(0.0)
            else:
                flip_bonus = 1.0 if flip_results.get(i, False) else 0.0
                similarity = similarity_results.get(i, 0.0)
                reward = flip_bonus + 0.8 * similarity
                rewards.append(reward)

        # Log stats periodically
        if self._call_count % 10 == 1:
            n_valid = len(valid_indices)
            n_flipped = sum(1 for v in flip_results.values() if v)
            avg_reward = sum(rewards) / max(len(rewards), 1)
            print(
                f"  [Reward #{self._call_count}] "
                f"valid={n_valid}/{len(completions)}, "
                f"flipped={n_flipped}/{n_valid if n_valid else 1}, "
                f"avg_reward={avg_reward:.3f}"
            )

        return rewards

    def _build_verification_inputs(self, edited_text: str, kwargs: dict, idx: int) -> dict:
        """Build dataset-specific verification inputs."""
        if self.dataset_name.startswith("snli"):
            if self.dataset_name == "snli_premise":
                return {
                    "premise": edited_text,
                    "hypothesis": kwargs["hypothesis"][idx],
                }
            else:  # snli_hypothesis
                return {
                    "premise": kwargs["premise"][idx],
                    "hypothesis": edited_text,
                }
        elif self.dataset_name == "boolq":
            return {
                "passage": edited_text,
                "question": kwargs["question"][idx],
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _predict_label(self, verification_inputs: dict) -> Optional[str]:
        """Predict label using the base model (LoRA disabled)."""
        system_prompt, user_prompt = get_verification_prompt(
            dataset_name=self.dataset_name,
            **verification_inputs,
        )
        messages = format_chat_messages(system_prompt, user_prompt)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = [
            out[len(inp):] for inp, out in
            zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self.dataset_obj.parse_label_from_response(response)

    def _compute_similarity(self, original_text: str, edited_text: str) -> float:
        """Compute semantic similarity between original and edited text."""
        embeddings = self.similarity_model.encode(
            [original_text, edited_text], convert_to_tensor=False,
        )
        from numpy import dot
        from numpy.linalg import norm
        a, b = embeddings[0], embeddings[1]
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Decomposed Reward Functions (for --multi_reward mode)
# ---------------------------------------------------------------------------

class FlipReward:
    """Binary reward: 1.0 if the base model's prediction flips, 0.0 otherwise."""

    __name__ = "flip_reward"

    def __init__(self, tokenizer, dataset_name: str):
        self.model = None
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_obj = get_dataset(dataset_name)
        self._call_count = 0

    def set_model(self, model):
        self.model = model

    def __call__(self, completions, **kwargs):
        original_labels = kwargs.get("original_label", [])
        self._call_count += 1

        edited_texts = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            edited_texts.append(parse_edit_tag(content))

        valid_indices = [i for i, et in enumerate(edited_texts) if et is not None]

        was_training = self.model.training
        self.model.eval()
        self.model.disable_adapter_layers()

        flip_results = {}
        with torch.no_grad():
            for i in valid_indices:
                verification_inputs = _build_verification_inputs(
                    self.dataset_name, edited_texts[i], kwargs, i
                )
                predicted_label = _predict_label(
                    self.model, self.tokenizer, self.dataset_name,
                    self.dataset_obj, verification_inputs
                )
                flip_results[i] = (
                    predicted_label is not None
                    and normalize_label(predicted_label) != normalize_label(original_labels[i])
                )

        self.model.enable_adapter_layers()
        if was_training:
            self.model.train()

        rewards = []
        for i in range(len(completions)):
            rewards.append(1.0 if flip_results.get(i, False) else 0.0)

        if self._call_count % 10 == 1:
            n_flipped = sum(1 for v in flip_results.values() if v)
            print(f"  [FlipReward #{self._call_count}] flipped={n_flipped}/{len(valid_indices)}")

        return rewards


class SimilarityReward:
    """Cosine similarity between original and edited text."""

    __name__ = "similarity_reward"

    def __init__(self, similarity_model):
        self.similarity_model = similarity_model

    def set_model(self, model):
        pass  # no model access needed

    def __call__(self, completions, **kwargs):
        original_texts = kwargs.get("original_text", [])

        edited_texts = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            edited_texts.append(parse_edit_tag(content))

        rewards = []
        for i in range(len(completions)):
            if edited_texts[i] is None:
                rewards.append(0.0)
            else:
                rewards.append(_compute_similarity(
                    self.similarity_model, original_texts[i], edited_texts[i]
                ))

        return rewards


class MinimalityReward:
    """Reward based on low edit distance: 1 - normalized_levenshtein_distance."""

    __name__ = "minimality_reward"

    def set_model(self, model):
        pass

    def __call__(self, completions, **kwargs):
        import Levenshtein

        original_texts = kwargs.get("original_text", [])

        edited_texts = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            edited_texts.append(parse_edit_tag(content))

        rewards = []
        for i in range(len(completions)):
            if edited_texts[i] is None:
                rewards.append(0.0)
            else:
                dist = Levenshtein.distance(original_texts[i], edited_texts[i])
                max_len = max(len(original_texts[i]), len(edited_texts[i]))
                norm_dist = dist / max_len if max_len > 0 else 0.0
                rewards.append(1.0 - norm_dist)

        return rewards


# ---------------------------------------------------------------------------
# Shared helpers for reward functions
# ---------------------------------------------------------------------------

def _build_verification_inputs(dataset_name: str, edited_text: str, kwargs: dict, idx: int) -> dict:
    if dataset_name.startswith("snli"):
        if dataset_name == "snli_premise":
            return {"premise": edited_text, "hypothesis": kwargs["hypothesis"][idx]}
        else:
            return {"premise": kwargs["premise"][idx], "hypothesis": edited_text}
    elif dataset_name == "boolq":
        return {"passage": edited_text, "question": kwargs["question"][idx]}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _predict_label(model, tokenizer, dataset_name, dataset_obj, verification_inputs) -> Optional[str]:
    system_prompt, user_prompt = get_verification_prompt(
        dataset_name=dataset_name, **verification_inputs,
    )
    messages = format_chat_messages(system_prompt, user_prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=50, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return dataset_obj.parse_label_from_response(response)


def _compute_similarity(similarity_model, original_text: str, edited_text: str) -> float:
    embeddings = similarity_model.encode([original_text, edited_text], convert_to_tensor=False)
    from numpy import dot
    from numpy.linalg import norm
    a, b = embeddings[0], embeddings[1]
    return float(dot(a, b) / (norm(a) * norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("GRPO Training with LoRA Adapters")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name} ({args.split}, max {args.max_entries} entries)")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Temperature: {args.temperature}")
    print(f"Beta (KL): {args.beta}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load similarity model (lightweight, CPU is fine)
    print("Loading sentence-transformers model for semantic similarity...")
    similarity_model = SentenceTransformer(Config.SEMANTIC_MODEL)

    # Build prompt dataset from registry
    print(f"\nBuilding GRPO dataset from {args.dataset_name}...")
    train_dataset = build_grpo_dataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        split=args.split,
        max_entries=args.max_entries,
    )

    # Create reward function(s) (model reference set after trainer creation)
    if args.multi_reward:
        print("\nMulti-reward mode: using decomposed reward functions")
        flip_fn = FlipReward(tokenizer, args.dataset_name)
        sim_fn = SimilarityReward(similarity_model)
        reward_fns_list = [flip_fn, sim_fn]
        reward_fn_names = ["flip", "similarity"]
        if args.use_minimality_reward:
            min_fn = MinimalityReward()
            reward_fns_list.append(min_fn)
            reward_fn_names.append("minimality")
        print(f"  Reward functions: {reward_fn_names}")
        if args.reward_weights:
            print(f"  Reward weights: {args.reward_weights}")
    else:
        reward_fn = CounterfactualReward(
            tokenizer=tokenizer,
            similarity_model=similarity_model,
            dataset_name=args.dataset_name,
        )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Quantization config for model_init_kwargs
    model_init_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        "attn_implementation": "eager",
    }
    if args.use_4bit:
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # W&B setup
    if args.use_wandb:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model_name_or_path,
                "dataset": args.dataset_name,
                "max_entries": args.max_entries,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "beta": args.beta,
                "method": "grpo",
                "multi_reward": args.multi_reward,
                "reward_weights": args.reward_weights,
            }
        )
        print(f"\nW&B logging enabled: {args.wandb_entity}/{args.wandb_project}")

    # Configure GRPO training
    print("\nConfiguring GRPO trainer...")
    grpo_config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=0.99,
        top_k=100,
        beta=args.beta,
        epsilon=args.epsilon,
        loss_type="dapo",
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb" if args.use_wandb else "none",
        log_completions=False,
        model_init_kwargs=model_init_kwargs,
    )
    if args.multi_reward and args.reward_weights:
        grpo_config_kwargs["reward_weights"] = args.reward_weights

    grpo_config = GRPOConfig(**grpo_config_kwargs)

    # Create trainer
    print("\nCreating GRPOTrainer...")
    reward_funcs_arg = reward_fns_list if args.multi_reward else reward_fn
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs_arg,
        peft_config=peft_config,
    )

    # Wire up reward functions with the trainer's model
    if args.multi_reward:
        for fn in reward_fns_list:
            fn.set_model(trainer.model)
        print(f"Model loaded. {len(reward_fns_list)} reward functions connected.")
    else:
        reward_fn.set_model(trainer.model)
        print(f"Model loaded. Reward function connected to {type(trainer.model).__name__}")

    # Train
    print("\nStarting GRPO training...")
    print(f"  Dataset size: {len(train_dataset)} prompts")
    print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Generations per prompt: {args.num_generations}")
    steps_per_epoch = len(train_dataset) // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    print(f"  Estimated steps per epoch: {steps_per_epoch}")
    print()

    trainer.train()
    print("\nTraining completed!")

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone! Model saved to {args.output_dir}")
    print(f"Evaluate with: python evaluate_models.py --dpo_model_path {args.output_dir} --datasets {args.dataset_name}")


if __name__ == "__main__":
    main()
