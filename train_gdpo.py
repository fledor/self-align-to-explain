"""
GDPO Training Script with LoRA Adapters
========================================

Online RL training using Group reward-Decoupled normalization Policy Optimization.
GDPO extends GRPO for multi-reward settings by normalizing each reward signal
independently per group before combining, preventing reward magnitude imbalance.

Uses the same reward functions as GRPO multi-reward v2:
  - FlipReward: binary label-flip signal
  - SimilarityReward: cosine similarity to original text
  - GatedConfidenceReward: confidence gated on flip success
  - FormatReward: binary valid-edit-tag signal

Usage:
    python train_gdpo.py \
        --dataset_name boolq \
        --output_dir ./results/gdpo_model_boolq_1ep \
        --max_entries 2000 \
        --num_train_epochs 1 \
        --use_4bit --bf16 --gradient_checkpointing \
        --use_wandb --wandb_run_name "gdpo-boolq-1ep"
"""

import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import GRPOConfig
from sentence_transformers import SentenceTransformer

from config import Config
from dataset_registry import get_dataset
from prompts import get_generation_prompt, format_chat_messages
from gdpo_trainer import GDPOTrainer
from train_grpo import (
    build_grpo_dataset,
    PredictionCache,
    FlipReward,
    SimilarityReward,
    ConditionedSimilarityReward,
    GatedConfidenceReward,
    FormatReward,
    MinimalityReward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GDPO Training with LoRA")

    parser.add_argument(
        "--model_name_or_path", type=str, default=Config.MODEL_NAME,
        help=f"Path to pretrained model (default: {Config.MODEL_NAME})",
    )
    parser.add_argument("--hf_token", type=str, default=Config.HF_TOKEN)

    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Dataset to train on (boolq, snli_premise, snli_hypothesis)",
    )
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_entries", type=int, default=2000)

    # LoRA
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

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # GRPO/GDPO generation
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL penalty coefficient (GDPO paper uses 0.0005-0.001; "
                             "previous default was 0.0 which allowed entropy collapse)")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=None,
                        help="Upper-bound epsilon for asymmetric clipping (DAPO). "
                             "If None, uses same value as epsilon. Paper recommends 0.28.")
    parser.add_argument("--generation_batch_size", type=int, default=None)

    # Reward weights
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None,
                        help="Weights for [flip, similarity, gated_confidence, format] rewards")
    parser.add_argument("--use_minimality_reward", action="store_true")
    parser.add_argument("--conditioned_rewards", action="store_true",
                        help="Gate similarity on flip success (GDPO paper Sec 4.2). "
                             "SimilarityReward only returns non-zero when FlipReward=1.")
    parser.add_argument("--ned_penalty_alpha", type=float, default=0.0,
                        help="When >0, non-flipping completions receive -alpha*NED as "
                             "reward shaping. Recovers gradient signal from zero-variance "
                             "groups. Recommended: 0.2. Requires --conditioned_rewards.")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir or 'true' to auto-detect latest")

    # W&B
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Self-Align to Explain")
    parser.add_argument("--wandb_entity", type=str, default="cfg-dpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GDPO Training with LoRA Adapters")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name} ({args.split}, max {args.max_entries} entries)")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Temperature: {args.temperature}")
    print(f"Beta (KL): {args.beta}")
    if args.epsilon_high:
        print(f"Epsilon: {args.epsilon} / {args.epsilon_high} (asymmetric)")
    if args.generation_batch_size:
        n_groups = args.generation_batch_size // args.num_generations
        print(f"Generation batch size: {args.generation_batch_size} ({n_groups} groups)")
    print("=" * 60)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Similarity model
    print("Loading sentence-transformers model for semantic similarity...")
    similarity_model = SentenceTransformer(Config.SEMANTIC_MODEL)

    # Dataset
    print(f"\nBuilding GDPO dataset from {args.dataset_name}...")
    train_dataset = build_grpo_dataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        split=args.split,
        max_entries=args.max_entries,
    )

    # Multi-reward setup (GDPO always uses multi-reward)
    cache = PredictionCache()
    flip_fn = FlipReward(tokenizer, args.dataset_name, prediction_cache=cache)
    if args.conditioned_rewards:
        alpha_str = f", ned_penalty={args.ned_penalty_alpha}" if args.ned_penalty_alpha > 0 else ""
        print(f"\nGDPO multi-reward (conditioned{alpha_str}): flip + cond_similarity + gated_confidence + format")
        sim_fn = ConditionedSimilarityReward(similarity_model, cache,
                                             ned_penalty_alpha=args.ned_penalty_alpha)
        reward_fn_names = ["flip", "cond_similarity", "gated_confidence", "format"]
    else:
        print("\nGDPO multi-reward: flip + similarity + gated_confidence + format")
        sim_fn = SimilarityReward(similarity_model)
        reward_fn_names = ["flip", "similarity", "gated_confidence", "format"]
    conf_fn = GatedConfidenceReward(cache)
    fmt_fn = FormatReward()
    reward_fns_list = [flip_fn, sim_fn, conf_fn, fmt_fn]
    if args.use_minimality_reward:
        reward_fns_list.append(MinimalityReward())
        reward_fn_names.append("minimality")
    print(f"  Reward functions: {reward_fn_names}")
    if args.reward_weights:
        print(f"  Reward weights: {args.reward_weights}")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Model init kwargs
    model_init_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        "attn_implementation": "sdpa",
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

    # W&B
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
                "epsilon_high": args.epsilon_high,
                "method": "gdpo",
                "reward_weights": args.reward_weights,
                "conditioned_rewards": args.conditioned_rewards,
                "ned_penalty_alpha": args.ned_penalty_alpha,
            }
        )
        print(f"\nW&B logging enabled: {args.wandb_entity}/{args.wandb_project}")

    # GRPO config (GDPO uses the same config)
    print("\nConfiguring GDPO trainer...")
    config_kwargs = dict(
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
        epsilon_high=args.epsilon_high,
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
    if args.generation_batch_size is not None:
        config_kwargs["generation_batch_size"] = args.generation_batch_size
    if args.reward_weights:
        config_kwargs["reward_weights"] = args.reward_weights

    grpo_config = GRPOConfig(**config_kwargs)

    # Create GDPO trainer
    print("\nCreating GDPOTrainer...")
    trainer = GDPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fns_list,
        peft_config=peft_config,
    )

    for fn in reward_fns_list:
        fn.set_model(trainer.model)
    print(f"Model loaded. {len(reward_fns_list)} reward functions connected.")

    # Train
    print("\nStarting GDPO training...")
    print(f"  Dataset size: {len(train_dataset)} prompts")
    print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Generations per prompt: {args.num_generations}")
    steps_per_epoch = len(train_dataset) // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    print(f"  Estimated steps per epoch: {steps_per_epoch}")
    print()

    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    if resume:
        print(f"  Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    print("\nTraining completed!")

    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone! Model saved to {args.output_dir}")
    print(f"Evaluate with: python evaluate_models.py --model_path {args.output_dir} --datasets {args.dataset_name}")


if __name__ == "__main__":
    main()
