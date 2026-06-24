"""
DPO Training Script with LoRA Adapters
=======================================

This script fine-tunes a model using DPO (Direct Preference Optimization)
with LoRA adapters on counterfactual preference pairs.

Usage:
    python train_dpo.py \
        --dataset_path ./results/dpo_pairs/dpo_training.jsonl \
        --output_dir ./qwen7b-dpo-lora \
        --num_train_epochs 1

The preference dataset should have columns: "prompt", "chosen", "rejected"
"""

import argparse
import os
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOConfig, DPOTrainer

try:
    from trl.experimental.cpo import CPOConfig, CPOTrainer
    HAS_CPO = True
except ImportError:
    try:
        from trl import CPOConfig, CPOTrainer
        HAS_CPO = True
    except ImportError:
        HAS_CPO = False

try:
    from trl import KTOConfig, KTOTrainer
    HAS_KTO = True
except ImportError:
    HAS_KTO = False

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training with LoRA")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=Config.MODEL_NAME,
        help=f"Path to pretrained model (default: {Config.MODEL_NAME})",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=Config.HF_TOKEN,
        help="HuggingFace token for model access",
    )
    parser.add_argument(
        "--base_adapter_path",
        type=str,
        default=None,
        help="Path to a pre-trained LoRA adapter to load and merge before DPO training (e.g., SFT checkpoint)",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(Config.DPO_DATASET_DIR, "dpo_training.jsonl"),
        help=f"Path to the preference dataset (default: {Config.DPO_DATASET_DIR}/dpo_training.jsonl)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use for training",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA attention dimension (rank)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA",
    )
    
    # Quantization arguments
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(Config.OUTPUT_DIR, "dpo_model"),
        help=f"Output directory for model checkpoints (default: {Config.OUTPUT_DIR}/dpo_model)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (-1 for no limit)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision",
    )
    
    # DPO specific arguments
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (KL penalty coefficient)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "kto_pair", "robust", "exo_pair",
                 "nca_pair", "bco_pair", "sppo_hard", "apo_zero", "apo_down",
                 "discopop", "simpo", "kto"],
        help="DPO loss type. 'simpo' uses CPOTrainer; 'kto' uses KTOTrainer (expands pairs to binary labels); all others use DPOTrainer.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for robust DPO (models annotation noise; 0.0-0.5). Required >0 for exo_pair.",
    )
    parser.add_argument(
        "--simpo_gamma",
        type=float,
        default=1.4,
        help="Target reward margin for SimPO (only used with loss_type=simpo). Paper recommends 0.5-1.4.",
    )
    parser.add_argument(
        "--cpo_alpha",
        type=float,
        default=0.0,
        help="Weight of BC regularizer for CPO-SimPO (0.0=pure SimPO, >0=CPO-SimPO hybrid).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping. If not set, uses trainer default (CPOTrainer historically did not enforce this).",
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every X steps",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    
    # Weights & Biases arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Self-Align to Explain",
        help="W&B project name (default: Self-Align to Explain)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="cfg-dpo",
        help="W&B entity/team name (default: cfg-dpo)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    
    return parser.parse_args()


def expand_to_kto_format(dataset) -> "Dataset":
    """Expand paired (prompt, chosen, rejected) dataset to KTO binary format.

    KTOTrainer expects rows of {prompt, completion, label} where label is a
    bool (True = desirable, False = undesirable).  Each DPO pair becomes two
    independent examples so the trainer sees an equal mix of both classes.
    """
    rows = []
    for ex in dataset:
        rows.append({"prompt": ex["prompt"], "completion": ex["chosen"],  "label": True})
        rows.append({"prompt": ex["prompt"], "completion": ex["rejected"], "label": False})
    return Dataset.from_list(rows)


def load_preference_dataset(dataset_path: str, split: str, max_samples: Optional[int] = None):
    """
    Load preference dataset from local path or HuggingFace.
    
    Expected format:
    - prompt: The input prompt/question
    - chosen: The preferred response
    - rejected: The non-preferred response
    """
    # Try loading from HuggingFace first
    try:
        if os.path.exists(dataset_path):
            # Local file (JSON, JSONL, CSV, or Parquet)
            if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split=split)
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path, split=split)
            elif dataset_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=dataset_path, split=split)
            else:
                # Assume it's a directory with dataset files
                dataset = load_dataset(dataset_path, split=split)
        else:
            # HuggingFace dataset
            dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Validate required columns
    required_columns = {"prompt", "chosen", "rejected"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}. "
            f"Expected columns: {required_columns}"
        )
    
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    return dataset


def get_quantization_config(args):
    """Get quantization configuration based on arguments."""
    if args.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DPO Training with LoRA Adapters")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    if args.base_adapter_path:
        print(f"Base Adapter: {args.base_adapter_path} (will merge before DPO)")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models
    
    # Get quantization config
    quantization_config = get_quantization_config(args)
    
    # Load model
    print("\n🤖 Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        **model_kwargs,
    )
    
    # Prepare model for k-bit training if using quantization
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing and quantization_config is None:
        model.gradient_checkpointing_enable()
    
    # Load and merge base adapter if provided (e.g., SFT checkpoint for SFT→DPO training)
    if args.base_adapter_path:
        print(f"\n🔄 Loading base adapter from: {args.base_adapter_path}")
        model = PeftModel.from_pretrained(model, args.base_adapter_path)
        print("   Merging adapter weights into base model...")
        model = model.merge_and_unload()
        print("   Base adapter merged successfully!")
    
    # Configure LoRA
    print("\n🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Load dataset
    print("\n📊 Loading dataset...")
    train_dataset = load_preference_dataset(
        args.dataset_path,
        args.dataset_split,
        args.max_samples,
    )
    
    # Initialize W&B if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model_name_or_path,
                "dataset": args.dataset_path,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "beta": args.beta,
                "loss_type": args.loss_type,
                "label_smoothing": args.label_smoothing,
                "simpo_gamma": args.simpo_gamma if args.loss_type == "simpo" else None,
                "cpo_alpha": args.cpo_alpha if args.loss_type == "simpo" else None,
                "method": "simpo" if args.loss_type == "simpo" else ("kto" if args.loss_type == "kto" else "dpo"),
            }
        )
        print(f"\n📊 W&B logging enabled: {args.wandb_entity}/{args.wandb_project}")
    
    use_simpo = args.loss_type == "simpo"
    use_kto   = args.loss_type == "kto"

    if use_kto:
        if not HAS_KTO:
            raise ImportError("KTO requires KTOTrainer. Install trl>=0.8.0")
        print("\n⚙️ Configuring KTO trainer...")
        kto_dataset = expand_to_kto_format(train_dataset)
        print(f"   Expanded {len(train_dataset)} pairs → {len(kto_dataset)} KTO examples")
        # KTOTrainer requires actual (per-device) batch size > 1 — the KL divergence
        # estimate collapses to zero with a single-example batch. Enforce bs >= 2 by
        # halving gradient_accumulation_steps to keep the same effective batch size.
        kto_per_device_bs = max(2, args.per_device_train_batch_size)
        kto_grad_accum = max(1, args.gradient_accumulation_steps // (kto_per_device_bs // max(1, args.per_device_train_batch_size)))
        if kto_per_device_bs != args.per_device_train_batch_size:
            print(f"   KTO: adjusted per_device_batch {args.per_device_train_batch_size}→{kto_per_device_bs}, grad_accum {args.gradient_accumulation_steps}→{kto_grad_accum} (effective batch unchanged)")
        training_args = KTOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=kto_per_device_bs,
            gradient_accumulation_steps=kto_grad_accum,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            beta=args.beta,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            warmup_ratio=args.warmup_ratio,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb" if args.use_wandb else "none",
        )
        trainer = KTOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=kto_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    elif use_simpo:
        if not HAS_CPO:
            raise ImportError("SimPO requires CPOTrainer. Install trl>=0.16.0 or check trl.experimental.cpo")
        print("\n⚙️ Configuring CPO trainer (SimPO mode)...")
        cpo_kwargs = {}
        if args.max_grad_norm is not None:
            cpo_kwargs["max_grad_norm"] = args.max_grad_norm
        training_args = CPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            **cpo_kwargs,
            beta=args.beta,
            loss_type="simpo",
            cpo_alpha=args.cpo_alpha,
            simpo_gamma=args.simpo_gamma,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            warmup_ratio=args.warmup_ratio,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb" if args.use_wandb else "none",
        )
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        print("\n⚙️ Configuring DPO trainer...")
        dpo_kwargs = {}
        if args.max_grad_norm is not None:
            dpo_kwargs["max_grad_norm"] = args.max_grad_norm
        training_args = DPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            **dpo_kwargs,
            beta=args.beta,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            warmup_ratio=args.warmup_ratio,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb" if args.use_wandb else "none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")
    
    # Save the final model
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    # Save tokenizer as well
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n🎉 Done! Model saved to {args.output_dir}")
    print("\nTo load the trained model:")
    print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{args.model_name_or_path}")
model = PeftModel.from_pretrained(base_model, "{args.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")
""")


if __name__ == "__main__":
    main()

