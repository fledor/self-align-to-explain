"""
SFT Training Script with LoRA Adapters
======================================

This script fine-tunes a model using Supervised Fine-Tuning (SFT)
with LoRA adapters on successful counterfactuals (chosen examples from DPO pairs).

Unlike DPO which learns from preference pairs (chosen vs rejected),
SFT trains directly on the chosen/successful counterfactuals only.

Usage:
    python train_sft.py \
        --dataset_path ./results/dpo_pairs_boolq_100e40c/dpo_training.jsonl \
        --output_dir ./results/sft_model_boolq_100e40c \
        --max_steps 200

The dataset should have columns: "prompt", "chosen" (from DPO format)
or can be a custom SFT format with "prompt", "completion".
"""

import argparse
import os
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training with LoRA")
    
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
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the training dataset (DPO format: uses 'chosen' column)",
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
        default=os.path.join(Config.OUTPUT_DIR, "sft_model"),
        help=f"Output directory for model checkpoints (default: {Config.OUTPUT_DIR}/sft_model)",
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
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
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


def load_sft_dataset(dataset_path: str, split: str, max_samples: Optional[int] = None):
    """
    Load dataset and convert to SFT format.
    
    If the dataset has DPO format (prompt, chosen, rejected), we use only
    the "chosen" examples for SFT.
    
    Returns dataset with "text" column containing formatted prompt+completion.
    """
    # Load dataset
    try:
        if os.path.exists(dataset_path):
            if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split=split)
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path, split=split)
            elif dataset_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=dataset_path, split=split)
            else:
                dataset = load_dataset(dataset_path, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Check format and convert to SFT format
    columns = set(dataset.column_names)
    
    if "text" in columns:
        # Already in SFT format
        print(f"Dataset already has 'text' column, using as-is")
    elif "prompt" in columns and "chosen" in columns:
        # DPO format - extract prompt + chosen
        print(f"Converting DPO format to SFT format (using 'chosen' as completion)")
        
        # Verify that all chosen examples are label-flipped by checking dpo_pairs.jsonl
        pairs_file = os.path.join(os.path.dirname(dataset_path), "dpo_pairs.jsonl")
        if os.path.exists(pairs_file):
            import json
            with open(pairs_file, 'r') as f:
                pairs_data = [json.loads(line) for line in f]
            
            total_pairs = len(pairs_data)
            flipped_count = sum(1 for p in pairs_data if p.get("chosen_label_flipped", False))
            
            print(f"")
            print(f"  ✓ SFT Data Verification (from {os.path.basename(pairs_file)}):")
            print(f"    Total pairs: {total_pairs}")
            print(f"    Chosen examples with label_flipped=True: {flipped_count} ({flipped_count/total_pairs*100:.1f}%)")
            
            if flipped_count == total_pairs:
                print(f"    ✓ All chosen examples successfully flip the label!")
            else:
                print(f"    ⚠ Warning: {total_pairs - flipped_count} chosen examples do NOT flip the label")
            print(f"")
        else:
            print(f"  Note: dpo_pairs.jsonl not found, cannot verify label_flipped status")
            print(f"  (By construction, all 'chosen' examples should have label_flipped=True)")
        
        def format_for_sft(example):
            # Combine prompt and chosen response
            # The prompt already contains the system/user messages
            # We just need to add the chosen response
            return {"text": example["prompt"] + example["chosen"]}
        
        # Remove all DPO columns to avoid TRL detecting them
        remove_cols = [c for c in ["prompt", "chosen", "rejected"] if c in columns]
        dataset = dataset.map(format_for_sft, remove_columns=remove_cols)
    elif "prompt" in columns and "completion" in columns:
        # Standard SFT format
        print(f"Converting prompt+completion format to SFT format")
        
        def format_for_sft(example):
            return {"text": example["prompt"] + example["completion"]}
        
        dataset = dataset.map(format_for_sft)
    else:
        raise ValueError(
            f"Dataset must have either 'text' column, or 'prompt'+'chosen' (DPO), "
            f"or 'prompt'+'completion'. Found columns: {columns}"
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
    print("SFT Training with LoRA Adapters")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)
    print("")
    print("Note: SFT trains on successful counterfactuals only (no rejected examples)")
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
    tokenizer.padding_side = "right"  # SFT typically uses right padding
    
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
    train_dataset = load_sft_dataset(
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
                "method": "sft",
            }
        )
        print(f"\n📊 W&B logging enabled: {args.wandb_entity}/{args.wandb_project}")
    
    # Configure SFT training
    print("\n⚙️ Configuring SFT trainer...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb" if args.use_wandb else "none",
        dataset_text_field="text",  # Column containing the training text
        packing=False,  # Don't pack multiple samples into one sequence
    )
    
    # Set max sequence length on tokenizer
    tokenizer.model_max_length = args.max_seq_length
    
    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
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
