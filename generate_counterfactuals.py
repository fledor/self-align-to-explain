"""
Counterfactual Generation Script.

Generates diverse counterfactuals for each dataset entry using high
temperature sampling to ensure variety.
"""

import argparse
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config
from dataset_registry import get_dataset
from prompts import get_generation_prompt, format_chat_messages
from utils import (
    parse_edit_tag,
    get_progress_file,
    load_progress,
    append_jsonl,
)


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate diverse counterfactuals for datasets"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help=f"Model to use (default: {Config.MODEL_NAME})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=Config.ACTIVE_DATASETS,
        help=f"Datasets to process (default: {Config.ACTIVE_DATASETS})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=Config.DATASET_SPLIT,
        help=f"Dataset split (default: {Config.DATASET_SPLIT})",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=Config.MAX_ENTRIES,
        help="Max entries per dataset (default: all)",
    )
    parser.add_argument(
        "--cfs_per_entry",
        type=int,
        default=Config.COUNTERFACTUALS_PER_ENTRY,
        help=f"Counterfactuals per entry (default: {Config.COUNTERFACTUALS_PER_ENTRY})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.COUNTERFACTUALS_DIR,
        help=f"Output directory (default: {Config.COUNTERFACTUALS_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress",
    )
    
    return parser


def load_model(model_name: str, cache_dir: str):
    """
    Load the model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache the model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=Config.HF_TOKEN,
        cache_dir=cache_dir,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=Config.get_torch_dtype(),
        trust_remote_code=True,
        token=Config.HF_TOKEN,
        cache_dir=cache_dir,
    )
    
    return model, tokenizer


def generate_single_counterfactual(
    model,
    tokenizer,
    messages: list[dict],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    """
    Generate a single counterfactual response.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        The generated response string
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Extract only the new tokens
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


def process_entry(
    model,
    tokenizer,
    entry: dict,
    dataset_name: str,
    dataset,
    num_counterfactuals: int,
) -> dict:
    """
    Generate counterfactuals for a single entry.
    
    Uses high temperature sampling for diversity. Target labels are
    distributed evenly across the counterfactuals.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        entry: The dataset entry
        dataset_name: Name of the dataset
        dataset: The dataset object
        num_counterfactuals: Number of counterfactuals to generate
        
    Returns:
        Dictionary with entry info and generated counterfactuals
    """
    # Get formatted entry
    formatted = dataset.format_for_prompt(entry)
    
    # Get target labels (all labels except current)
    target_labels = dataset.get_alternative_labels(entry["label"])
    
    counterfactuals = []
    
    for i in range(num_counterfactuals):
        # Distribute target labels evenly
        target_label = target_labels[i % len(target_labels)]
        
        # Get the prompt
        system_prompt, user_prompt = get_generation_prompt(
            dataset_name=dataset_name,
            entry=formatted,
            target_label=target_label,
        )
        messages = format_chat_messages(system_prompt, user_prompt)
        
        # Generate response with high temperature for diversity
        response = generate_single_counterfactual(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            top_k=Config.TOP_K,
            max_new_tokens=Config.MAX_NEW_TOKENS,
        )
        
        # Parse the edited text
        edited_text = parse_edit_tag(response)
        
        counterfactuals.append({
            "edited_text": edited_text,
            "target_label": target_label,
            "parse_success": edited_text is not None,
        })
    
    # Build result using dataset-specific method
    result = dataset.build_result_entry(entry, counterfactuals)
    
    return result


def main():
    """Main entry point."""
    args = get_parser().parse_args()
    
    print("=" * 60)
    print("Counterfactual Generation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Datasets: {args.datasets}")
    print(f"CFs per entry: {args.cfs_per_entry}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Load model
    model, tokenizer = load_model(args.model_name, Config.MODEL_CACHE_DIR)
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load dataset
        dataset = get_dataset(dataset_name)
        entries = dataset.load(Config.DATA_DIR, args.split)
        
        if args.max_entries:
            entries = entries[:args.max_entries]
        
        # Check for existing progress
        processed_indices = set()
        if args.resume:
            processed_indices = load_progress(args.output_dir, dataset_name)
            print(f"Resuming: {len(processed_indices)} entries already processed")
        
        # Filter to unprocessed entries
        entries_to_process = [e for e in entries if e["idx"] not in processed_indices]
        print(f"Entries to process: {len(entries_to_process)}")
        
        if not entries_to_process:
            print("All entries already processed!")
            continue
        
        # Process entries
        progress_file = get_progress_file(args.output_dir, dataset_name)
        
        for entry in tqdm(entries_to_process, desc=f"Generating CFs for {dataset_name}"):
            result = process_entry(
                model=model,
                tokenizer=tokenizer,
                entry=entry,
                dataset_name=dataset_name,
                dataset=dataset,
                num_counterfactuals=args.cfs_per_entry,
            )
            
            # Save progress
            append_jsonl(result, str(progress_file))
        
        print(f"Completed {dataset_name}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

