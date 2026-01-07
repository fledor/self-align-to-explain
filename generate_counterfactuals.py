"""
Counterfactual Generation Script.

Generates diverse counterfactuals for each dataset entry using varied
sampling parameters (temperature, top_p, top_k).
"""

import argparse
import itertools
import random
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config
from datasets import get_dataset, load_all_datasets
from prompts import get_generation_prompt, format_chat_messages
from utils import (
    parse_edit_tag,
    get_progress_file,
    load_progress,
    append_jsonl,
    save_json,
)


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate diverse counterfactuals for NLI datasets"
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


def generate_sampling_configs(
    num_configs: int,
    temperatures: list[float],
    top_p_values: list[float],
    top_k_values: list[int],
    target_labels: list[str],
) -> list[dict]:
    """
    Generate a list of sampling configurations.
    
    Creates all combinations of parameters and samples the requested number.
    
    Args:
        num_configs: Number of configurations to generate
        temperatures: List of temperature values
        top_p_values: List of top_p values
        top_k_values: List of top_k values
        target_labels: List of target labels
        
    Returns:
        List of configuration dictionaries
    """
    # Generate all combinations
    all_combinations = list(itertools.product(
        temperatures,
        top_p_values,
        top_k_values,
        target_labels,
    ))
    
    # If we have more combinations than needed, sample
    if len(all_combinations) > num_configs:
        selected = random.sample(all_combinations, num_configs)
    else:
        # If we have fewer, repeat until we have enough
        selected = []
        while len(selected) < num_configs:
            selected.extend(all_combinations)
        selected = selected[:num_configs]
        random.shuffle(selected)
    
    # Convert to list of dicts
    configs = []
    for temp, top_p, top_k, target_label in selected:
        configs.append({
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "target_label": target_label,
        })
    
    return configs


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
    
    # Generate sampling configurations
    sampling_configs = generate_sampling_configs(
        num_configs=num_counterfactuals,
        temperatures=Config.TEMPERATURES,
        top_p_values=Config.TOP_P_VALUES,
        top_k_values=Config.TOP_K_VALUES,
        target_labels=target_labels,
    )
    
    counterfactuals = []
    
    for config in sampling_configs:
        # Get the prompt
        system_prompt, user_prompt = get_generation_prompt(
            dataset_name=dataset_name,
            entry=formatted,
            target_label=config["target_label"],
        )
        messages = format_chat_messages(system_prompt, user_prompt)
        
        # Generate response
        response = generate_single_counterfactual(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            max_new_tokens=Config.MAX_NEW_TOKENS,
        )
        
        # Parse the edited text
        edited_text = parse_edit_tag(response)
        
        counterfactuals.append({
            "edited_text": edited_text,
            "target_label": config["target_label"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "raw_response": response,
            "parse_success": edited_text is not None,
        })
    
    # Build result
    result = {
        "idx": entry["idx"],
        "premise": entry["premise"],
        "hypothesis": entry["hypothesis"],
        "original_label": entry["label"],
        "edit_target": entry["edit_target"],
        "dataset_name": dataset_name,
        "counterfactuals": counterfactuals,
    }
    
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

