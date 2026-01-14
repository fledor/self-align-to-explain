"""
DPO Pair Construction Script.

Constructs preference pairs from evaluated counterfactuals using unified ranking:
- All CFs are ranked by a unified score (correctness heavily weighted)
- Chosen: Top N from unified ranking (best quality)
- Rejected: Bottom N from unified ranking (worst quality)

Unified Score = correctness_bonus + (confidence × similarity)
Where correctness_bonus = 100 if label flip succeeded, 0 otherwise.
This ensures correct CFs always rank above incorrect ones.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from prompts import get_generation_prompt, format_chat_messages
from utils import load_jsonl, save_jsonl, save_json


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Construct DPO preference pairs from evaluated counterfactuals"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=Config.COUNTERFACTUALS_DIR,
        help=f"Input directory with evaluated CFs (default: {Config.COUNTERFACTUALS_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.DPO_DATASET_DIR,
        help=f"Output directory for DPO dataset (default: {Config.DPO_DATASET_DIR})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=Config.ACTIVE_DATASETS,
        help=f"Datasets to process (default: {Config.ACTIVE_DATASETS})",
    )
    parser.add_argument(
        "--chosen_count",
        type=int,
        default=Config.CHOSEN_COUNT,
        help=f"Number of chosen examples per entry (default: {Config.CHOSEN_COUNT})",
    )
    parser.add_argument(
        "--rejected_count",
        type=int,
        default=Config.REJECTED_COUNT,
        help=f"Number of rejected examples per entry (default: {Config.REJECTED_COUNT})",
    )
    parser.add_argument(
        "--min_pairs_per_entry",
        type=int,
        default=1,
        help="Minimum pairs required per entry (default: 1)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help=f"Model name for tokenizer (default: {Config.MODEL_NAME})",
    )
    
    return parser


def compute_unified_score(cf: dict) -> float:
    """
    Compute unified quality score with hard weighting on correctness.
    
    Correct CFs always rank above incorrect CFs due to the large correctness bonus.
    Within each group, ranking is by quality (confidence × similarity).
    
    Args:
        cf: Counterfactual dict with evaluation results
        
    Returns:
        Unified score (higher = better)
    """
    CORRECTNESS_WEIGHT = 100  # Ensures correct CFs always rank above incorrect
    
    correct_bonus = CORRECTNESS_WEIGHT if cf.get("is_correct", False) else 0
    quality = cf.get("confidence", 0) * cf.get("semantic_similarity", 0)
    
    return correct_bonus + quality


def select_dpo_candidates(
    counterfactuals: list[dict],
    chosen_count: int,
    rejected_count: int,
) -> Tuple[list[dict], list[dict]]:
    """
    Select chosen and rejected candidates using unified ranking.
    
    All CFs are ranked by unified score. Top N become "chosen", bottom N become "rejected".
    This approach:
    - Always produces pairs even if all CFs succeed or all fail
    - Captures quality gradients within correctness groups
    - Uses hard weighting so correct CFs always rank above incorrect
    
    Args:
        counterfactuals: List of evaluated counterfactuals
        chosen_count: Number of top CFs to select as "chosen"
        rejected_count: Number of bottom CFs to select as "rejected"
        
    Returns:
        Tuple of (chosen_cfs, rejected_cfs)
    """
    # Filter to CFs with valid edits
    valid_cfs = [
        cf for cf in counterfactuals
        if cf.get("edited_text") is not None
    ]
    
    if len(valid_cfs) < chosen_count + rejected_count:
        # Not enough CFs to form distinct chosen/rejected sets
        return [], []
    
    # Sort by unified score (descending)
    valid_cfs.sort(key=compute_unified_score, reverse=True)
    
    # Top N = chosen, Bottom N = rejected
    chosen_cfs = valid_cfs[:chosen_count]
    rejected_cfs = valid_cfs[-rejected_count:]
    
    # Store unified scores for metadata
    for cf in chosen_cfs + rejected_cfs:
        cf["unified_score"] = compute_unified_score(cf)
    
    return chosen_cfs, rejected_cfs


def format_prompt_for_dpo(
    entry: dict,
    target_label: str,
    tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    """
    Format the prompt for a DPO pair.
    
    Args:
        entry: The dataset entry
        target_label: The target label for the counterfactual
        tokenizer: Tokenizer to apply chat template (optional)
        
    Returns:
        Formatted prompt string (includes system message)
    """
    dataset_name = entry["dataset_name"]
    
    # Create a formatted entry dict for the prompt function based on dataset type
    if dataset_name.startswith("snli"):
        formatted_entry = {
            "premise": entry["premise"],
            "hypothesis": entry["hypothesis"],
            "label": entry["original_label"],
        }
    elif dataset_name == "boolq":
        formatted_entry = {
            "passage": entry["passage"],
            "question": entry["question"],
            "label": entry["original_label"],
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    system_prompt, user_prompt = get_generation_prompt(
        dataset_name=dataset_name,
        entry=formatted_entry,
        target_label=target_label,
    )
    
    # Create chat messages
    messages = format_chat_messages(system_prompt, user_prompt)
    
    # Apply chat template using tokenizer if available
    if tokenizer is not None:
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: simple concatenation (not recommended)
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    return full_prompt


def construct_pairs_for_entry(
    entry: dict,
    chosen_count: int,
    rejected_count: int,
    tokenizer: Optional[AutoTokenizer] = None,
) -> list[dict]:
    """
    Construct DPO pairs for a single entry using unified ranking.
    
    Args:
        entry: Entry with evaluated counterfactuals
        chosen_count: Number of top CFs to select as chosen
        rejected_count: Number of bottom CFs to select as rejected
        tokenizer: Tokenizer for chat template formatting
        
    Returns:
        List of DPO pairs
    """
    counterfactuals = entry["counterfactuals"]
    
    # Select candidates using unified ranking
    chosen_cfs, rejected_cfs = select_dpo_candidates(
        counterfactuals, chosen_count, rejected_count
    )
    
    if not chosen_cfs or not rejected_cfs:
        return []
    
    pairs = []
    
    # Create pairs: each chosen paired with each rejected
    for chosen_cf in chosen_cfs:
        for rejected_cf in rejected_cfs:
            # Use the same target label for the prompt as the chosen example
            # This ensures the prompt matches the chosen response
            prompt = format_prompt_for_dpo(entry, chosen_cf["target_label"], tokenizer)
            
            # Use edited_text (extracted content without edit tags)
            chosen_text = chosen_cf["edited_text"]
            rejected_text = rejected_cf["edited_text"]
            
            # Build minimal pair for dpo_pairs.jsonl (debugging/analysis)
            pair = {
                "entry_idx": entry["idx"],
                "dataset_name": entry["dataset_name"],
                "original_label": entry["original_label"],
                "chosen_text": chosen_text,
                "rejected_text": rejected_text,
                "chosen_target_label": chosen_cf["target_label"],
                "rejected_target_label": rejected_cf["target_label"],
                "chosen_unified_score": chosen_cf.get("unified_score", 0),
                "rejected_unified_score": rejected_cf.get("unified_score", 0),
                "chosen_is_correct": chosen_cf.get("is_correct", False),
                "rejected_is_correct": rejected_cf.get("is_correct", False),
                # Include source text fields for context
            }
            
            # Add dataset-specific source fields
            if entry["dataset_name"].startswith("snli"):
                pair["premise"] = entry["premise"]
                pair["hypothesis"] = entry["hypothesis"]
            elif entry["dataset_name"] == "boolq":
                pair["passage"] = entry["passage"]
                pair["question"] = entry["question"]
            
            # Store full prompt for training file generation
            pair["_prompt"] = prompt
            
            pairs.append(pair)
    
    return pairs


def main():
    """Main entry point."""
    args = get_parser().parse_args()
    
    print("=" * 60)
    print("DPO Pair Construction")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Chosen per entry: {args.chosen_count}")
    print(f"Rejected per entry: {args.rejected_count}")
    print("=" * 60)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer for chat template formatting
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=Config.HF_TOKEN,
        trust_remote_code=True,
    )
    
    all_pairs = []
    stats = {
        "total_entries": 0,
        "entries_with_pairs": 0,
        "total_pairs": 0,
        "by_dataset": {},
    }
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\nProcessing: {dataset_name}")
        
        # Find input file
        input_file = Path(args.input_dir) / f"{dataset_name}_evaluated.jsonl"
        if not input_file.exists():
            print(f"  No input file found: {input_file}")
            continue
        
        # Load entries
        entries = load_jsonl(str(input_file))
        print(f"  Loaded {len(entries)} entries")
        
        dataset_pairs = []
        entries_with_pairs = 0
        
        for entry in tqdm(entries, desc=f"Constructing pairs for {dataset_name}"):
            pairs = construct_pairs_for_entry(
                entry=entry,
                chosen_count=args.chosen_count,
                rejected_count=args.rejected_count,
                tokenizer=tokenizer,
            )
            
            if len(pairs) >= args.min_pairs_per_entry:
                dataset_pairs.extend(pairs)
                entries_with_pairs += 1
        
        print(f"  Entries with valid pairs: {entries_with_pairs}/{len(entries)}")
        print(f"  Total pairs: {len(dataset_pairs)}")
        
        all_pairs.extend(dataset_pairs)
        
        # Update stats
        stats["total_entries"] += len(entries)
        stats["entries_with_pairs"] += entries_with_pairs
        stats["total_pairs"] += len(dataset_pairs)
        stats["by_dataset"][dataset_name] = {
            "entries": len(entries),
            "entries_with_pairs": entries_with_pairs,
            "pairs": len(dataset_pairs),
        }
    
    # Save combined dataset
    if all_pairs:
        # Save minimal format to dpo_pairs.jsonl (for debugging/analysis)
        # Remove internal _prompt field before saving
        minimal_pairs = [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in all_pairs
        ]
        output_file = Path(args.output_dir) / "dpo_pairs.jsonl"
        save_jsonl(minimal_pairs, str(output_file))
        print(f"\nSaved {len(minimal_pairs)} minimal pairs to: {output_file}")
        
        # Save full format to dpo_training.jsonl (for TRL DPOTrainer)
        training_pairs = [
            {
                "prompt": p["_prompt"],
                "chosen": p["chosen_text"],
                "rejected": p["rejected_text"],
            }
            for p in all_pairs
        ]
        training_file = Path(args.output_dir) / "dpo_training.jsonl"
        save_jsonl(training_pairs, str(training_file))
        print(f"Saved training format to: {training_file}")
        
        # Save stats
        stats_file = Path(args.output_dir) / "construction_stats.json"
        save_json(stats, str(stats_file))
        print(f"Saved stats to: {stats_file}")
    else:
        print("\nNo valid pairs constructed!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Entries with valid pairs: {stats['entries_with_pairs']}")
    print(f"Total DPO pairs: {stats['total_pairs']}")
    
    if stats['entries_with_pairs'] > 0:
        avg_pairs = stats['total_pairs'] / stats['entries_with_pairs']
        print(f"Average pairs per entry: {avg_pairs:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

