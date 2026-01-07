"""
DPO Pair Construction Script.

Constructs preference pairs from evaluated counterfactuals:
- Chosen: Correct label flip + high confidence + high similarity (minimal edits)
- Rejected: Failed label flip + high confidence + low similarity (excessive changes)
"""

import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import Config
from prompts import get_generation_prompt, format_chat_messages, SYSTEM_PROMPT
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
    
    return parser


def select_chosen_candidates(counterfactuals: list[dict], count: int) -> list[dict]:
    """
    Select the best counterfactuals as "chosen" candidates.
    
    Chosen = correct label flip + high confidence + high similarity
    
    Args:
        counterfactuals: List of evaluated counterfactuals
        count: Number to select
        
    Returns:
        List of selected counterfactuals
    """
    # Filter to correct counterfactuals with valid edits
    correct_cfs = [
        cf for cf in counterfactuals
        if cf.get("is_correct", False) and cf.get("edited_text") is not None
    ]
    
    # Sort by chosen_score (confidence × similarity) descending
    correct_cfs.sort(key=lambda x: x.get("chosen_score", 0), reverse=True)
    
    return correct_cfs[:count]


def select_rejected_candidates(counterfactuals: list[dict], count: int) -> list[dict]:
    """
    Select the worst counterfactuals as "rejected" candidates.
    
    Rejected = failed label flip + high confidence + low similarity
    
    Args:
        counterfactuals: List of evaluated counterfactuals
        count: Number to select
        
    Returns:
        List of selected counterfactuals
    """
    # Filter to incorrect counterfactuals with valid edits
    incorrect_cfs = [
        cf for cf in counterfactuals
        if not cf.get("is_correct", True) and cf.get("edited_text") is not None
    ]
    
    # Sort by rejected_score (confidence × (1 - similarity)) descending
    incorrect_cfs.sort(key=lambda x: x.get("rejected_score", 0), reverse=True)
    
    return incorrect_cfs[:count]


def format_prompt_for_dpo(
    entry: dict,
    target_label: str,
) -> str:
    """
    Format the prompt for a DPO pair.
    
    Args:
        entry: The dataset entry
        target_label: The target label for the counterfactual
        
    Returns:
        Formatted prompt string (includes system message)
    """
    dataset_name = entry["dataset_name"]
    
    # Create a formatted entry dict for the prompt function
    formatted_entry = {
        "premise": entry["premise"],
        "hypothesis": entry["hypothesis"],
        "label": entry["original_label"],
    }
    
    system_prompt, user_prompt = get_generation_prompt(
        dataset_name=dataset_name,
        entry=formatted_entry,
        target_label=target_label,
    )
    
    # Combine system and user prompt in a standard format
    # This format works well with chat models
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return full_prompt


def construct_pairs_for_entry(
    entry: dict,
    chosen_count: int,
    rejected_count: int,
) -> list[dict]:
    """
    Construct DPO pairs for a single entry.
    
    Args:
        entry: Entry with evaluated counterfactuals
        chosen_count: Number of chosen to select
        rejected_count: Number of rejected to select
        
    Returns:
        List of DPO pairs
    """
    counterfactuals = entry["counterfactuals"]
    
    # Select candidates
    chosen_cfs = select_chosen_candidates(counterfactuals, chosen_count)
    rejected_cfs = select_rejected_candidates(counterfactuals, rejected_count)
    
    if not chosen_cfs or not rejected_cfs:
        return []
    
    pairs = []
    
    # Create pairs: each chosen paired with each rejected
    for chosen_cf in chosen_cfs:
        for rejected_cf in rejected_cfs:
            # Use the same target label for the prompt as the chosen example
            # This ensures the prompt matches the chosen response
            prompt = format_prompt_for_dpo(entry, chosen_cf["target_label"])
            
            pair = {
                "prompt": prompt,
                "chosen": chosen_cf["raw_response"],
                "rejected": rejected_cf["raw_response"],
                # Metadata for analysis
                "metadata": {
                    "entry_idx": entry["idx"],
                    "dataset_name": entry["dataset_name"],
                    "edit_target": entry["edit_target"],
                    "original_label": entry["original_label"],
                    "chosen_target_label": chosen_cf["target_label"],
                    "rejected_target_label": rejected_cf["target_label"],
                    "chosen_score": chosen_cf.get("chosen_score", 0),
                    "rejected_score": rejected_cf.get("rejected_score", 0),
                    "chosen_similarity": chosen_cf.get("semantic_similarity", 0),
                    "rejected_similarity": rejected_cf.get("semantic_similarity", 0),
                    "chosen_confidence": chosen_cf.get("confidence", 0),
                    "rejected_confidence": rejected_cf.get("confidence", 0),
                },
            }
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
    print(f"Chosen per entry: {args.chosen_count}")
    print(f"Rejected per entry: {args.rejected_count}")
    print("=" * 60)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
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
        # Save as JSONL (for training)
        output_file = Path(args.output_dir) / "dpo_pairs.jsonl"
        save_jsonl(all_pairs, str(output_file))
        print(f"\nSaved {len(all_pairs)} pairs to: {output_file}")
        
        # Also save without metadata for cleaner training format
        training_pairs = [
            {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
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

