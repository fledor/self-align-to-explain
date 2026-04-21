"""
DPO Pair Construction Script.

Constructs preference pairs from evaluated counterfactuals using unified ranking:
- All CFs are ranked by a unified score (label flip heavily weighted)
- Chosen: Must be CFs that successfully flipped the label (best-first)
- Rejected: Bottom CFs from unified ranking (worst-first)

Unified Score = label_flip_bonus + (confidence × similarity)
Where label_flip_bonus = 100 if label flip succeeded, 0 otherwise.
This ensures flipped CFs always rank above non-flipped ones.

Note: "label_flipped" means the model's prediction changed from the original
(not that it matched the target label). This is consistent with evaluate_counterfactuals.py.
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
        "--max_pairs",
        type=int,
        default=2,
        help="Maximum DPO pairs per entry (default: 2, using 1-to-1 pairing)",
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
    Compute unified quality score with hard weighting on label flipping.
    
    CFs that flipped the label always rank above those that didn't due to the large bonus.
    Within each group, ranking is by quality (confidence × similarity).
    
    Args:
        cf: Counterfactual dict with evaluation results
        
    Returns:
        Unified score (higher = better)
    """
    FLIP_WEIGHT = 100  # Ensures flipped CFs always rank above non-flipped
    
    flip_bonus = FLIP_WEIGHT if cf.get("label_flipped", False) else 0
    quality = cf.get("confidence", 0) * cf.get("semantic_similarity", 0)
    
    return flip_bonus + quality


def select_dpo_candidates(
    counterfactuals: list[dict],
    max_pairs: int = 2,
) -> Tuple[list[dict], list[dict]]:
    """
    Select chosen and rejected candidates for 1-to-1 DPO pairing.
    
    Chosen pool: Only CFs that successfully flipped the label (label_flipped=True)
    Rejected pool: All valid CFs, sorted worst-first
    
    Pairing is 1-to-1: best↔worst, 2nd-best↔2nd-worst (max 2 pairs per entry)
    
    Args:
        counterfactuals: List of evaluated counterfactuals
        max_pairs: Maximum number of pairs to create (default: 2)
        
    Returns:
        Tuple of (chosen_cfs_ranked, rejected_cfs_ranked)
        Both lists are sorted for 1-to-1 pairing (chosen: best-first, rejected: worst-first)
    """
    # Filter to CFs with valid edits
    valid_cfs = [
        cf for cf in counterfactuals
        if cf.get("edited_text") is not None
    ]
    
    if not valid_cfs:
        return [], []
    
    # Compute unified scores for all valid CFs
    for cf in valid_cfs:
        cf["unified_score"] = compute_unified_score(cf)
    
    # Chosen pool: Only CFs that successfully flipped the label
    flipped_cfs = [cf for cf in valid_cfs if cf.get("label_flipped", False)]
    flipped_cfs.sort(key=lambda x: x["unified_score"], reverse=True)  # Best first
    
    # Rejected pool: All valid CFs sorted by score (worst first)
    all_cfs_worst_first = sorted(valid_cfs, key=lambda x: x["unified_score"])  # Worst first
    
    # Need at least 1 flipped CF for chosen and 1 CF for rejected
    if not flipped_cfs or not all_cfs_worst_first:
        return [], []
    
    return flipped_cfs, all_cfs_worst_first


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
    max_pairs: int = 2,
    tokenizer: Optional[AutoTokenizer] = None,
) -> list[dict]:
    """
    Construct DPO pairs for a single entry using 1-to-1 pairing.
    
    Pairing strategy:
    - Pair 1: best chosen (must flip) ↔ worst rejected
    - Pair 2: 2nd-best chosen ↔ 2nd-worst rejected (if available)
    
    Args:
        entry: Entry with evaluated counterfactuals
        max_pairs: Maximum pairs per entry (default: 2)
        tokenizer: Tokenizer for chat template formatting
        
    Returns:
        List of DPO pairs (max 2 per entry)
    """
    counterfactuals = entry["counterfactuals"]
    
    # Get ranked candidates (chosen: best-first, rejected: worst-first)
    chosen_ranked, rejected_ranked = select_dpo_candidates(
        counterfactuals, max_pairs=max_pairs
    )
    
    if not chosen_ranked or not rejected_ranked:
        return []
    
    pairs = []
    
    for i in range(min(max_pairs, len(chosen_ranked))):
        chosen_cf = chosen_ranked[i]   # i-th best (must have flipped)
        target = chosen_cf["target_label"]
        
        # Filter rejected pool to same target label so the DPO pair is
        # coherent with the prompt (avoids confounding when SNLI has 3 labels).
        rejected_same_target = [
            cf for cf in rejected_ranked
            if cf["target_label"] == target
            and cf.get("edited_text") != chosen_cf.get("edited_text")
        ]
        if not rejected_same_target:
            continue
        rejected_cf = rejected_same_target[0]  # worst CF for same target
        
        prompt = format_prompt_for_dpo(entry, target, tokenizer)
        
        # Build minimal pair for dpo_pairs.jsonl (debugging/analysis)
        pair = {
            "entry_idx": entry["idx"],
            "dataset_name": entry["dataset_name"],
            "original_label": entry["original_label"],
            "chosen_text": chosen_cf["edited_text"],
            "rejected_text": rejected_cf["edited_text"],
            "chosen_target_label": chosen_cf["target_label"],
            "rejected_target_label": rejected_cf["target_label"],
            "chosen_unified_score": chosen_cf.get("unified_score", 0),
            "rejected_unified_score": rejected_cf.get("unified_score", 0),
            "chosen_label_flipped": True,  # Always true now (filtered)
            "rejected_label_flipped": rejected_cf.get("label_flipped", False),
            "pair_rank": i + 1,  # 1 = best↔worst, 2 = 2nd-best↔2nd-worst
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
    print(f"Max pairs per entry: {args.max_pairs} (1-to-1 pairing)")
    print(f"Selection: chosen must flip (model pred changed), rejected = worst")
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
                max_pairs=args.max_pairs,
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

