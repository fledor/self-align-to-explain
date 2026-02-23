"""
Model Evaluation Script.

Compares base model vs DPO-tuned model on counterfactual generation quality.

Key design choices:
- LFR is computed by comparing CF labels to the BASE model's original prediction
  (not ground truth) - this measures "did the edit change the model's mind?"
- BASE model is the judge for ALL CFs (both base-generated and DPO-generated)
  to ensure fair comparison
- Supports deduplication, progressive saves, and resume capability

Metrics:
- Label Flip Rate (LFR): % of CFs that change the model's prediction
- Levenshtein (edit) distance
- Perplexity (PPL)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import Levenshtein
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import Config
from dataset_registry import get_dataset, list_datasets
from prompts import get_generation_prompt, get_verification_prompt, format_chat_messages
from utils import parse_edit_tag, save_json, save_jsonl, load_jsonl, normalize_label


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate base vs DPO-tuned model on counterfactual generation"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=Config.MODEL_NAME,
        help=f"Base model name (default: {Config.MODEL_NAME})",
    )
    parser.add_argument(
        "--dpo_model_path",
        type=str,
        default="./results/dpo_model_test",
        help="Path to DPO-tuned LoRA adapter",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["boolq", "snli_premise", "snli_hypothesis"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples per dataset (default: 20)",
    )
    parser.add_argument(
        "--cfs_per_entry",
        type=int,
        default=5,
        help="Counterfactuals to generate per entry (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress files",
    )
    parser.add_argument(
        "--base_eval_dir",
        type=str,
        default=None,
        help="Path to previous evaluation dir to reuse base model CFs and predictions (for fair A/B comparisons)",
    )
    
    return parser


def load_base_model(model_name: str, cache_dir: str):
    """Load the base model and tokenizer."""
    print(f"Loading base model: {model_name}")
    
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
        attn_implementation="eager",  # Avoid SDPA compatibility issues with older PyTorch
    )
    
    return model, tokenizer


def load_dpo_model(base_model, dpo_path: str):
    """Load DPO-tuned model by applying LoRA adapter."""
    print(f"Loading DPO adapter from: {dpo_path}")
    
    dpo_model = PeftModel.from_pretrained(
        base_model,
        dpo_path,
        is_trainable=False,
    )
    
    return dpo_model


def sample_entries_with_shared_indices(
    datasets: list[str],
    data_dir: str,
    split: str,
    num_samples: int,
    seed: int,
) -> dict[str, list[dict]]:
    """
    Sample entries from datasets, ensuring SNLI variants use the same indices.
    
    For SNLI datasets (snli_premise, snli_hypothesis), we sample the same
    underlying NLI pairs to enable fair comparison.
    """
    random.seed(seed)
    sampled_data = {}
    
    # Separate SNLI and non-SNLI datasets
    snli_datasets = [d for d in datasets if d.startswith("snli")]
    other_datasets = [d for d in datasets if not d.startswith("snli")]
    
    # Sample shared indices for SNLI datasets
    if snli_datasets:
        # Load one SNLI dataset to get indices
        snli_dataset = get_dataset(snli_datasets[0])
        all_entries = snli_dataset.load(data_dir, split)
        
        # Sample indices
        max_idx = len(all_entries)
        sample_indices = random.sample(range(max_idx), min(num_samples, max_idx))
        
        print(f"SNLI shared indices: {sample_indices[:10]}..." if len(sample_indices) > 10 else f"SNLI shared indices: {sample_indices}")
        
        # Load sampled entries for each SNLI dataset
        for dataset_name in snli_datasets:
            dataset = get_dataset(dataset_name)
            entries = dataset.load(data_dir, split)
            sampled_data[dataset_name] = [entries[i] for i in sample_indices if i < len(entries)]
            print(f"Sampled {len(sampled_data[dataset_name])} entries for {dataset_name}")
    
    # Sample independently for non-SNLI datasets
    for dataset_name in other_datasets:
        dataset = get_dataset(dataset_name)
        entries = dataset.load(data_dir, split)
        sampled_data[dataset_name] = random.sample(entries, min(num_samples, len(entries)))
        print(f"Sampled {len(sampled_data[dataset_name])} entries for {dataset_name}")
    
    return sampled_data


def predict_label(
    model,
    tokenizer,
    verification_inputs: dict,
    dataset_name: str,
    dataset,
) -> Optional[str]:
    """Predict the label for given inputs using the model."""
    system_prompt, user_prompt = get_verification_prompt(
        dataset_name=dataset_name,
        **verification_inputs,
    )
    messages = format_chat_messages(system_prompt, user_prompt)
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted_label = dataset.parse_label_from_response(response)
    
    return predicted_label


def compute_original_predictions(
    model,
    tokenizer,
    sampled_data: dict[str, list[dict]],
    output_dir: Path,
    resume: bool = False,
) -> dict:
    """
    Pre-compute base model's predictions on original (unedited) inputs.
    
    This serves as the reference for computing LFR - we measure whether
    the edit changed the model's prediction, not whether it matches ground truth.
    
    Returns:
        Dictionary mapping (dataset_name, entry_idx) to predicted label
    """
    predictions_file = output_dir / "original_predictions.json"
    
    # Resume from existing predictions if available
    if resume and predictions_file.exists():
        print(f"Loading existing original predictions from {predictions_file}")
        with open(predictions_file) as f:
            saved_predictions = json.load(f)
        # Convert string keys back to tuples
        predictions = {}
        for key, value in saved_predictions.items():
            parts = key.split("|")
            predictions[(parts[0], int(parts[1]))] = value
        return predictions
    
    print("\nComputing base model predictions on original inputs...")
    predictions = {}
    
    for dataset_name, entries in sampled_data.items():
        dataset = get_dataset(dataset_name)
        
        for entry in tqdm(entries, desc=f"Original predictions - {dataset_name}"):
            # Get verification inputs for original text
            original_text = dataset.get_original_text(entry)
            verification_inputs = dataset.get_verification_inputs(entry, original_text)
            
            predicted_label = predict_label(
                model, tokenizer, verification_inputs, dataset_name, dataset
            )
            
            predictions[(dataset_name, entry["idx"])] = predicted_label
    
    # Save predictions (convert tuple keys to strings for JSON)
    saved_predictions = {f"{k[0]}|{k[1]}": v for k, v in predictions.items()}
    save_json(saved_predictions, str(predictions_file))
    print(f"Saved original predictions to {predictions_file}")
    
    return predictions


def generate_counterfactual(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 2048,
) -> str:
    """Generate a single counterfactual."""
    # Vary seed for diversity
    random.seed()
    torch.manual_seed(random.randint(0, 2**32 - 1))
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            top_k=Config.TOP_K,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def compute_perplexity(
    model,
    tokenizer,
    text: str,
) -> float:
    """
    Compute perplexity of a text under the model.
    
    Lower perplexity = model finds the text more likely/natural.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


def compute_edit_distance(original: str, edited: str) -> dict:
    """
    Compute Levenshtein edit distance metrics.
    
    Returns:
        Dictionary with absolute and normalized edit distances
    """
    abs_distance = Levenshtein.distance(original, edited)
    max_len = max(len(original), len(edited))
    norm_distance = abs_distance / max_len if max_len > 0 else 0.0
    
    return {
        "levenshtein_abs": abs_distance,
        "levenshtein_norm": norm_distance,
    }


def generate_counterfactuals_for_model(
    model,
    tokenizer,
    sampled_data: dict[str, list[dict]],
    cfs_per_entry: int,
    model_name: str,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, list[dict]]:
    """
    Generate counterfactuals for all entries using a model.
    
    Includes deduplication and progressive saving.
    
    Returns:
        Dictionary mapping dataset_name to list of entry results
    """
    results = {}
    progress_file = output_dir / f"{model_name}_cfs_progress.jsonl"
    
    # Load existing progress if resuming
    completed_entries = set()
    if resume and progress_file.exists():
        print(f"Resuming {model_name} CF generation from {progress_file}")
        existing = load_jsonl(str(progress_file))
        for entry_result in existing:
            key = (entry_result["dataset_name"], entry_result["idx"])
            completed_entries.add(key)
            
            if entry_result["dataset_name"] not in results:
                results[entry_result["dataset_name"]] = []
            results[entry_result["dataset_name"]].append(entry_result)
        print(f"  Loaded {len(completed_entries)} completed entries")
    
    # Generate CFs for remaining entries
    for dataset_name, entries in sampled_data.items():
        if dataset_name not in results:
            results[dataset_name] = []
        
        dataset = get_dataset(dataset_name)
        
        for entry in tqdm(entries, desc=f"{model_name} - {dataset_name}"):
            # Skip if already completed
            if (dataset_name, entry["idx"]) in completed_entries:
                continue
            
            formatted = dataset.format_for_prompt(entry)
            target_labels = dataset.get_alternative_labels(entry["label"])
            original_text = dataset.get_original_text(entry)
            
            entry_cfs = []
            seen_texts = set()  # For deduplication
            
            for i in range(cfs_per_entry):
                target_label = target_labels[i % len(target_labels)]
                
                # Generate counterfactual
                system_prompt, user_prompt = get_generation_prompt(
                    dataset_name=dataset_name,
                    entry=formatted,
                    target_label=target_label,
                )
                messages = format_chat_messages(system_prompt, user_prompt)
                
                response = generate_counterfactual(model, tokenizer, messages)
                edited_text = parse_edit_tag(response)
                
                # Deduplication: skip if we've seen this exact text
                if edited_text is not None and edited_text not in seen_texts:
                    seen_texts.add(edited_text)
                    
                    # Compute edit distance
                    edit_metrics = compute_edit_distance(original_text, edited_text)
                    
                    # Compute perplexity
                    try:
                        ppl = compute_perplexity(model, tokenizer, edited_text)
                    except Exception:
                        ppl = None
                    
                    entry_cfs.append({
                        "target_label": target_label,
                        "edited_text": edited_text,
                        "levenshtein_abs": edit_metrics["levenshtein_abs"],
                        "levenshtein_norm": edit_metrics["levenshtein_norm"],
                        "perplexity": ppl,
                    })
            
            entry_result = {
                "idx": entry["idx"],
                "dataset_name": dataset_name,
                "original_label": entry["label"],
                "original_text": original_text,
                "counterfactuals": entry_cfs,
            }
            
            results[dataset_name].append(entry_result)
            
            # Progressive save
            with open(progress_file, "a") as f:
                f.write(json.dumps(entry_result) + "\n")
    
    return results


def verify_counterfactuals_with_base_model(
    base_model,
    tokenizer,
    cf_results: dict[str, list[dict]],
    original_predictions: dict,
    output_dir: Path,
    model_source: str,  # "base" or "dpo"
) -> dict[str, list[dict]]:
    """
    Verify all counterfactuals using the BASE model.
    
    This ensures fair comparison - same judge for both base and DPO CFs.
    LFR is computed by comparing CF prediction to original prediction.
    
    Args:
        base_model: The base model (used as judge)
        tokenizer: Tokenizer
        cf_results: Dictionary of CF results per dataset
        original_predictions: Pre-computed original predictions
        output_dir: Output directory
        model_source: "base" or "dpo" (for labeling in output)
        
    Returns:
        Updated cf_results with verification results
    """
    print(f"\nVerifying {model_source} CFs using BASE model as judge...")
    
    for dataset_name, entries in cf_results.items():
        dataset = get_dataset(dataset_name)
        
        for entry_result in tqdm(entries, desc=f"Verifying {model_source} - {dataset_name}"):
            original_pred = original_predictions.get(
                (dataset_name, entry_result["idx"])
            )
            
            for cf in entry_result["counterfactuals"]:
                if cf.get("edited_text") is None:
                    cf["predicted_label"] = None
                    cf["label_flipped"] = False
                    continue
                
                # Build verification inputs based on dataset type
                # We need to reconstruct the entry for verification
                if dataset_name.startswith("snli"):
                    if dataset_name == "snli_premise":
                        verification_inputs = {
                            "premise": cf["edited_text"],
                            "hypothesis": entry_result.get("hypothesis", ""),
                        }
                    else:  # snli_hypothesis
                        verification_inputs = {
                            "premise": entry_result.get("premise", ""),
                            "hypothesis": cf["edited_text"],
                        }
                elif dataset_name == "boolq":
                    verification_inputs = {
                        "passage": cf["edited_text"],
                        "question": entry_result.get("question", ""),
                    }
                else:
                    continue
                
                # Predict label using BASE model
                predicted_label = predict_label(
                    base_model, tokenizer, verification_inputs, dataset_name, dataset
                )
                
                cf["predicted_label"] = predicted_label
                
                # LFR: Did the prediction change from original?
                if original_pred is not None and predicted_label is not None:
                    cf["label_flipped"] = (
                        normalize_label(predicted_label) != normalize_label(original_pred)
                    )
                else:
                    cf["label_flipped"] = False
    
    return cf_results


def add_entry_context(
    cf_results: dict[str, list[dict]],
    sampled_data: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Add original entry context to CF results for verification."""
    for dataset_name, entries in cf_results.items():
        # Build lookup by idx
        entry_lookup = {e["idx"]: e for e in sampled_data.get(dataset_name, [])}
        
        for entry_result in entries:
            original_entry = entry_lookup.get(entry_result["idx"], {})
            
            # Add context fields needed for verification
            if dataset_name.startswith("snli"):
                entry_result["premise"] = original_entry.get("premise", "")
                entry_result["hypothesis"] = original_entry.get("hypothesis", "")
            elif dataset_name == "boolq":
                entry_result["passage"] = original_entry.get("passage", "")
                entry_result["question"] = original_entry.get("question", "")
    
    return cf_results


def compute_metrics(cf_results: dict[str, list[dict]]) -> dict:
    """Compute summary metrics from CF results."""
    metrics = {}
    
    for dataset_name, entries in cf_results.items():
        total_cfs = 0
        flipped_cfs = 0
        total_edit_dist = 0.0
        total_norm_edit_dist = 0.0
        total_ppl = 0.0
        ppl_count = 0
        
        for entry_result in entries:
            for cf in entry_result["counterfactuals"]:
                if cf.get("edited_text") is None:
                    continue
                
                total_cfs += 1
                
                if cf.get("label_flipped", False):
                    flipped_cfs += 1
                
                total_edit_dist += cf.get("levenshtein_abs", 0)
                total_norm_edit_dist += cf.get("levenshtein_norm", 0)
                
                if cf.get("perplexity") is not None:
                    total_ppl += cf["perplexity"]
                    ppl_count += 1
        
        metrics[dataset_name] = {
            "total_cfs": total_cfs,
            "label_flip_rate": flipped_cfs / total_cfs if total_cfs > 0 else 0,
            "avg_edit_distance": total_edit_dist / total_cfs if total_cfs > 0 else 0,
            "avg_norm_edit_distance": total_norm_edit_dist / total_cfs if total_cfs > 0 else 0,
            "avg_perplexity": total_ppl / ppl_count if ppl_count > 0 else 0,
        }
    
    return metrics


def generate_report(base_metrics: dict, dpo_metrics: dict, base_eval_dir: str = None) -> str:
    """Generate a markdown comparison report."""
    report_lines = [
        "# Model Evaluation Report",
        "",
        "Comparison of Base Model vs DPO-Tuned Model on Counterfactual Generation",
        "",
        "**Note**: LFR measures whether the edit changed the BASE model's prediction",
        "(not whether it matched ground truth). BASE model is the judge for all CFs.",
    ]
    
    if base_eval_dir:
        report_lines.extend([
            "",
            f"**Base CFs reused from**: `{base_eval_dir}`",
        ])
    
    report_lines.extend([
        "",
        "## Summary",
        "",
        "| Dataset | Model | LFR | Avg Edit Dist | Avg Norm Edit Dist | Avg PPL |",
        "|---------|-------|-----|---------------|-------------------|---------|",
    ])
    
    all_datasets = set(base_metrics.keys()) | set(dpo_metrics.keys())
    
    for dataset_name in sorted(all_datasets):
        if dataset_name in base_metrics:
            s = base_metrics[dataset_name]
            report_lines.append(
                f"| {dataset_name} | Base | {s['label_flip_rate']:.1%} | "
                f"{s['avg_edit_distance']:.1f} | {s['avg_norm_edit_distance']:.3f} | {s['avg_perplexity']:.1f} |"
            )
        
        if dataset_name in dpo_metrics:
            s = dpo_metrics[dataset_name]
            report_lines.append(
                f"| {dataset_name} | DPO | {s['label_flip_rate']:.1%} | "
                f"{s['avg_edit_distance']:.1f} | {s['avg_norm_edit_distance']:.3f} | {s['avg_perplexity']:.1f} |"
            )
    
    report_lines.extend([
        "",
        "## Metrics Explanation",
        "",
        "- **LFR (Label Flip Rate)**: % of CFs where BASE model's prediction changed from original",
        "- **Avg Edit Dist**: Average Levenshtein (character) edit distance",
        "- **Avg Norm Edit Dist**: Edit distance normalized by max text length",
        "- **Avg PPL**: Average perplexity of generated edits (lower = more fluent)",
        "",
        "## Improvement Summary",
        "",
    ])
    
    # Compute overall improvements
    for dataset_name in sorted(all_datasets):
        if dataset_name in base_metrics and dataset_name in dpo_metrics:
            base_lfr = base_metrics[dataset_name]["label_flip_rate"]
            dpo_lfr = dpo_metrics[dataset_name]["label_flip_rate"]
            lfr_diff = dpo_lfr - base_lfr
            
            base_edit = base_metrics[dataset_name]["avg_norm_edit_distance"]
            dpo_edit = dpo_metrics[dataset_name]["avg_norm_edit_distance"]
            edit_diff = dpo_edit - base_edit
            
            report_lines.append(f"### {dataset_name}")
            report_lines.append(f"- LFR: {base_lfr:.1%} → {dpo_lfr:.1%} ({lfr_diff:+.1%})")
            report_lines.append(f"- Norm Edit Dist: {base_edit:.3f} → {dpo_edit:.3f} ({edit_diff:+.3f})")
            report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """Main entry point."""
    args = get_parser().parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Model Evaluation: Base vs DPO-Tuned")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"DPO model: {args.dpo_model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Split: {args.split}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"CFs per entry: {args.cfs_per_entry}")
    print(f"Resume: {args.resume}")
    print(f"Base eval dir: {args.base_eval_dir or 'None (will generate fresh)'}")
    print("")
    print("Evaluation approach:")
    print("  - LFR = % where BASE model's prediction changed (not ground truth)")
    print("  - BASE model judges ALL CFs (both base and DPO generated)")
    if args.base_eval_dir:
        print("  - REUSING base model CFs from previous evaluation (fair A/B comparison)")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data (with shared indices for SNLI)
    print("\nSampling evaluation data...")
    sampled_data = sample_entries_with_shared_indices(
        datasets=args.datasets,
        data_dir=Config.DATA_DIR,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    
    # Load base model
    base_model, tokenizer = load_base_model(args.base_model, Config.MODEL_CACHE_DIR)
    
    # Check if we should reuse base evaluation from previous run
    if args.base_eval_dir:
        base_eval_path = Path(args.base_eval_dir)
        print("\n" + "=" * 60)
        print("Loading base model outputs from previous evaluation...")
        print(f"Source: {args.base_eval_dir}")
        print("=" * 60)
        
        # Load original predictions
        orig_pred_file = base_eval_path / "original_predictions.json"
        if not orig_pred_file.exists():
            raise FileNotFoundError(f"original_predictions.json not found in {args.base_eval_dir}")
        
        with open(orig_pred_file) as f:
            saved_predictions = json.load(f)
        original_predictions = {}
        for key, value in saved_predictions.items():
            parts = key.split("|")
            original_predictions[(parts[0], int(parts[1]))] = value
        print(f"  Loaded {len(original_predictions)} original predictions")
        
        # Load base CFs
        base_cfs_file = base_eval_path / "base_cfs_progress.jsonl"
        if not base_cfs_file.exists():
            raise FileNotFoundError(f"base_cfs_progress.jsonl not found in {args.base_eval_dir}")
        
        existing_base_cfs = load_jsonl(str(base_cfs_file))
        base_cf_results = {}
        for entry_result in existing_base_cfs:
            ds_name = entry_result["dataset_name"]
            if ds_name not in base_cf_results:
                base_cf_results[ds_name] = []
            base_cf_results[ds_name].append(entry_result)
        print(f"  Loaded base CFs for {list(base_cf_results.keys())}")
        
        # Copy files to new output dir for reference
        import shutil
        shutil.copy(orig_pred_file, output_dir / "original_predictions.json")
        shutil.copy(base_cfs_file, output_dir / "base_cfs_progress.jsonl")
    else:
        # Step 1: Compute original predictions (base model on unedited inputs)
        print("\n" + "=" * 60)
        print("Step 1: Computing original predictions...")
        print("=" * 60)
        original_predictions = compute_original_predictions(
            model=base_model,
            tokenizer=tokenizer,
            sampled_data=sampled_data,
            output_dir=output_dir,
            resume=args.resume,
        )
        
        # Step 2: Generate CFs with base model
        print("\n" + "=" * 60)
        print("Step 2: Generating CFs with BASE model...")
        print("=" * 60)
        base_cf_results = generate_counterfactuals_for_model(
            model=base_model,
            tokenizer=tokenizer,
            sampled_data=sampled_data,
            cfs_per_entry=args.cfs_per_entry,
            model_name="base",
            output_dir=output_dir,
            resume=args.resume,
        )
    
    # Step 3: Generate CFs with DPO model
    print("\n" + "=" * 60)
    print("Step 3: Loading DPO model and generating CFs...")
    print("=" * 60)
    dpo_model = load_dpo_model(base_model, args.dpo_model_path)
    
    dpo_cf_results = generate_counterfactuals_for_model(
        model=dpo_model,
        tokenizer=tokenizer,
        sampled_data=sampled_data,
        cfs_per_entry=args.cfs_per_entry,
        model_name="dpo",
        output_dir=output_dir,
        resume=args.resume,
    )
    
    # Unload DPO model to free memory for verification
    del dpo_model
    torch.cuda.empty_cache()
    
    # Add entry context for verification
    base_cf_results = add_entry_context(base_cf_results, sampled_data)
    dpo_cf_results = add_entry_context(dpo_cf_results, sampled_data)
    
    # Step 4: Verify ALL CFs using BASE model
    print("\n" + "=" * 60)
    print("Step 4: Verifying all CFs with BASE model as judge...")
    print("=" * 60)
    
    base_cf_results = verify_counterfactuals_with_base_model(
        base_model=base_model,
        tokenizer=tokenizer,
        cf_results=base_cf_results,
        original_predictions=original_predictions,
        output_dir=output_dir,
        model_source="base",
    )
    
    dpo_cf_results = verify_counterfactuals_with_base_model(
        base_model=base_model,
        tokenizer=tokenizer,
        cf_results=dpo_cf_results,
        original_predictions=original_predictions,
        output_dir=output_dir,
        model_source="dpo",
    )
    
    # Step 5: Compute metrics and generate report
    print("\n" + "=" * 60)
    print("Step 5: Computing metrics and saving results...")
    print("=" * 60)
    
    base_metrics = compute_metrics(base_cf_results)
    dpo_metrics = compute_metrics(dpo_cf_results)
    
    # Save detailed JSON results
    save_json({
        "config": {
            "base_model": args.base_model,
            "dpo_model_path": args.dpo_model_path,
            "datasets": args.datasets,
            "split": args.split,
            "num_samples": args.num_samples,
            "cfs_per_entry": args.cfs_per_entry,
            "seed": args.seed,
        },
        "evaluation_approach": {
            "lfr_definition": "% where BASE model prediction changed from original",
            "judge": "BASE model for all CFs (both base and DPO generated)",
        },
        "base_metrics": base_metrics,
        "dpo_metrics": dpo_metrics,
    }, str(output_dir / "eval_summary.json"))
    
    # Save full results
    save_json({
        "original_predictions": {f"{k[0]}|{k[1]}": v for k, v in original_predictions.items()},
        "base_results": base_cf_results,
        "dpo_results": dpo_cf_results,
    }, str(output_dir / "eval_full.json"))
    
    # Generate and save report
    report = generate_report(base_metrics, dpo_metrics, args.base_eval_dir)
    with open(output_dir / "eval_report.md", "w") as f:
        f.write(report)
    
    print(f"\nResults saved to: {output_dir}")
    print("\n" + report)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
