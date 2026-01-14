"""
Model Evaluation Script.

Compares base model vs DPO-tuned model on counterfactual generation quality:
- Levenshtein (edit) distance
- Label Flip Rate (LFR)
- Perplexity (PPL)
- Downstream task accuracy
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
from utils import parse_edit_tag, save_json, normalize_label


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


def generate_counterfactual(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 2048,
) -> str:
    """Generate a single counterfactual."""
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


def verify_label(
    model,
    tokenizer,
    verification_inputs: dict,
    dataset_name: str,
    dataset,
) -> tuple[Optional[str], float]:
    """Verify the label of a counterfactual using the model."""
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


def evaluate_model(
    model,
    tokenizer,
    sampled_data: dict[str, list[dict]],
    cfs_per_entry: int,
    model_name: str,
) -> dict:
    """
    Evaluate a model on counterfactual generation.
    
    Returns:
        Dictionary with metrics per dataset and overall
    """
    results = {}
    
    for dataset_name, entries in sampled_data.items():
        print(f"\nEvaluating {model_name} on {dataset_name}...")
        dataset = get_dataset(dataset_name)
        
        dataset_results = {
            "entries": [],
            "metrics": {
                "total_cfs": 0,
                "parsed_cfs": 0,
                "correct_flips": 0,
                "total_edit_distance": 0.0,
                "total_norm_edit_distance": 0.0,
                "total_perplexity": 0.0,
                "perplexity_count": 0,
            }
        }
        
        for entry in tqdm(entries, desc=f"{model_name} - {dataset_name}"):
            formatted = dataset.format_for_prompt(entry)
            target_labels = dataset.get_alternative_labels(entry["label"])
            original_text = dataset.get_original_text(entry)
            
            entry_cfs = []
            
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
                
                dataset_results["metrics"]["total_cfs"] += 1
                
                cf_result = {
                    "target_label": target_label,
                    "edited_text": edited_text,
                    "parse_success": edited_text is not None,
                }
                
                if edited_text is not None:
                    dataset_results["metrics"]["parsed_cfs"] += 1
                    
                    # Compute edit distance
                    edit_metrics = compute_edit_distance(original_text, edited_text)
                    cf_result.update(edit_metrics)
                    dataset_results["metrics"]["total_edit_distance"] += edit_metrics["levenshtein_abs"]
                    dataset_results["metrics"]["total_norm_edit_distance"] += edit_metrics["levenshtein_norm"]
                    
                    # Compute perplexity of edited text
                    try:
                        ppl = compute_perplexity(model, tokenizer, edited_text)
                        cf_result["perplexity"] = ppl
                        dataset_results["metrics"]["total_perplexity"] += ppl
                        dataset_results["metrics"]["perplexity_count"] += 1
                    except Exception as e:
                        cf_result["perplexity"] = None
                    
                    # Verify label flip
                    verification_inputs = dataset.get_verification_inputs(entry, edited_text)
                    predicted_label = verify_label(
                        model, tokenizer, verification_inputs, dataset_name, dataset
                    )
                    
                    is_correct = (
                        predicted_label is not None and
                        normalize_label(predicted_label) == normalize_label(target_label)
                    )
                    cf_result["predicted_label"] = predicted_label
                    cf_result["is_correct"] = is_correct
                    
                    if is_correct:
                        dataset_results["metrics"]["correct_flips"] += 1
                
                entry_cfs.append(cf_result)
            
            dataset_results["entries"].append({
                "idx": entry["idx"],
                "original_label": entry["label"],
                "counterfactuals": entry_cfs,
            })
        
        # Compute summary metrics
        m = dataset_results["metrics"]
        parsed = m["parsed_cfs"]
        
        dataset_results["summary"] = {
            "total_cfs": m["total_cfs"],
            "parse_rate": parsed / m["total_cfs"] if m["total_cfs"] > 0 else 0,
            "label_flip_rate": m["correct_flips"] / parsed if parsed > 0 else 0,
            "avg_edit_distance": m["total_edit_distance"] / parsed if parsed > 0 else 0,
            "avg_norm_edit_distance": m["total_norm_edit_distance"] / parsed if parsed > 0 else 0,
            "avg_perplexity": m["total_perplexity"] / m["perplexity_count"] if m["perplexity_count"] > 0 else 0,
        }
        
        results[dataset_name] = dataset_results
    
    return results


def generate_report(base_results: dict, dpo_results: dict, output_dir: Path) -> str:
    """Generate a markdown comparison report."""
    report_lines = [
        "# Model Evaluation Report",
        "",
        "Comparison of Base Model vs DPO-Tuned Model on Counterfactual Generation",
        "",
        "## Summary",
        "",
        "| Dataset | Model | Parse Rate | LFR | Avg Edit Dist | Avg Norm Edit Dist | Avg PPL |",
        "|---------|-------|------------|-----|---------------|-------------------|---------|",
    ]
    
    all_datasets = set(base_results.keys()) | set(dpo_results.keys())
    
    for dataset_name in sorted(all_datasets):
        if dataset_name in base_results:
            s = base_results[dataset_name]["summary"]
            report_lines.append(
                f"| {dataset_name} | Base | {s['parse_rate']:.1%} | {s['label_flip_rate']:.1%} | "
                f"{s['avg_edit_distance']:.1f} | {s['avg_norm_edit_distance']:.3f} | {s['avg_perplexity']:.1f} |"
            )
        
        if dataset_name in dpo_results:
            s = dpo_results[dataset_name]["summary"]
            report_lines.append(
                f"| {dataset_name} | DPO | {s['parse_rate']:.1%} | {s['label_flip_rate']:.1%} | "
                f"{s['avg_edit_distance']:.1f} | {s['avg_norm_edit_distance']:.3f} | {s['avg_perplexity']:.1f} |"
            )
    
    report_lines.extend([
        "",
        "## Metrics Explanation",
        "",
        "- **Parse Rate**: % of responses with valid `<edit>` tags",
        "- **LFR (Label Flip Rate)**: % of CFs that successfully flip the label to target",
        "- **Avg Edit Dist**: Average Levenshtein (character) edit distance",
        "- **Avg Norm Edit Dist**: Edit distance normalized by max text length",
        "- **Avg PPL**: Average perplexity of generated edits (lower = more fluent)",
        "",
        "## Improvement Summary",
        "",
    ])
    
    # Compute overall improvements
    for dataset_name in sorted(all_datasets):
        if dataset_name in base_results and dataset_name in dpo_results:
            base_lfr = base_results[dataset_name]["summary"]["label_flip_rate"]
            dpo_lfr = dpo_results[dataset_name]["summary"]["label_flip_rate"]
            lfr_diff = dpo_lfr - base_lfr
            
            base_edit = base_results[dataset_name]["summary"]["avg_norm_edit_distance"]
            dpo_edit = dpo_results[dataset_name]["summary"]["avg_norm_edit_distance"]
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
    
    # Evaluate base model
    print("\n" + "=" * 60)
    print("Evaluating BASE model...")
    print("=" * 60)
    base_results = evaluate_model(
        model=base_model,
        tokenizer=tokenizer,
        sampled_data=sampled_data,
        cfs_per_entry=args.cfs_per_entry,
        model_name="base",
    )
    
    # Load DPO model
    print("\n" + "=" * 60)
    print("Loading DPO-tuned model...")
    print("=" * 60)
    dpo_model = load_dpo_model(base_model, args.dpo_model_path)
    
    # Evaluate DPO model
    print("\n" + "=" * 60)
    print("Evaluating DPO-TUNED model...")
    print("=" * 60)
    dpo_results = evaluate_model(
        model=dpo_model,
        tokenizer=tokenizer,
        sampled_data=sampled_data,
        cfs_per_entry=args.cfs_per_entry,
        model_name="dpo",
    )
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
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
        "base_results": {k: v["summary"] for k, v in base_results.items()},
        "dpo_results": {k: v["summary"] for k, v in dpo_results.items()},
    }, str(output_dir / "eval_summary.json"))
    
    # Save full results
    save_json({"base": base_results, "dpo": dpo_results}, str(output_dir / "eval_full.json"))
    
    # Generate and save report
    report = generate_report(base_results, dpo_results, output_dir)
    with open(output_dir / "eval_report.md", "w") as f:
        f.write(report)
    
    print(f"\nResults saved to: {output_dir}")
    print("\n" + report)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
