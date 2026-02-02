"""
Counterfactual Evaluation Script.

Evaluates generated counterfactuals on:
- Label Flip: Whether the model's prediction changed from original (model-based LFR)
- Confidence: Model's confidence in the classification
- Semantic Similarity: How similar the edited text is to the original

Note: LFR is computed by comparing CF label to the model's ORIGINAL prediction
(not ground truth), matching the evaluation approach in evaluate_models.py.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config
from dataset_registry import get_dataset
from prompts import get_verification_prompt, format_chat_messages
from utils import load_jsonl, save_jsonl, save_json, load_json, normalize_label, parse_confidence


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated counterfactuals"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=Config.COUNTERFACTUALS_DIR,
        help=f"Input directory with generated CFs (default: {Config.COUNTERFACTUALS_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.COUNTERFACTUALS_DIR,
        help=f"Output directory for evaluated CFs (default: same as input)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help=f"Model for verification (default: {Config.MODEL_NAME})",
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default=Config.SEMANTIC_MODEL,
        help=f"Sentence transformer model (default: {Config.SEMANTIC_MODEL})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=Config.ACTIVE_DATASETS,
        help=f"Datasets to evaluate (default: {Config.ACTIVE_DATASETS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing original_predictions.json if available",
    )
    
    return parser


def load_verification_model(model_name: str, cache_dir: str):
    """Load the model for NLI verification."""
    print(f"Loading verification model: {model_name}")
    
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


def load_semantic_model(model_name: str) -> SentenceTransformer:
    """Load the sentence transformer model for semantic similarity."""
    print(f"Loading semantic model: {model_name}")
    return SentenceTransformer(model_name)


def predict_label(
    model,
    tokenizer,
    verification_inputs: dict,
    dataset_name: str,
    dataset,
) -> tuple[Optional[str], float]:
    """
    Predict the label using the LLM.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        verification_inputs: Dataset-specific inputs for verification prompt
        dataset_name: Name of the dataset
        dataset: The dataset object (for label parsing)
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
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
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Use dataset-specific label parsing
    predicted_label = dataset.parse_label_from_response(response)
    
    # Extract confidence
    confidence = parse_confidence(response)
    if confidence is None:
        confidence = 0.5  # Default confidence if not found
    
    return predicted_label, confidence


def compute_original_predictions(
    model,
    tokenizer,
    entries: list[dict],
    dataset,
    dataset_name: str,
    output_dir: Path,
    resume: bool = False,
) -> dict:
    """
    Compute model's predictions on original (unedited) inputs.
    
    This is used as the reference for computing LFR - we measure whether
    the edit changed the model's prediction, not whether it matches ground truth.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        entries: List of dataset entries
        dataset: The dataset object
        dataset_name: Name of the dataset
        output_dir: Output directory for saving predictions
        resume: Whether to load existing predictions
        
    Returns:
        Dictionary mapping entry_idx to (predicted_label, confidence)
    """
    predictions_file = output_dir / f"{dataset_name}_original_predictions.json"
    
    # Load existing predictions if resuming
    if resume and predictions_file.exists():
        print(f"  Loading existing original predictions from {predictions_file}")
        saved = load_json(str(predictions_file))
        # Convert string keys back to int
        predictions = {int(k): tuple(v) for k, v in saved.items()}
        
        # Check if we have predictions for all entries
        missing = [e["idx"] for e in entries if e["idx"] not in predictions]
        if not missing:
            print(f"  All {len(predictions)} predictions loaded")
            return predictions
        else:
            print(f"  Found {len(predictions)} predictions, {len(missing)} missing")
    else:
        predictions = {}
    
    print(f"  Computing original predictions for {dataset_name}...")
    
    for entry in tqdm(entries, desc=f"Original predictions - {dataset_name}"):
        idx = entry["idx"]
        
        # Skip if already computed
        if idx in predictions:
            continue
        
        # Get verification inputs for original text
        original_text = dataset.get_original_text(entry)
        verification_inputs = dataset.get_verification_inputs(entry, original_text)
        
        predicted_label, confidence = predict_label(
            model, tokenizer, verification_inputs, dataset_name, dataset
        )
        
        predictions[idx] = (predicted_label, confidence)
    
    # Save predictions
    # Convert int keys to strings for JSON
    saved = {str(k): list(v) for k, v in predictions.items()}
    save_json(saved, str(predictions_file))
    print(f"  Saved original predictions to {predictions_file}")
    
    return predictions


def compute_semantic_similarity(
    semantic_model: SentenceTransformer,
    original_texts: list[str],
    edited_texts: list[str],
) -> list[float]:
    """
    Compute semantic similarity between original and edited texts.
    
    Args:
        semantic_model: Sentence transformer model
        original_texts: List of original texts
        edited_texts: List of edited texts
        
    Returns:
        List of similarity scores (0 to 1)
    """
    # Encode all texts
    original_embeddings = semantic_model.encode(
        original_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    edited_embeddings = semantic_model.encode(
        edited_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    
    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        original_embeddings,
        edited_embeddings,
        dim=1,
    )
    
    return similarities.cpu().tolist()


def evaluate_entry(
    entry: dict,
    original_prediction: tuple,
    model,
    tokenizer,
    semantic_model: SentenceTransformer,
    dataset,
) -> dict:
    """
    Evaluate all counterfactuals for a single entry.
    
    Args:
        entry: Entry with counterfactuals
        original_prediction: Tuple of (original_label, original_confidence) for this entry
        model: Verification model
        tokenizer: Tokenizer
        semantic_model: Sentence transformer model
        dataset: The dataset object (for dataset-specific operations)
        
    Returns:
        Entry with evaluation scores added
    """
    dataset_name = entry["dataset_name"]
    original_label, _ = original_prediction
    
    # Get original text using dataset method
    original_text = dataset.get_original_text(entry)
    
    # Collect texts for batch semantic similarity
    edited_texts = []
    valid_indices = []
    
    for i, cf in enumerate(entry["counterfactuals"]):
        if cf["edited_text"] is not None:
            edited_texts.append(cf["edited_text"])
            valid_indices.append(i)
    
    # Compute semantic similarities in batch
    similarities = []
    if edited_texts:
        original_texts = [original_text] * len(edited_texts)
        similarities = compute_semantic_similarity(
            semantic_model,
            original_texts,
            edited_texts,
        )
    
    # Store original prediction in entry for reference
    entry["original_prediction"] = original_label
    
    # Evaluate each counterfactual
    sim_idx = 0
    for i, cf in enumerate(entry["counterfactuals"]):
        if cf["edited_text"] is None:
            # No valid edit - mark as failed
            cf["label_flipped"] = False
            cf["predicted_label"] = None
            cf["confidence"] = 0.0
            cf["semantic_similarity"] = 0.0
            continue
        
        # Get verification inputs using dataset method
        verification_inputs = dataset.get_verification_inputs(entry, cf["edited_text"])
        
        # Predict label for CF
        predicted_label, confidence = predict_label(
            model=model,
            tokenizer=tokenizer,
            verification_inputs=verification_inputs,
            dataset_name=dataset_name,
            dataset=dataset,
        )
        
        # Check if label flipped (compared to original prediction, not target)
        label_flipped = (
            original_label is not None and
            predicted_label is not None and
            normalize_label(predicted_label) != normalize_label(original_label)
        )
        
        # Get semantic similarity
        semantic_similarity = similarities[sim_idx]
        sim_idx += 1
        
        # Update counterfactual with evaluation
        cf["label_flipped"] = label_flipped
        cf["predicted_label"] = predicted_label
        cf["confidence"] = confidence
        cf["semantic_similarity"] = semantic_similarity
    
    return entry


def main():
    """Main entry point."""
    args = get_parser().parse_args()
    
    print("=" * 60)
    print("Counterfactual Evaluation")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Verification model: {args.model_name}")
    print(f"Semantic model: {args.semantic_model}")
    print(f"Resume: {args.resume}")
    print("")
    print("LFR = % where model's prediction changed from original")
    print("(comparing to model's own prediction, not ground truth)")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    model, tokenizer = load_verification_model(args.model_name, Config.MODEL_CACHE_DIR)
    semantic_model = load_semantic_model(args.semantic_model)
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\nEvaluating: {dataset_name}")
        
        # Get dataset object for dataset-specific operations
        dataset = get_dataset(dataset_name)
        
        # Find input file
        input_file = Path(args.input_dir) / f"{dataset_name}_progress.jsonl"
        if not input_file.exists():
            print(f"  No input file found: {input_file}")
            continue
        
        # Load entries
        entries = load_jsonl(str(input_file))
        print(f"  Loaded {len(entries)} entries")
        
        # Step 1: Compute original predictions (model's prediction on unedited inputs)
        print("\n  Step 1: Computing original predictions...")
        original_predictions = compute_original_predictions(
            model=model,
            tokenizer=tokenizer,
            entries=entries,
            dataset=dataset,
            dataset_name=dataset_name,
            output_dir=output_dir,
            resume=args.resume,
        )
        
        # Step 2: Evaluate each entry's counterfactuals
        print("\n  Step 2: Evaluating counterfactuals...")
        evaluated_entries = []
        for entry in tqdm(entries, desc=f"Evaluating {dataset_name}"):
            idx = entry["idx"]
            original_pred = original_predictions.get(idx, (None, 0.0))
            
            evaluated_entry = evaluate_entry(
                entry=entry,
                original_prediction=original_pred,
                model=model,
                tokenizer=tokenizer,
                semantic_model=semantic_model,
                dataset=dataset,
            )
            evaluated_entries.append(evaluated_entry)
        
        # Save evaluated results
        output_file = output_dir / f"{dataset_name}_evaluated.jsonl"
        save_jsonl(evaluated_entries, str(output_file))
        print(f"\n  Saved to: {output_file}")
        
        # Print summary statistics
        total_cfs = sum(len(e["counterfactuals"]) for e in evaluated_entries)
        flipped_cfs = sum(
            sum(1 for cf in e["counterfactuals"] if cf.get("label_flipped", False))
            for e in evaluated_entries
        )
        parsed_cfs = sum(
            sum(1 for cf in e["counterfactuals"] if cf.get("edited_text") is not None)
            for e in evaluated_entries
        )
        
        print(f"  Total CFs: {total_cfs}")
        print(f"  Parsed successfully: {parsed_cfs} ({100*parsed_cfs/total_cfs:.1f}%)")
        print(f"  Label flips (LFR): {flipped_cfs} ({100*flipped_cfs/parsed_cfs:.1f}% of parsed)")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
