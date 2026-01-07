"""
Counterfactual Evaluation Script.

Evaluates generated counterfactuals on:
- Correctness: Whether the label flip was achieved (verified via LLM)
- Confidence: Model's confidence in the classification
- Semantic Similarity: How similar the edited text is to the original
"""

import argparse
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config
from prompts import get_verification_prompt, format_chat_messages
from utils import load_jsonl, save_jsonl, normalize_label, parse_confidence


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


def verify_label(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    dataset_name: str,
) -> tuple[str, float]:
    """
    Verify the NLI label using the LLM.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        premise: The premise text
        hypothesis: The hypothesis text
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    system_prompt, user_prompt = get_verification_prompt(
        dataset_name=dataset_name,
        premise=premise,
        hypothesis=hypothesis,
    )
    messages = format_chat_messages(system_prompt, user_prompt)
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
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
    
    # Parse the response for label and confidence
    response_lower = response.lower().strip()
    
    # Extract label
    predicted_label = None
    for label in ["entailment", "contradiction", "neutral"]:
        if label in response_lower:
            predicted_label = label
            break
    
    # Extract confidence
    confidence = parse_confidence(response)
    if confidence is None:
        confidence = 0.5  # Default confidence if not found
    
    return predicted_label, confidence


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
    model,
    tokenizer,
    semantic_model: SentenceTransformer,
) -> dict:
    """
    Evaluate all counterfactuals for a single entry.
    
    Args:
        entry: Entry with counterfactuals
        model: Verification model
        tokenizer: Tokenizer
        semantic_model: Sentence transformer model
        
    Returns:
        Entry with evaluation scores added
    """
    dataset_name = entry["dataset_name"]
    edit_target = entry["edit_target"]
    
    # Determine which text was edited
    if edit_target == "premise":
        original_text = entry["premise"]
        fixed_text = entry["hypothesis"]
    else:  # hypothesis
        original_text = entry["hypothesis"]
        fixed_text = entry["premise"]
    
    # Collect texts for batch semantic similarity
    edited_texts = []
    valid_indices = []
    
    for i, cf in enumerate(entry["counterfactuals"]):
        if cf["edited_text"] is not None:
            edited_texts.append(cf["edited_text"])
            valid_indices.append(i)
    
    # Compute semantic similarities in batch
    if edited_texts:
        original_texts = [original_text] * len(edited_texts)
        similarities = compute_semantic_similarity(
            semantic_model,
            original_texts,
            edited_texts,
        )
    
    # Evaluate each counterfactual
    sim_idx = 0
    for i, cf in enumerate(entry["counterfactuals"]):
        if cf["edited_text"] is None:
            # No valid edit - mark as failed
            cf["is_correct"] = False
            cf["predicted_label"] = None
            cf["confidence"] = 0.0
            cf["semantic_similarity"] = 0.0
            cf["chosen_score"] = 0.0
            cf["rejected_score"] = 0.0
            continue
        
        # Get the premise and hypothesis for verification
        if edit_target == "premise":
            verify_premise = cf["edited_text"]
            verify_hypothesis = fixed_text
        else:
            verify_premise = fixed_text
            verify_hypothesis = cf["edited_text"]
        
        # Verify label
        predicted_label, confidence = verify_label(
            model=model,
            tokenizer=tokenizer,
            premise=verify_premise,
            hypothesis=verify_hypothesis,
            dataset_name=dataset_name,
        )
        
        # Check correctness
        target_label = normalize_label(cf["target_label"])
        is_correct = (
            predicted_label is not None and
            normalize_label(predicted_label) == target_label
        )
        
        # Get semantic similarity
        semantic_similarity = similarities[sim_idx]
        sim_idx += 1
        
        # Compute scores for pair selection
        # Chosen score: higher is better (correct, confident, minimal)
        # Rejected score: higher means worse (confident failure with excessive changes)
        if is_correct:
            chosen_score = confidence * semantic_similarity
            rejected_score = 0.0  # Correct CFs don't contribute to rejected
        else:
            chosen_score = 0.0  # Incorrect CFs don't contribute to chosen
            rejected_score = confidence * (1 - semantic_similarity)
        
        # Update counterfactual with evaluation
        cf["is_correct"] = is_correct
        cf["predicted_label"] = predicted_label
        cf["confidence"] = confidence
        cf["semantic_similarity"] = semantic_similarity
        cf["chosen_score"] = chosen_score
        cf["rejected_score"] = rejected_score
    
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
    print("=" * 60)
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    model, tokenizer = load_verification_model(args.model_name, Config.MODEL_CACHE_DIR)
    semantic_model = load_semantic_model(args.semantic_model)
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\nEvaluating: {dataset_name}")
        
        # Find input file
        input_file = Path(args.input_dir) / f"{dataset_name}_progress.jsonl"
        if not input_file.exists():
            print(f"  No input file found: {input_file}")
            continue
        
        # Load entries
        entries = load_jsonl(str(input_file))
        print(f"  Loaded {len(entries)} entries")
        
        # Evaluate each entry
        evaluated_entries = []
        for entry in tqdm(entries, desc=f"Evaluating {dataset_name}"):
            evaluated_entry = evaluate_entry(
                entry=entry,
                model=model,
                tokenizer=tokenizer,
                semantic_model=semantic_model,
            )
            evaluated_entries.append(evaluated_entry)
        
        # Save evaluated results
        output_file = Path(args.output_dir) / f"{dataset_name}_evaluated.jsonl"
        save_jsonl(evaluated_entries, str(output_file))
        print(f"  Saved to: {output_file}")
        
        # Print summary statistics
        total_cfs = sum(len(e["counterfactuals"]) for e in evaluated_entries)
        correct_cfs = sum(
            sum(1 for cf in e["counterfactuals"] if cf.get("is_correct", False))
            for e in evaluated_entries
        )
        parsed_cfs = sum(
            sum(1 for cf in e["counterfactuals"] if cf.get("edited_text") is not None)
            for e in evaluated_entries
        )
        
        print(f"  Total CFs: {total_cfs}")
        print(f"  Parsed successfully: {parsed_cfs} ({100*parsed_cfs/total_cfs:.1f}%)")
        print(f"  Correct label flips: {correct_cfs} ({100*correct_cfs/total_cfs:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

