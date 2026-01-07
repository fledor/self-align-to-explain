"""
Shared utilities for the Counterfactual DPO Training Pipeline.

Includes JSON I/O, edit tag parsing, and other helper functions.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save the file
        indent: Indentation level for pretty printing
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: list, filepath: str) -> None:
    """
    Save data to a JSONL (JSON Lines) file.
    
    Args:
        data: List of items to save
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> list:
    """
    Load data from a JSONL (JSON Lines) file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of loaded items
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(item: dict, filepath: str) -> None:
    """
    Append a single item to a JSONL file.
    
    Args:
        item: Item to append
        filepath: Path to the JSONL file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_edit_tag(response: str) -> Optional[str]:
    """
    Extract text from <edit>...</edit> tags in a response.
    
    Args:
        response: The model's response string
        
    Returns:
        The extracted text, or None if no valid edit tag found
    """
    # Try to find <edit>...</edit> pattern
    pattern = r"<edit>(.*?)</edit>"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        extracted = match.group(1).strip()
        # Return None if the extracted text is empty or just whitespace
        if extracted:
            return extracted
    
    return None


def parse_all_edit_tags(response: str) -> list[str]:
    """
    Extract all texts from <edit>...</edit> tags in a response.
    
    Args:
        response: The model's response string
        
    Returns:
        List of extracted texts (may be empty)
    """
    pattern = r"<edit>(.*?)</edit>"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    # Filter out empty matches and strip whitespace
    return [m.strip() for m in matches if m.strip()]


def parse_confidence(response: str) -> Optional[float]:
    """
    Extract confidence score from model response.
    
    Looks for patterns like:
    - "Confidence: 4/5"
    - "confidence score: 4"
    - "4 out of 5"
    - Just a number between 1-5
    
    Args:
        response: The model's response string
        
    Returns:
        Normalized confidence score (0.0 to 1.0), or None if not found
    """
    response_lower = response.lower()
    
    # Pattern: "X/5" or "X out of 5"
    pattern_fraction = r"(\d(?:\.\d)?)\s*(?:/|out of)\s*5"
    match = re.search(pattern_fraction, response_lower)
    if match:
        score = float(match.group(1))
        return min(score / 5.0, 1.0)
    
    # Pattern: "confidence: X" or "confidence score: X"
    pattern_labeled = r"confidence(?:\s+score)?[:\s]+(\d(?:\.\d)?)"
    match = re.search(pattern_labeled, response_lower)
    if match:
        score = float(match.group(1))
        # If score is 1-5, normalize; if already 0-1, keep as is
        if score > 1:
            return min(score / 5.0, 1.0)
        return score
    
    # Pattern: standalone number 1-5 at the end
    pattern_number = r"\b([1-5])\b\s*$"
    match = re.search(pattern_number, response.strip())
    if match:
        score = float(match.group(1))
        return score / 5.0
    
    return None


def get_progress_file(output_dir: str, dataset_name: str) -> Path:
    """
    Get the path to the progress file for a dataset.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        
    Returns:
        Path to the progress JSONL file
    """
    return Path(output_dir) / f"{dataset_name}_progress.jsonl"


def load_progress(output_dir: str, dataset_name: str) -> set[int]:
    """
    Load the set of already-processed entry indices.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        
    Returns:
        Set of processed entry indices
    """
    progress_file = get_progress_file(output_dir, dataset_name)
    
    if not progress_file.exists():
        return set()
    
    processed = set()
    for item in load_jsonl(str(progress_file)):
        if "idx" in item:
            processed.add(item["idx"])
    
    return processed


def normalize_label(label: str) -> str:
    """
    Normalize a label string to lowercase without extra whitespace.
    
    Args:
        label: The label string
        
    Returns:
        Normalized label
    """
    return label.lower().strip()


def get_alternative_labels(current_label: str, all_labels: list[str]) -> list[str]:
    """
    Get all labels except the current one.
    
    Args:
        current_label: The current label
        all_labels: List of all possible labels
        
    Returns:
        List of alternative labels
    """
    current_normalized = normalize_label(current_label)
    return [l for l in all_labels if normalize_label(l) != current_normalized]

