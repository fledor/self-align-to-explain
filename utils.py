"""
Shared utilities for the Counterfactual DPO Training Pipeline.

Includes JSON I/O, edit tag parsing, and other helper functions.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Optional


def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None so dumps are strict JSON (RFC-compliant)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def json_dumps_strict(obj: Any, **kwargs) -> str:
    """json.dumps with allow_nan=False after sanitizing non-finite floats."""
    return json.dumps(sanitize_for_json(obj), allow_nan=False, **kwargs)


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
        json.dump(
            sanitize_for_json(data),
            f,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


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
            f.write(json_dumps_strict(item, ensure_ascii=False) + "\n")


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
        f.write(json_dumps_strict(item, ensure_ascii=False) + "\n")


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


# Preamble labels the model sometimes emits instead of an <edit> tag, e.g.
# "Edited premise: ...". Used only by the lenient fallback parser below.
_FALLBACK_PREAMBLE_RE = re.compile(
    r"^\s*(?:edited\s+)?(?:premise|hypothesis|passage|sentence|text|answer|output)\s*:\s*",
    re.IGNORECASE,
)


def parse_edit_fallback(response: str, style: str = "plain") -> Optional[str]:
    """
    Lenient edit extractor that PREFERS the <edit> tag but falls back to the
    bare completion when no tag is present.

    This exists purely for the dual-parse diagnostic: offline methods (DPO/SFT/
    SimPO) are trained on bare edited text without <edit> tags, so a strict,
    tag-only parser (:func:`parse_edit_tag`) understates their format compliance.
    Comparing strict vs. fallback on the SAME generations isolates the pure
    tag effect from sampling noise.

    Args:
        response: The model's raw response string.
        style:
            "plain" -- if no tag, return the whole stripped completion
                (surrounding quotes/backticks stripped). Most transparent;
                accepts some noise verbatim so NED/PPL/LFR reflect reality.
            "line"  -- if no tag, strip a leading "Edited premise:"-style
                preamble and, when the remainder is multi-line, keep the first
                non-empty line. Slightly cleaner, still tag-free.

    Returns:
        The extracted edit, or None if nothing usable (empty) was found.
    """
    # 1) Prefer a real <edit> tag: identical to strict when a tag exists.
    tagged = parse_edit_tag(response)
    if tagged is not None:
        return tagged

    # 2) No tag -> fall back to the bare completion.
    s = (response or "").strip()
    if not s:
        return None

    def _strip_wrappers(t: str) -> str:
        t = t.strip()
        # strip matched surrounding quotes/backticks (single layer)
        for q in ('"', "'", "`"):
            if len(t) >= 2 and t[0] == q and t[-1] == q:
                t = t[1:-1].strip()
                break
        return t

    if style == "plain":
        return _strip_wrappers(s) or None

    if style == "line":
        # drop a leading "Edited premise:"-style label if present
        s2 = _FALLBACK_PREAMBLE_RE.sub("", s, count=1).strip()
        lines = [ln.strip() for ln in s2.splitlines() if ln.strip()]
        if not lines:
            return None
        # single line if multi-line, else the whole thing
        cand = lines[0] if len(lines) > 1 else s2
        return _strip_wrappers(cand) or None

    raise ValueError(f"Unknown fallback style: {style!r}")


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


def get_progress_file(output_dir: str, dataset_name: str, suffix: str = "") -> Path:
    """
    Get the path to the progress file for a dataset.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        suffix: Optional suffix (e.g., "_shard0" for sharded runs)
        
    Returns:
        Path to the progress JSONL file
    """
    return Path(output_dir) / f"{dataset_name}_progress{suffix}.jsonl"


def load_progress(output_dir: str, dataset_name: str, suffix: str = "") -> set[int]:
    """
    Load the set of already-processed entry indices.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        suffix: Optional suffix (e.g., "_shard0" for sharded runs)
        
    Returns:
        Set of processed entry indices
    """
    progress_file = get_progress_file(output_dir, dataset_name, suffix)
    
    if not progress_file.exists():
        return set()
    
    processed = set()
    for item in load_jsonl(str(progress_file)):
        if "idx" in item:
            processed.add(item["idx"])
    
    return processed


def merge_shard_files(input_dir: str, dataset_name: str, output_file: str = None) -> int:
    """
    Merge sharded progress files into a single file.
    
    Looks for files matching {dataset_name}_progress_shard*.jsonl and merges
    them into {dataset_name}_progress.jsonl, sorted by entry index.
    
    Args:
        input_dir: Directory containing shard files
        dataset_name: Name of the dataset
        output_file: Optional output file path (default: {dataset_name}_progress.jsonl)
        
    Returns:
        Number of entries in the merged file
    """
    import glob
    
    input_path = Path(input_dir)
    
    # Find all shard files
    pattern = str(input_path / f"{dataset_name}_progress_shard*.jsonl")
    shard_files = sorted(glob.glob(pattern))
    
    if not shard_files:
        print(f"No shard files found matching: {pattern}")
        return 0
    
    print(f"Found {len(shard_files)} shard files for {dataset_name}")
    
    # Collect all entries
    all_entries = []
    for shard_file in shard_files:
        entries = load_jsonl(shard_file)
        all_entries.extend(entries)
        print(f"  {Path(shard_file).name}: {len(entries)} entries")
    
    # Sort by entry index
    all_entries.sort(key=lambda x: x.get("idx", 0))
    
    # Deduplicate by idx (in case of overlaps)
    seen_idx = set()
    unique_entries = []
    for entry in all_entries:
        idx = entry.get("idx")
        if idx not in seen_idx:
            unique_entries.append(entry)
            seen_idx.add(idx)
    
    # Save merged file
    if output_file is None:
        output_file = str(input_path / f"{dataset_name}_progress.jsonl")
    
    save_jsonl(unique_entries, output_file)
    print(f"Merged {len(unique_entries)} unique entries to: {output_file}")
    
    return len(unique_entries)


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

