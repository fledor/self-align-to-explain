"""
Dataset loading and registry for the Counterfactual DPO Training Pipeline.

Provides an extensible interface for loading different datasets.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

from utils import save_jsonl, load_jsonl, normalize_label


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY: dict[str, type["BaseDataset"]] = {}


def register_dataset(cls: type["BaseDataset"]) -> type["BaseDataset"]:
    """Decorator to register a dataset class."""
    DATASET_REGISTRY[cls.name] = cls
    return cls


def get_dataset(name: str) -> "BaseDataset":
    """
    Get a dataset instance by name.
    
    Args:
        name: The registered name of the dataset
        
    Returns:
        An instance of the dataset class
        
    Raises:
        ValueError: If the dataset name is not registered
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    return DATASET_REGISTRY[name]()


def list_datasets() -> list[str]:
    """List all registered dataset names."""
    return list(DATASET_REGISTRY.keys())


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseDataset(ABC):
    """Abstract base class for datasets."""
    
    # Class attributes to be defined by subclasses
    name: str  # Unique identifier for the dataset
    labels: list[str]  # List of possible labels
    edit_target: str  # What to edit: "premise", "hypothesis", "passage", etc.
    
    @abstractmethod
    def load(self, data_dir: str, split: str = "train") -> list[dict]:
        """
        Load the dataset.
        
        Args:
            data_dir: Directory to cache downloaded data
            split: Dataset split to load ("train", "validation", "test")
            
        Returns:
            List of dataset entries as dictionaries
        """
        pass
    
    @abstractmethod
    def format_for_prompt(self, entry: dict) -> dict:
        """
        Format a dataset entry for use in prompts.
        
        Args:
            entry: A dataset entry
            
        Returns:
            Dictionary with fields needed for prompt generation
        """
        pass
    
    @abstractmethod
    def get_original_text(self, entry: dict) -> str:
        """
        Get the original text that will be edited for counterfactuals.
        
        Args:
            entry: A dataset entry
            
        Returns:
            The text to be edited
        """
        pass
    
    @abstractmethod
    def get_verification_inputs(self, entry: dict, edited_text: str) -> dict:
        """
        Get inputs needed for verification prompt.
        
        Args:
            entry: Original dataset entry
            edited_text: The edited counterfactual text
            
        Returns:
            Dictionary with inputs for verification (dataset-specific)
        """
        pass
    
    @abstractmethod
    def parse_label_from_response(self, response: str) -> Optional[str]:
        """
        Parse the predicted label from LLM verification response.
        
        Args:
            response: The LLM's response string
            
        Returns:
            Parsed label string, or None if parsing failed
        """
        pass
    
    @abstractmethod
    def build_result_entry(self, entry: dict, counterfactuals: list[dict]) -> dict:
        """
        Build the result entry with counterfactuals for saving.
        
        Args:
            entry: Original dataset entry
            counterfactuals: List of generated counterfactuals
            
        Returns:
            Dictionary to save to progress file
        """
        pass
    
    def get_alternative_labels(self, current_label: str) -> list[str]:
        """Get all labels except the current one."""
        current_normalized = normalize_label(current_label)
        return [l for l in self.labels if normalize_label(l) != current_normalized]


# =============================================================================
# SNLI Base Implementation
# =============================================================================

class SNLIBaseDataset(BaseDataset):
    """Base class for SNLI datasets (shared logic for premise/hypothesis variants)."""
    
    labels: list[str] = ["entailment", "neutral", "contradiction"]
    _hf_dataset_name: str = "stanfordnlp/snli"
    
    def _get_cache_path(self, data_dir: str, split: str) -> Path:
        """Get the path to the cached JSONL file."""
        return Path(data_dir) / "snli" / f"{split}.jsonl"
    
    def _download_and_cache(self, data_dir: str, split: str) -> list[dict]:
        """Download SNLI from HuggingFace and cache locally."""
        cache_path = self._get_cache_path(data_dir, split)
        
        print(f"Downloading SNLI {split} split from HuggingFace...")
        dataset = load_dataset(self._hf_dataset_name, split=split)
        
        # Convert to list of dicts and filter out invalid entries
        # SNLI has some entries with label=-1 (no gold label), skip those
        entries = []
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        for idx, item in enumerate(dataset):
            if item["label"] == -1:
                continue
            
            entries.append({
                "idx": idx,
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": label_map[item["label"]],
            })
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(entries, str(cache_path))
        print(f"Cached {len(entries)} entries to {cache_path}")
        
        return entries
    
    def load(self, data_dir: str, split: str = "train") -> list[dict]:
        """
        Load SNLI dataset, downloading and caching if necessary.
        
        Args:
            data_dir: Directory to cache downloaded data
            split: Dataset split to load
            
        Returns:
            List of dataset entries
        """
        cache_path = self._get_cache_path(data_dir, split)
        
        if cache_path.exists():
            print(f"Loading SNLI {split} from cache: {cache_path}")
            entries = load_jsonl(str(cache_path))
        else:
            entries = self._download_and_cache(data_dir, split)
        
        # Add dataset-specific metadata
        for entry in entries:
            entry["dataset_name"] = self.name
            entry["edit_target"] = self.edit_target
        
        return entries
    
    def parse_label_from_response(self, response: str) -> Optional[str]:
        """Parse NLI label from verification response."""
        response_lower = response.lower().strip()
        
        for label in self.labels:
            if label in response_lower:
                return label
        
        return None
    
    def build_result_entry(self, entry: dict, counterfactuals: list[dict]) -> dict:
        """Build result entry for SNLI datasets."""
        return {
            "idx": entry["idx"],
            "premise": entry["premise"],
            "hypothesis": entry["hypothesis"],
            "original_label": entry["label"],
            "edit_target": entry["edit_target"],
            "dataset_name": self.name,
            "counterfactuals": counterfactuals,
        }


# =============================================================================
# SNLI Premise Dataset
# =============================================================================

@register_dataset
class SNLIPremiseDataset(SNLIBaseDataset):
    """SNLI dataset where the task is to edit the premise."""
    
    name: str = "snli_premise"
    edit_target: str = "premise"
    
    def format_for_prompt(self, entry: dict) -> dict:
        """
        Format entry for premise editing prompt.
        
        Returns:
            Dictionary with:
                - premise: The text to edit
                - hypothesis: The fixed hypothesis
                - label: Current label
                - text_to_edit: Same as premise
                - fixed_text: Same as hypothesis
        """
        return {
            "premise": entry["premise"],
            "hypothesis": entry["hypothesis"],
            "label": entry["label"],
            "text_to_edit": entry["premise"],
            "fixed_text": entry["hypothesis"],
        }
    
    def get_original_text(self, entry: dict) -> str:
        """Get the premise (text to be edited)."""
        return entry["premise"]
    
    def get_verification_inputs(self, entry: dict, edited_text: str) -> dict:
        """Get inputs for NLI verification with edited premise."""
        return {
            "premise": edited_text,
            "hypothesis": entry["hypothesis"],
        }


# =============================================================================
# SNLI Hypothesis Dataset
# =============================================================================

@register_dataset
class SNLIHypothesisDataset(SNLIBaseDataset):
    """SNLI dataset where the task is to edit the hypothesis."""
    
    name: str = "snli_hypothesis"
    edit_target: str = "hypothesis"
    
    def format_for_prompt(self, entry: dict) -> dict:
        """
        Format entry for hypothesis editing prompt.
        
        Returns:
            Dictionary with:
                - premise: The fixed premise
                - hypothesis: The text to edit
                - label: Current label
                - text_to_edit: Same as hypothesis
                - fixed_text: Same as premise
        """
        return {
            "premise": entry["premise"],
            "hypothesis": entry["hypothesis"],
            "label": entry["label"],
            "text_to_edit": entry["hypothesis"],
            "fixed_text": entry["premise"],
        }
    
    def get_original_text(self, entry: dict) -> str:
        """Get the hypothesis (text to be edited)."""
        return entry["hypothesis"]
    
    def get_verification_inputs(self, entry: dict, edited_text: str) -> dict:
        """Get inputs for NLI verification with edited hypothesis."""
        return {
            "premise": entry["premise"],
            "hypothesis": edited_text,
        }


# =============================================================================
# BoolQ Dataset
# =============================================================================

@register_dataset
class BoolQDataset(BaseDataset):
    """BoolQ yes/no question answering dataset."""
    
    name: str = "boolq"
    labels: list[str] = ["true", "false"]
    edit_target: str = "passage"
    _hf_dataset_name: str = "google/boolq"
    
    def _get_cache_path(self, data_dir: str, split: str) -> Path:
        """Get the path to the cached JSONL file."""
        return Path(data_dir) / "boolq" / f"{split}.jsonl"
    
    def _download_and_cache(self, data_dir: str, split: str) -> list[dict]:
        """Download BoolQ from HuggingFace and cache locally."""
        cache_path = self._get_cache_path(data_dir, split)
        
        print(f"Downloading BoolQ {split} split from HuggingFace...")
        dataset = load_dataset(self._hf_dataset_name, split=split)
        
        entries = []
        for idx, item in enumerate(dataset):
            entries.append({
                "idx": idx,
                "passage": item["passage"],
                "question": item["question"],
                "label": "true" if item["answer"] else "false",
            })
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(entries, str(cache_path))
        print(f"Cached {len(entries)} entries to {cache_path}")
        
        return entries
    
    def load(self, data_dir: str, split: str = "train") -> list[dict]:
        """
        Load BoolQ dataset, downloading and caching if necessary.
        
        Args:
            data_dir: Directory to cache downloaded data
            split: Dataset split to load
            
        Returns:
            List of dataset entries
        """
        cache_path = self._get_cache_path(data_dir, split)
        
        if cache_path.exists():
            print(f"Loading BoolQ {split} from cache: {cache_path}")
            entries = load_jsonl(str(cache_path))
        else:
            entries = self._download_and_cache(data_dir, split)
        
        # Add dataset-specific metadata
        for entry in entries:
            entry["dataset_name"] = self.name
            entry["edit_target"] = self.edit_target
        
        return entries
    
    def format_for_prompt(self, entry: dict) -> dict:
        """
        Format entry for passage editing prompt.
        
        Returns:
            Dictionary with:
                - passage: The text to edit
                - question: The yes/no question
                - label: Current answer (true/false)
        """
        return {
            "passage": entry["passage"],
            "question": entry["question"],
            "label": entry["label"],
        }
    
    def get_original_text(self, entry: dict) -> str:
        """Get the passage (text to be edited)."""
        return entry["passage"]
    
    def get_verification_inputs(self, entry: dict, edited_text: str) -> dict:
        """Get inputs for yes/no verification with edited passage."""
        return {
            "passage": edited_text,
            "question": entry["question"],
        }
    
    def parse_label_from_response(self, response: str) -> Optional[str]:
        """Parse yes/no label from verification response."""
        response_lower = response.lower().strip()
        
        # Check for explicit true/false
        if "true" in response_lower or "yes" in response_lower:
            return "true"
        if "false" in response_lower or "no" in response_lower:
            return "false"
        
        return None
    
    def build_result_entry(self, entry: dict, counterfactuals: list[dict]) -> dict:
        """Build result entry for BoolQ dataset."""
        return {
            "idx": entry["idx"],
            "passage": entry["passage"],
            "question": entry["question"],
            "original_label": entry["label"],
            "edit_target": entry["edit_target"],
            "dataset_name": self.name,
            "counterfactuals": counterfactuals,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def load_all_datasets(
    data_dir: str,
    dataset_names: list[str],
    split: str = "train",
    max_entries: int = None,
) -> dict[str, list[dict]]:
    """
    Load multiple datasets.
    
    Args:
        data_dir: Directory to cache downloaded data
        dataset_names: List of dataset names to load
        split: Dataset split to load
        max_entries: Maximum entries per dataset (None for all)
        
    Returns:
        Dictionary mapping dataset names to their entries
    """
    all_data = {}
    
    for name in dataset_names:
        dataset = get_dataset(name)
        entries = dataset.load(data_dir, split)
        
        if max_entries is not None:
            entries = entries[:max_entries]
        
        all_data[name] = entries
        print(f"Loaded {len(entries)} entries from {name}")
    
    return all_data

