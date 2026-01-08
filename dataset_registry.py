"""
Dataset loading and registry for the Counterfactual DPO Training Pipeline.

Provides an extensible interface for loading different datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import load_dataset

from utils import save_jsonl, load_jsonl


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
    edit_target: str  # What to edit: "premise", "hypothesis", "text", etc.
    
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
    
    def get_alternative_labels(self, current_label: str) -> list[str]:
        """Get all labels except the current one."""
        return [l for l in self.labels if l != current_label]


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

