"""
Central configuration for the Counterfactual DPO Training Pipeline.

All adjustable parameters are defined here for easy experimentation.
"""

import os
from pathlib import Path


class Config:
    """Configuration class with all pipeline settings."""
    
    # ==========================================================================
    # Model Settings
    # ==========================================================================
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_CACHE_DIR: str = "./model_cache"
    TORCH_DTYPE: str = "bfloat16"  # Full precision for HPC
    
    # HuggingFace token (optional, set via environment variable)
    HF_TOKEN: str = os.environ.get("HF_TOKEN", None)
    
    # ==========================================================================
    # Dataset Settings
    # ==========================================================================
    # Available: "snli_premise", "snli_hypothesis", "boolq"
    # Add more dataset names here as they are implemented
    ACTIVE_DATASETS: list = ["snli_premise", "snli_hypothesis"]
    
    # Local cache directory for downloaded datasets
    DATA_DIR: str = "./data"
    
    # Dataset split to use for counterfactual generation
    DATASET_SPLIT: str = "train"
    
    # Maximum number of entries to process (None for all)
    MAX_ENTRIES: int = None
    
    # ==========================================================================
    # Generation Parameters
    # ==========================================================================
    # High temperature for diverse sampling (variety comes from sampling, not param variation)
    TEMPERATURE: float = 1.2
    
    # Top-p (nucleus sampling) - high value for diversity
    TOP_P: float = 0.95
    
    # Top-k - high value for diversity
    TOP_K: int = 100
    
    # Target number of counterfactuals per entry
    COUNTERFACTUALS_PER_ENTRY: int = 40
    
    # Maximum new tokens for generation
    MAX_NEW_TOKENS: int = 2048
    
    # ==========================================================================
    # Evaluation Settings
    # ==========================================================================
    # Sentence transformer model for semantic similarity
    SEMANTIC_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ==========================================================================
    # DPO Pair Selection
    # ==========================================================================
    # Number of top-scoring correct CFs to use as "chosen"
    CHOSEN_COUNT: int = 2
    
    # Number of top-scoring incorrect CFs to use as "rejected"
    REJECTED_COUNT: int = 2
    
    # ==========================================================================
    # Output Paths
    # ==========================================================================
    OUTPUT_DIR: str = "./results"
    COUNTERFACTUALS_DIR: str = "./results/counterfactuals"
    DPO_DATASET_DIR: str = "./results/dpo_pairs"
    
    # ==========================================================================
    # Processing Settings
    # ==========================================================================
    # Save progress every N entries (for resume capability)
    SAVE_EVERY: int = 10
    
    # Batch size for semantic similarity computation
    SIMILARITY_BATCH_SIZE: int = 32
    
    @classmethod
    def get_torch_dtype(cls):
        """Get the torch dtype object from string."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(cls.TORCH_DTYPE, torch.bfloat16)
    
    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        Path(cls.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.COUNTERFACTUALS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.DPO_DATASET_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

