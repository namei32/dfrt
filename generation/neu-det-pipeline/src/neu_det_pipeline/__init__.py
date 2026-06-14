"""
NEU-DET DRFT-v2 Data Augmentation Pipeline

A retained DRFT-v2 pipeline for steel surface defect evidence generation
using defect residual fields and DRFT-LoRA inpainting.
"""

__version__ = "0.1.0"

from .config import (
    ConfigBundle,
    DatasetConfig,
    GenerationConfig,
    GuidanceConfig,
    LoRAConfig,
    load_config_bundle,
)

__all__ = [
    "__version__",
    "ConfigBundle",
    "DatasetConfig",
    "GenerationConfig",
    "GuidanceConfig",
    "LoRAConfig",
    "load_config_bundle",
]
