### Summer Internship - Earth Embeddings
### Model - Supervised Learning 
### By Edgar Daniel

"""Shared paths and config loading for the supervised workflows.

Single source for the repo-root anchors: YAMLs live in ``<repo>/config``
and results land under ``<repo>/outputs``.
"""
### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "models"

### -------------------------------------------------------------------------------
### Functions and Classes ---------------------------------------------------------


def load_yaml(name: str) -> dict:
    """Load one YAML file from the repo-level ``config/`` folder."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path} (expected under {CONFIG_DIR})."
        )
    with open(path) as f:
        return yaml.safe_load(f)
