"""
Utility helpers for the Context Compression pipeline.
"""

import yaml
import tiktoken


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def compute_compression_ratio(original: str, compressed: str) -> float:
    """Compute the compression ratio (tokens saved / original tokens)."""
    orig_tokens = count_tokens(original)
    comp_tokens = count_tokens(compressed)
    if orig_tokens == 0:
        return 0.0
    return 1.0 - (comp_tokens / orig_tokens)
