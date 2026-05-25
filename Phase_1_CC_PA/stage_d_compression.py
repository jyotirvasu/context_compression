"""
Stage D: Selective Context Compression (Plug-and-Play)
------------------------------------------------------
Integrates the Selective Context library (Li et al., 2023) for
information-theoretic compression using self-information.

Reference:
    - GitHub: https://github.com/liyucheng09/Selective_Context
    - Paper: "Compressing Context to Enhance Inference Efficiency of LLMs"
             (EMNLP 2023)

The method computes self-information for each lexical unit (sentence,
phrase, or token) using a base LM (e.g., GPT-2) and removes units with
low self-information (i.e., predictable/redundant content).
"""

from typing import List, Tuple, Optional

import tiktoken

from .stage_a_chunking import Chunk


class Compressor:
    """Applies Selective Context compression to retrieved chunks.

    This is a plug-and-play wrapper around the selective_context library.
    If the library is unavailable, falls back to a simple truncation strategy.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.model_type = config.get("model_type", "gpt2")
        self.lang = config.get("lang", "en")
        self.reduce_ratio = config.get("reduce_ratio", 0.5)
        self.target_token_budget = config.get("target_token_budget", 2048)
        self.granularity = config.get("granularity", "phrase")
        self._sc = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_selective_context(self):
        """Lazy-load Selective Context model."""
        if self._sc is None:
            from selective_context import SelectiveContext
            self._sc = SelectiveContext(
                model_type=self.model_type,
                lang=self.lang,
            )
        return self._sc

    def compress(
        self, chunks: List[Tuple[Chunk, float]], query: Optional[str] = None
    ) -> List[Chunk]:
        """Compress retrieved chunks using Selective Context.

        Args:
            chunks: List of (chunk, relevance_score) tuples from Stage C.
            query: Optional query for context-aware compression.

        Returns:
            List of compressed Chunk objects within token budget.
        """
        if not self.enabled:
            return [chunk for chunk, _ in chunks]

        compressed_chunks = []
        total_tokens = 0

        for chunk, score in chunks:
            if total_tokens >= self.target_token_budget:
                break

            compressed_text = self._compress_text(chunk.text)
            token_count = len(self._tokenizer.encode(compressed_text))

            # Respect token budget
            if total_tokens + token_count > self.target_token_budget:
                # Truncate last chunk to fit budget
                remaining = self.target_token_budget - total_tokens
                tokens = self._tokenizer.encode(compressed_text)[:remaining]
                compressed_text = self._tokenizer.decode(tokens)
                token_count = remaining

            compressed_chunk = Chunk(
                text=compressed_text,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=token_count,
                metadata={**chunk.metadata, "relevance_score": score, "compressed": True},
            )
            compressed_chunks.append(compressed_chunk)
            total_tokens += token_count

        return compressed_chunks

    def _compress_text(self, text: str) -> str:
        """Apply Selective Context compression to a single text.

        Falls back to simple truncation if the library is unavailable.
        """
        try:
            sc = self._get_selective_context()
            compressed, reduced = sc(text, reduce_ratio=self.reduce_ratio)
            return compressed
        except ImportError:
            # Fallback: simple ratio-based truncation
            return self._fallback_compress(text)
        except Exception as e:
            # If compression fails for any reason, return original
            print(f"[Stage D] Compression failed, using original: {e}")
            return text

    def _fallback_compress(self, text: str) -> str:
        """Fallback compression: keep first N tokens based on reduce_ratio."""
        tokens = self._tokenizer.encode(text)
        keep_count = int(len(tokens) * (1 - self.reduce_ratio))
        return self._tokenizer.decode(tokens[:keep_count])
