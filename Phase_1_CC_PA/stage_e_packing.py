"""
Stage E: Position-Aware Packing
--------------------------------
Arranges compressed chunks to maximize LLM attention based on findings
from "Lost in the Middle: How Language Models Use Long Contexts"
(Liu et al., 2023, arXiv:2307.03172).

Key insight: LLMs attend most to content at the BEGINNING and END of the
context window, with significantly degraded attention to content in the
MIDDLE. This stage re-orders chunks so that the most relevant information
occupies high-attention positions.

Strategies:
    - edges_first: Place top-ranked chunks at start and end, lower-ranked in middle
    - decreasing: Simple descending relevance order (baseline)
    - round_robin: Alternate placing chunks at start and end
"""

from typing import List

import tiktoken

from .stage_a_chunking import Chunk


class PositionAwarePacker:
    """Packs compressed chunks into a context window with position awareness."""

    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "edges_first")
        self.max_context_tokens = config.get("max_context_tokens", 4096)
        self.start_ratio = config.get("prioritize_start_ratio", 0.4)
        self.end_ratio = config.get("prioritize_end_ratio", 0.3)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def pack(self, chunks: List[Chunk]) -> str:
        """Pack chunks into a position-aware context string.

        Args:
            chunks: Ordered list of compressed chunks (by relevance, descending).

        Returns:
            Final packed context string ready for LLM input.
        """
        if not chunks:
            return ""

        if self.strategy == "edges_first":
            ordered = self._edges_first(chunks)
        elif self.strategy == "decreasing":
            ordered = self._decreasing(chunks)
        elif self.strategy == "round_robin":
            ordered = self._round_robin(chunks)
        else:
            raise ValueError(f"Unknown packing strategy: {self.strategy}")

        # Concatenate and enforce token budget
        return self._concat_with_budget(ordered)

    def _edges_first(self, chunks: List[Chunk]) -> List[Chunk]:
        """Place most relevant chunks at beginning and end of context.

        Distribution:
            - Top start_ratio of chunks -> beginning
            - Next end_ratio of chunks -> end
            - Remaining -> middle (lowest attention zone)
        """
        n = len(chunks)
        n_start = max(1, int(n * self.start_ratio))
        n_end = max(1, int(n * self.end_ratio))
        n_middle = n - n_start - n_end

        if n_middle < 0:
            # Not enough chunks to split three ways
            n_start = n // 2
            n_end = n - n_start
            n_middle = 0

        start_chunks = chunks[:n_start]
        end_chunks = chunks[n_start:n_start + n_end]
        middle_chunks = chunks[n_start + n_end:]

        # Final order: start (high attention) -> middle (low attention) -> end (high attention)
        return start_chunks + middle_chunks + end_chunks

    def _decreasing(self, chunks: List[Chunk]) -> List[Chunk]:
        """Simple baseline: keep descending relevance order."""
        return chunks

    def _round_robin(self, chunks: List[Chunk]) -> List[Chunk]:
        """Alternate placing chunks at edges.

        Chunk 1 -> start, Chunk 2 -> end, Chunk 3 -> start+1, Chunk 4 -> end-1, ...
        """
        n = len(chunks)
        result = [None] * n
        left = 0
        right = n - 1

        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                result[left] = chunk
                left += 1
            else:
                result[right] = chunk
                right -= 1

        return [c for c in result if c is not None]

    def _concat_with_budget(self, chunks: List[Chunk]) -> str:
        """Concatenate chunks respecting the token budget."""
        parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(self._tokenizer.encode(chunk.text))
            if total_tokens + chunk_tokens > self.max_context_tokens:
                # Truncate final chunk to fit
                remaining = self.max_context_tokens - total_tokens
                if remaining > 0:
                    tokens = self._tokenizer.encode(chunk.text)[:remaining]
                    parts.append(self._tokenizer.decode(tokens))
                break
            parts.append(chunk.text)
            total_tokens += chunk_tokens

        return "\n\n".join(parts)

    def get_position_map(self, chunks: List[Chunk]) -> dict:
        """Return a mapping showing where each chunk was placed.

        Useful for analysis and debugging the packing strategy.
        """
        n = len(chunks)
        positions = {}
        for i, chunk in enumerate(chunks):
            if i < n * self.start_ratio:
                zone = "start (high attention)"
            elif i >= n * (1 - self.end_ratio):
                zone = "end (high attention)"
            else:
                zone = "middle (low attention)"
            positions[chunk.index] = {
                "packed_position": i,
                "attention_zone": zone,
                "relevance_score": chunk.metadata.get("relevance_score", None),
            }
        return positions
