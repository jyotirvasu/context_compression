"""
Stage E: Response Recovery & Post-Processing
----------------------------------------------
After compression, the LLM generates a response based on the compressed
prompt. This stage recovers the original text from the compressed output
by mapping response tokens back to the original prompt.

Also provides utility functions for:
    - Assembling the final compressed prompt
    - Computing compression statistics
    - Structured prompt compression with <llmlingua> tags

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken


@dataclass
class CompressionResult:
    """Final result of the LLMLingua compression pipeline."""
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    ratio: str
    rate: str
    saving: str
    instruction: str = ""
    question: str = ""
    metadata: dict = field(default_factory=dict)


class ResponseRecovery:
    """Assembles final output and recovers original text from compressed responses."""

    def __init__(self, config: dict):
        self.concate_question = config.get("concate_question", True)
        self.add_instruction = config.get("add_instruction", False)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def assemble(
        self,
        instruction: str,
        compressed_contexts: List[str],
        question: str,
        original_token_count: int,
    ) -> CompressionResult:
        """
        Assemble the final compressed prompt from components.

        Args:
            instruction: Original instruction text.
            compressed_contexts: Compressed context strings.
            question: Original question text.
            original_token_count: Token count of the original full prompt.

        Returns:
            CompressionResult with assembled compressed prompt and statistics.
        """
        parts = []

        if instruction and self.add_instruction:
            parts.append(instruction)

        # Join compressed contexts
        context_str = "\n\n".join(c for c in compressed_contexts if c.strip())
        if context_str:
            parts.append(context_str)

        if question and self.concate_question:
            parts.append(question)

        compressed_prompt = "\n\n".join(parts)
        compressed_tokens = self._count_tokens(compressed_prompt)

        ratio = original_token_count / max(compressed_tokens, 1)
        rate = compressed_tokens / max(original_token_count, 1)
        saving = (original_token_count - compressed_tokens) * 0.06 / 1000

        return CompressionResult(
            compressed_prompt=compressed_prompt,
            origin_tokens=original_token_count,
            compressed_tokens=compressed_tokens,
            ratio=f"{ratio:.1f}x",
            rate=f"{rate * 100:.1f}%",
            saving=f", Saving ${saving:.1f} in GPT-4.",
            instruction=instruction,
            question=question,
        )

    def recover(
        self,
        original_prompt: str,
        compressed_prompt: str,
        response: str,
    ) -> str:
        """
        Recover original text from a response generated from compressed prompt.

        Maps response words back to the original prompt by finding matching
        subsequences in the original token sequence.

        Args:
            original_prompt: The full original uncompressed prompt.
            compressed_prompt: The compressed prompt that was sent to the LLM.
            response: The LLM's response based on the compressed prompt.

        Returns:
            Recovered response with original text substituted where possible.
        """
        response_words = response.split(" ")
        recovered_words = []

        i = 0
        while i < len(response_words):
            word = response_words[i]

            # If this word doesn't appear in compressed prompt, keep as-is
            if word not in compressed_prompt:
                recovered_words.append(word)
                i += 1
                continue

            # Find longest match in compressed prompt
            j = i
            while (
                j + 1 < len(response_words)
                and " ".join(response_words[i : j + 2]) in compressed_prompt
            ):
                j += 1

            # Try to find the original version
            compressed_span = " ".join(response_words[i : j + 1])
            original_span = self._find_in_original(
                compressed_span, original_prompt
            )

            recovered_words.append(original_span)
            i = j + 1

        return " ".join(recovered_words)

    def _find_in_original(self, compressed_span: str, original_prompt: str) -> str:
        """
        Find the best matching span in the original prompt for a compressed span.

        Uses character-level fuzzy matching to locate the original text.
        """
        # Try exact match first
        if compressed_span in original_prompt:
            return compressed_span

        # Try finding a superset in the original by word overlap
        compressed_words = set(compressed_span.lower().split())
        original_sentences = re.split(r"[.!?\n]+", original_prompt)

        best_match = compressed_span
        best_overlap = 0

        for sentence in original_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(compressed_words & sentence_words)
            if overlap > best_overlap and overlap >= len(compressed_words) * 0.5:
                best_overlap = overlap
                best_match = sentence.strip()

        return best_match

    def parse_structured_prompt(
        self, structured_prompt: str, global_rate: float = 0.5
    ) -> Dict[str, List]:
        """
        Parse a structured prompt with <llmlingua> tags.

        Tags define per-segment compression rates:
            <llmlingua, rate=0.4>content</llmlingua>
            <llmlingua, compress=False>content</llmlingua>

        Returns:
            Dictionary with 'segments', 'rates', 'compress_flags'.
        """
        pattern = (
            r"<llmlingua\s*"
            r"(?:,\s*rate\s*=\s*([\d.]+))?\s*"
            r"(?:,\s*compress\s*=\s*(True|False))?\s*"
            r"(?:,\s*rate\s*=\s*([\d.]+))?\s*"
            r"(?:,\s*compress\s*=\s*(True|False))?\s*"
            r">([^<]+)</llmlingua>"
        )
        matches = re.findall(pattern, structured_prompt)

        segments = []
        rates = []
        compress_flags = []

        for match in matches:
            segments.append(match[4])

            # Extract rate (could be in position 0 or 2)
            rate = float(match[0]) if match[0] else (float(match[2]) if match[2] else None)

            # Extract compress flag (could be in position 1 or 3)
            compress = (
                match[1] == "True"
                if match[1]
                else (match[3] == "True" if match[3] else None)
            )

            if compress is None:
                compress = True

            if rate is None:
                rate = global_rate if compress else 1.0

            rates.append(rate)
            compress_flags.append(compress)

        return {
            "segments": segments,
            "rates": rates,
            "compress_flags": compress_flags,
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
