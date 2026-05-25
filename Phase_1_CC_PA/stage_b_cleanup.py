"""
Stage B: Cleanup Rules
----------------------
Applies text normalization and filtering to chunks:
- URL/email removal
- Whitespace normalization
- Unicode normalization
- Duplicate removal
- Length-based filtering
"""

import re
import unicodedata
from typing import List

from .stage_a_chunking import Chunk


class Cleaner:
    """Applies configurable cleanup rules to chunks."""

    def __init__(self, config: dict):
        self.remove_urls = config.get("remove_urls", True)
        self.remove_emails = config.get("remove_emails", True)
        self.remove_excessive_whitespace = config.get("remove_excessive_whitespace", True)
        self.remove_special_chars = config.get("remove_special_chars", False)
        self.normalize_unicode = config.get("normalize_unicode", True)
        self.min_chunk_length = config.get("min_chunk_length", 10)
        self.max_chunk_length = config.get("max_chunk_length", 2048)
        self.remove_duplicates = config.get("remove_duplicates", True)
        self.lowercase = config.get("lowercase", False)

        # Compiled patterns
        self._url_pattern = re.compile(
            r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        )
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        )
        self._whitespace_pattern = re.compile(r"\s+")
        self._special_char_pattern = re.compile(r"[^\w\s.,;:!?'\"\-\(\)\[\]{}]")

    def clean(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply all configured cleanup rules to a list of chunks."""
        cleaned = []
        seen_texts = set()

        for chunk in chunks:
            text = chunk.text

            # Apply text transformations
            text = self._apply_text_rules(text)

            # Skip empty or too-short chunks
            if len(text.strip()) < self.min_chunk_length:
                continue

            # Truncate overly long chunks
            if len(text) > self.max_chunk_length:
                text = text[: self.max_chunk_length]

            # Deduplicate
            if self.remove_duplicates:
                normalized_key = text.strip().lower()
                if normalized_key in seen_texts:
                    continue
                seen_texts.add(normalized_key)

            # Create cleaned chunk
            cleaned_chunk = Chunk(
                text=text,
                index=len(cleaned),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )
            cleaned.append(cleaned_chunk)

        return cleaned

    def _apply_text_rules(self, text: str) -> str:
        """Apply individual text transformation rules."""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.remove_urls:
            text = self._url_pattern.sub("", text)

        if self.remove_emails:
            text = self._email_pattern.sub("", text)

        if self.remove_special_chars:
            text = self._special_char_pattern.sub("", text)

        if self.remove_excessive_whitespace:
            text = self._whitespace_pattern.sub(" ", text).strip()

        if self.lowercase:
            text = text.lower()

        return text
