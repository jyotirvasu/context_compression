"""
Stage A: Chunking
-----------------
Splits input documents into manageable chunks using various strategies:
- Sentence-level (spacy/nltk)
- Paragraph-level (double newline split)
- Fixed token windows
- Sliding window with overlap
"""

from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken


@dataclass
class Chunk:
    """Represents a single chunk of text with metadata."""
    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class Chunker:
    """Document chunker supporting multiple strategies."""

    def __init__(self, config: dict):
        self.method = config.get("method", "sentence")
        self.max_chunk_tokens = config.get("max_chunk_tokens", 256)
        self.overlap_tokens = config.get("overlap_tokens", 32)
        self.sentence_splitter = config.get("sentence_splitter", "spacy")
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spacy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks based on configured method."""
        if self.method == "sentence":
            return self._chunk_by_sentence(text)
        elif self.method == "paragraph":
            return self._chunk_by_paragraph(text)
        elif self.method == "fixed_token":
            return self._chunk_by_fixed_tokens(text)
        elif self.method == "sliding_window":
            return self._chunk_by_sliding_window(text)
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")

    def _chunk_by_sentence(self, text: str) -> List[Chunk]:
        """Split into sentence-level chunks, merging short sentences."""
        if self.sentence_splitter == "spacy":
            sentences = self._split_sentences_spacy(text)
        else:
            sentences = self._split_sentences_nltk(text)

        chunks = []
        current_text = ""
        current_start = 0

        for sent in sentences:
            candidate = (current_text + " " + sent).strip() if current_text else sent
            token_count = len(self._tokenizer.encode(candidate))

            if token_count > self.max_chunk_tokens and current_text:
                # Flush current buffer
                chunks.append(self._make_chunk(current_text, len(chunks), current_start))
                current_text = sent
                current_start = text.find(sent, current_start)
            else:
                current_text = candidate

        if current_text.strip():
            chunks.append(self._make_chunk(current_text, len(chunks), current_start))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[Chunk]:
        """Split on double newlines (paragraphs)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        offset = 0

        for para in paragraphs:
            start = text.find(para, offset)
            chunks.append(self._make_chunk(para, len(chunks), start))
            offset = start + len(para)

        return chunks

    def _chunk_by_fixed_tokens(self, text: str) -> List[Chunk]:
        """Split into fixed-size token windows."""
        tokens = self._tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.max_chunk_tokens):
            window = tokens[i:i + self.max_chunk_tokens]
            chunk_text = self._tokenizer.decode(window)
            chunks.append(self._make_chunk(chunk_text, len(chunks), i))

        return chunks

    def _chunk_by_sliding_window(self, text: str) -> List[Chunk]:
        """Split with sliding window and overlap."""
        tokens = self._tokenizer.encode(text)
        step = self.max_chunk_tokens - self.overlap_tokens
        chunks = []

        for i in range(0, len(tokens), step):
            window = tokens[i:i + self.max_chunk_tokens]
            chunk_text = self._tokenizer.decode(window)
            chunks.append(self._make_chunk(chunk_text, len(chunks), i))
            if i + self.max_chunk_tokens >= len(tokens):
                break

        return chunks

    def _split_sentences_spacy(self, text: str) -> List[str]:
        """Use spacy for sentence segmentation."""
        nlp = self._get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _split_sentences_nltk(self, text: str) -> List[str]:
        """Use nltk for sentence segmentation."""
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def _make_chunk(self, text: str, index: int, start_char: int) -> Chunk:
        """Create a Chunk object with token count."""
        token_count = len(self._tokenizer.encode(text))
        return Chunk(
            text=text,
            index=index,
            start_char=start_char,
            end_char=start_char + len(text),
            token_count=token_count,
        )
