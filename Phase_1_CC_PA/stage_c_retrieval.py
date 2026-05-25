"""
Stage C: Retrieval (Embedding + BM25)
--------------------------------------
Ranks and selects top-N chunks relevant to a query using:
- Dense embeddings (sentence-transformers)
- Sparse BM25
- Hybrid fusion (weighted combination)
"""

from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .stage_a_chunking import Chunk


class Retriever:
    """Retrieves top-N relevant chunks using embedding, BM25, or hybrid methods."""

    def __init__(self, config: dict):
        self.method = config.get("method", "hybrid")
        self.top_n = config.get("top_n", 10)
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.bm25_k1 = config.get("bm25_k1", 1.5)
        self.bm25_b = config.get("bm25_b", 0.75)
        self.hybrid_alpha = config.get("hybrid_alpha", 0.5)
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load sentence transformer model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def retrieve(self, chunks: List[Chunk], query: str) -> List[Tuple[Chunk, float]]:
        """Retrieve top-N chunks most relevant to the query.

        Args:
            chunks: List of text chunks to search over.
            query: The query/question to match against.

        Returns:
            List of (chunk, score) tuples, sorted by relevance descending.
        """
        if not chunks:
            return []

        if self.method == "embedding":
            return self._retrieve_by_embedding(chunks, query)
        elif self.method == "bm25":
            return self._retrieve_by_bm25(chunks, query)
        elif self.method == "hybrid":
            return self._retrieve_hybrid(chunks, query)
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

    def _retrieve_by_embedding(
        self, chunks: List[Chunk], query: str
    ) -> List[Tuple[Chunk, float]]:
        """Dense retrieval using sentence embeddings + cosine similarity."""
        model = self._get_embedding_model()
        texts = [c.text for c in chunks]

        # Encode query and documents
        query_emb = model.encode([query], normalize_embeddings=True)
        doc_embs = model.encode(texts, normalize_embeddings=True)

        # Cosine similarity (already normalized)
        scores = np.dot(doc_embs, query_emb.T).flatten()

        return self._select_top_n(chunks, scores)

    def _retrieve_by_bm25(
        self, chunks: List[Chunk], query: str
    ) -> List[Tuple[Chunk, float]]:
        """Sparse retrieval using BM25."""
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        return self._select_top_n(chunks, scores)

    def _retrieve_hybrid(
        self, chunks: List[Chunk], query: str
    ) -> List[Tuple[Chunk, float]]:
        """Hybrid retrieval: weighted fusion of embedding and BM25 scores."""
        # Get embedding scores
        model = self._get_embedding_model()
        texts = [c.text for c in chunks]
        query_emb = model.encode([query], normalize_embeddings=True)
        doc_embs = model.encode(texts, normalize_embeddings=True)
        emb_scores = np.dot(doc_embs, query_emb.T).flatten()

        # Get BM25 scores
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        emb_scores_norm = self._min_max_normalize(emb_scores)
        bm25_scores_norm = self._min_max_normalize(bm25_scores)

        # Weighted fusion
        alpha = self.hybrid_alpha
        combined_scores = alpha * emb_scores_norm + (1 - alpha) * bm25_scores_norm

        return self._select_top_n(chunks, combined_scores)

    def _select_top_n(
        self, chunks: List[Chunk], scores: np.ndarray
    ) -> List[Tuple[Chunk, float]]:
        """Select top-N chunks by score."""
        n = min(self.top_n, len(chunks))
        top_indices = np.argsort(scores)[::-1][:n]
        return [(chunks[i], float(scores[i])) for i in top_indices]

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
