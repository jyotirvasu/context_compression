"""
Stage B: Context-Level Filter (Coarse-Grained Compression)
-----------------------------------------------------------
Ranks and selects the most informative contexts/demonstrations using
perplexity computed by a small language model.

The intuition: contexts with higher perplexity (less predictable by the
small LM) carry more information and are more likely to be useful for
the downstream task.

Supports multiple ranking methods:
    - llmlingua: perplexity-based ranking via small LM
    - longllmlingua: question-conditioned perplexity ranking
    - bm25: sparse retrieval ranking
    - embedding: dense retrieval ranking

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
    "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context
     Scenarios via Prompt Compression" (ACL 2024)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tiktoken


@dataclass
class RankedContext:
    """A context with its ranking score and metadata."""
    text: str
    original_index: int
    score: float
    token_count: int
    selected: bool = True


class ContextFilter:
    """Coarse-grained context-level filtering via perplexity ranking."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.rank_method = config.get("rank_method", "llmlingua")
        self.target_context = config.get("target_context", -1)
        self.context_level_rate = config.get("context_level_rate", 1.0)
        self.context_budget_expr = config.get("context_budget", "+100")
        self.force_context_ids = config.get("force_context_ids", None)
        self.force_context_number = config.get("force_context_number", None)
        self.reorder_context = config.get("reorder_context", "original")
        self.condition_in_question = config.get("condition_in_question", "none")
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._lm = None
        self._lm_tokenizer = None
        self._model_name = config.get("model_name", "gpt2")
        self._device = config.get("device", "cpu")

    def _load_model(self):
        """Lazy-load the small language model for perplexity computation."""
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            from ._compat import dtype_kwarg as _dtype_kwarg

            self._lm_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._lm = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **_dtype_kwarg(torch.float32),
            ).to(self._device)
            self._lm.eval()

    def filter(
        self,
        contexts: List[str],
        question: str = "",
        context_tokens: Optional[List[int]] = None,
        target_token: int = -1,
    ) -> Tuple[List[RankedContext], List[int]]:
        """
        Rank and filter contexts, returning selected contexts and their indices.

        Args:
            contexts: List of context strings.
            question: Question text for question-aware ranking.
            context_tokens: Pre-computed token counts per context.
            target_token: Token budget for context selection.

        Returns:
            Tuple of (ranked/selected contexts, selected indices).
        """
        if not self.enabled or len(contexts) <= 1:
            ranked = [
                RankedContext(
                    text=c,
                    original_index=i,
                    score=0.0,
                    token_count=context_tokens[i] if context_tokens else self._count_tokens(c),
                )
                for i, c in enumerate(contexts)
            ]
            return ranked, list(range(len(contexts)))

        if context_tokens is None:
            context_tokens = [self._count_tokens(c) for c in contexts]

        # Rank contexts
        scores = self._rank_contexts(contexts, question, context_tokens)

        # Select contexts within budget
        selected_indices = self._select_within_budget(
            scores, context_tokens, target_token
        )

        # Reorder selected contexts
        selected_indices = self._reorder(selected_indices, scores)

        # Build result
        ranked_contexts = []
        for i in selected_indices:
            ranked_contexts.append(
                RankedContext(
                    text=contexts[i],
                    original_index=i,
                    score=scores[i],
                    token_count=context_tokens[i],
                    selected=True,
                )
            )

        return ranked_contexts, selected_indices

    def _rank_contexts(
        self,
        contexts: List[str],
        question: str,
        context_tokens: List[int],
    ) -> List[float]:
        """Compute ranking scores for each context."""
        if self.rank_method == "llmlingua":
            return self._rank_by_perplexity(contexts)
        elif self.rank_method == "longllmlingua":
            return self._rank_by_conditioned_perplexity(contexts, question, context_tokens)
        elif self.rank_method == "bm25":
            return self._rank_by_bm25(contexts, question)
        else:
            return self._rank_by_perplexity(contexts)

    def _rank_by_perplexity(self, contexts: List[str]) -> List[float]:
        """Rank by perplexity from small LM (higher PPL = more informative)."""
        self._load_model()
        import torch

        scores = []
        with torch.no_grad():
            for ctx in contexts:
                ppl = self._compute_perplexity(ctx)
                scores.append(ppl)
        return scores

    def _rank_by_conditioned_perplexity(
        self,
        contexts: List[str],
        question: str,
        context_tokens: List[int],
    ) -> List[float]:
        """
        Question-conditioned perplexity ranking (LongLLMLingua).
        PPL(context | question) - lower means more relevant to question.
        """
        self._load_model()
        import torch

        condition_text = question + " We can get the answer to this question in the given documents."
        scores = []
        with torch.no_grad():
            for ctx, ct in zip(contexts, context_tokens):
                if self.condition_in_question == "after":
                    ppl = self._compute_perplexity(ctx + " " + condition_text)
                elif self.condition_in_question == "before":
                    ppl = self._compute_perplexity(condition_text + " " + ctx)
                else:
                    ppl = self._compute_perplexity(ctx)
                # Lower PPL relative to length → more relevant
                scores.append(-ppl + ct * 2 / 250)
        return scores

    def _rank_by_bm25(self, contexts: List[str], question: str) -> List[float]:
        """Rank using BM25 sparse retrieval."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [c.lower().split() for c in contexts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question.lower().split()
        scores = bm25.get_scores(tokenized_query)
        return scores.tolist()

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text using the small LM."""
        import torch

        inputs = self._lm_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)

        with torch.no_grad():
            outputs = self._lm(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return torch.exp(loss).item()

    def _select_within_budget(
        self,
        scores: List[float],
        context_tokens: List[int],
        target_token: int,
    ) -> List[int]:
        """Select contexts within token budget based on scores."""
        n = len(scores)

        # If force_context_number is set, take top N
        if self.force_context_number is not None:
            k = min(self.force_context_number, n)
        elif self.target_context > 0:
            k = min(self.target_context, n)
        elif self.context_level_rate < 1.0:
            k = max(1, int(n * self.context_level_rate))
        else:
            k = n

        # Sort by score descending (higher = more informative for llmlingua)
        if self.rank_method == "longllmlingua":
            # For longllmlingua, higher score = more relevant
            sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        else:
            # For llmlingua, higher PPL = more informative
            sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

        selected = []
        total_tokens = 0

        for idx in sorted_indices:
            # Force-include specific contexts
            if self.force_context_ids and idx in self.force_context_ids:
                selected.append(idx)
                total_tokens += context_tokens[idx]
                continue

            if len(selected) >= k:
                break

            if target_token > 0 and total_tokens + context_tokens[idx] > target_token:
                break

            selected.append(idx)
            total_tokens += context_tokens[idx]

        return selected

    def _reorder(self, indices: List[int], scores: List[float]) -> List[int]:
        """Reorder selected contexts based on strategy."""
        if self.reorder_context == "original":
            return sorted(indices)
        elif self.reorder_context == "sort":
            # Sort by score (most relevant first)
            return sorted(indices, key=lambda i: scores[i], reverse=True)
        elif self.reorder_context == "two_stage":
            # Interleave: even-indexed at front, odd-indexed reversed at back
            even = [indices[i] for i in range(0, len(indices), 2)]
            odd = [indices[i] for i in range(1, len(indices), 2)]
            return even + odd[::-1]
        return indices

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
