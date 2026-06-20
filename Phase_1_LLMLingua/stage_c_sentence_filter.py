"""
Stage C: Sentence-Level Filter
-------------------------------
Performs mid-granularity compression by ranking and filtering sentences
within each selected context.

Sentences are scored by perplexity from the small LM. High-perplexity
sentences (less predictable → more informative) are retained.

Supports:
    - keep_first_sentence / keep_last_sentence (force-retain boundaries)
    - high_priority_bonus for forced sentences
    - token_budget_ratio for flexible budget allocation

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tiktoken


@dataclass
class ScoredSentence:
    """A sentence with its perplexity score and retention status."""
    text: str
    index: int
    context_index: int
    score: float
    token_count: int
    retained: bool = True


class SentenceFilter:
    """Sentence-level filtering within contexts using perplexity ranking."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.keep_first_sentence = config.get("keep_first_sentence", 0)
        self.keep_last_sentence = config.get("keep_last_sentence", 0)
        self.keep_sentence_number = config.get("keep_sentence_number", 0)
        self.high_priority_bonus = config.get("high_priority_bonus", 100)
        self.token_budget_ratio = config.get("token_budget_ratio", 1.4)
        self.rank_method = config.get("rank_method", "llmlingua")
        self.condition_in_question = config.get("condition_in_question", "none")
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._lm = None
        self._lm_tokenizer = None
        self._model_name = config.get("model_name", "gpt2")
        self._device = config.get("device", "cpu")

    def _load_model(self):
        """Lazy-load the small language model."""
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
        target_token: int,
        question: str = "",
    ) -> Tuple[List[str], List[List[ScoredSentence]]]:
        """
        Filter sentences within each context based on perplexity scores.

        Args:
            contexts: List of context strings (already filtered at context level).
            target_token: Token budget for sentence-level output.
            question: Optional question for question-aware ranking.

        Returns:
            Tuple of (filtered context strings, sentence scoring info).
        """
        if not self.enabled:
            return contexts, []

        # Split contexts into sentences
        all_sentences = []
        context_sentence_map = []
        for ctx_idx, ctx in enumerate(contexts):
            sentences = self._split_sentences(ctx)
            context_sentence_map.append(sentences)
            for sent_idx, sent in enumerate(sentences):
                all_sentences.append((ctx_idx, sent_idx, sent))

        if not all_sentences:
            return contexts, []

        # Score all sentences
        scored = self._score_sentences(all_sentences, question)

        # Apply priority bonuses for first/last sentences
        self._apply_priority_bonus(scored, context_sentence_map)

        # Select sentences within budget
        retained_sentences = self._select_within_budget(scored, target_token)

        # Reconstruct contexts from retained sentences
        filtered_contexts = self._reconstruct_contexts(
            contexts, context_sentence_map, retained_sentences
        )

        return filtered_contexts, retained_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
        # Sync sentences with original text to preserve formatting
        return self._sync_sentences(sentences, text)

    def _sync_sentences(self, sentences: List[str], text: str) -> List[str]:
        """Ensure sentence boundaries align with original text."""
        if not sentences:
            return [text] if text.strip() else []

        synced = []
        seen = 0
        for i, sent in enumerate(sentences):
            start = text.find(sent, seen)
            if start == -1:
                start = seen
            if i == len(sentences) - 1:
                synced.append(text[start:])
            else:
                next_start = text.find(sentences[i + 1][:5], start + len(sent))
                if next_start == -1:
                    next_start = start + len(sent)
                synced.append(text[start:next_start])
                seen = next_start

        return synced

    def _score_sentences(
        self,
        sentences: List[Tuple[int, int, str]],
        question: str,
    ) -> List[ScoredSentence]:
        """Score sentences by perplexity."""
        self._load_model()
        import torch

        scored = []
        with torch.no_grad():
            for ctx_idx, sent_idx, sent_text in sentences:
                token_count = self._count_tokens(sent_text)

                if self.condition_in_question != "none" and question:
                    ppl = self._compute_sentence_ppl(sent_text, question)
                else:
                    ppl = self._compute_sentence_ppl(sent_text)

                scored.append(
                    ScoredSentence(
                        text=sent_text,
                        index=sent_idx,
                        context_index=ctx_idx,
                        score=ppl,
                        token_count=token_count,
                    )
                )

        return scored

    def _compute_sentence_ppl(self, text: str, condition: str = "") -> float:
        """Compute perplexity for a sentence, optionally conditioned on question."""
        import torch

        if condition and self.condition_in_question == "after":
            full_text = text + " " + condition
        elif condition and self.condition_in_question == "before":
            full_text = condition + " " + text
        else:
            full_text = text

        inputs = self._lm_tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        if inputs["input_ids"].shape[1] == 0:
            return 0.0

        with torch.no_grad():
            outputs = self._lm(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return loss.item()  # Return log-perplexity (NLL)

    def _apply_priority_bonus(
        self,
        scored: List[ScoredSentence],
        context_sentence_map: List[List[str]],
    ):
        """Apply high-priority bonus to first/last sentences."""
        for sentence in scored:
            ctx_idx = sentence.context_index
            n_sents = len(context_sentence_map[ctx_idx])
            sent_idx = sentence.index

            # Bonus for first sentences
            if self.keep_first_sentence > 0 and sent_idx < self.keep_first_sentence:
                sentence.score += self.high_priority_bonus

            # Bonus for last sentences
            if self.keep_last_sentence > 0 and sent_idx >= n_sents - self.keep_last_sentence:
                sentence.score += self.high_priority_bonus

    def _select_within_budget(
        self,
        scored: List[ScoredSentence],
        target_token: int,
    ) -> List[List[ScoredSentence]]:
        """Select sentences within token budget."""
        adjusted_budget = int(target_token * self.token_budget_ratio)

        # Sort by score descending (higher PPL/NLL = more informative for llmlingua)
        sorted_sentences = sorted(scored, key=lambda s: s.score, reverse=True)

        total_tokens = 0
        retained_indices = set()

        for sent in sorted_sentences:
            if total_tokens + sent.token_count > adjusted_budget:
                continue
            retained_indices.add((sent.context_index, sent.index))
            total_tokens += sent.token_count

        # Force-retain if keep_sentence_number is set
        if self.keep_sentence_number > 0:
            ctx_groups = {}
            for sent in scored:
                ctx_groups.setdefault(sent.context_index, []).append(sent)
            for ctx_idx, sents in ctx_groups.items():
                top_k = sorted(sents, key=lambda s: s.score, reverse=True)[
                    : self.keep_sentence_number
                ]
                for s in top_k:
                    retained_indices.add((s.context_index, s.index))

        # Mark retention
        for sent in scored:
            sent.retained = (sent.context_index, sent.index) in retained_indices

        # Group by context
        result = []
        ctx_groups = {}
        for sent in scored:
            ctx_groups.setdefault(sent.context_index, []).append(sent)
        for ctx_idx in sorted(ctx_groups.keys()):
            result.append(sorted(ctx_groups[ctx_idx], key=lambda s: s.index))

        return result

    def _reconstruct_contexts(
        self,
        original_contexts: List[str],
        context_sentence_map: List[List[str]],
        retained_sentences: List[List[ScoredSentence]],
    ) -> List[str]:
        """Reconstruct context strings from retained sentences."""
        filtered = []
        for ctx_idx, sentences_info in enumerate(retained_sentences):
            kept = [s.text for s in sentences_info if s.retained]
            filtered.append("".join(kept))
        return filtered

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
