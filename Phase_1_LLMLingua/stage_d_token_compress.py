"""
Stage D: Iterative Token-Level Compression
--------------------------------------------
The core fine-grained compression stage of LLMLingua.

Processes the prompt in fixed-size windows (iterative_size), computing
token-level perplexity using the small LM. Tokens with perplexity below
a dynamically estimated threshold are removed.

Key mechanism:
    1. Segment prompt into windows of `iterative_size` tokens
    2. For each window, compute per-token loss (negative log-likelihood)
    3. Estimate threshold based on target compression ratio using
       the distribution of losses
    4. Remove tokens whose loss < threshold (predictable → redundant)
    5. Optionally preserve split tokens (e.g., newlines) and forced tokens

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import tiktoken


@dataclass
class TokenCompressionResult:
    """Result of token-level compression."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    tokens_removed: int
    per_window_stats: List[dict] = field(default_factory=list)


class TokenCompressor:
    """Iterative token-level compression using LM perplexity."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.iterative_size = config.get("iterative_size", 200)
        self.target_ratio = config.get("rate", 0.5)
        self.keep_split = config.get("keep_split", False)
        self.split_token = config.get("split_token", "\n")
        self.force_tokens = config.get("force_tokens", [])
        self.force_reserve_digit = config.get("force_reserve_digit", False)
        self.condition_compare = config.get("condition_compare", False)
        self._model_name = config.get("model_name", "gpt2")
        self._device = config.get("device", "cpu")
        self._lm = None
        self._lm_tokenizer = None
        self._max_positions = 1024
        self._tiktoken = tiktoken.get_encoding("cl100k_base")

    def _load_model(self):
        """Lazy-load the small language model."""
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            from ._compat import dtype_kwarg as _dtype_kwarg

            self._lm_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if self._lm_tokenizer.pad_token is None:
                self._lm_tokenizer.pad_token = self._lm_tokenizer.eos_token
            self._lm = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **_dtype_kwarg(torch.float32),
            ).to(self._device)
            self._lm.eval()

            # Cap the per-forward-pass length to the model's positional limit.
            # GPT-2 only has 1024 positions; feeding more indexes past its
            # position-embedding table, which on macOS CPU manifests as a
            # native "bus error" (SIGBUS) rather than a clean exception.
            max_pos = getattr(self._lm.config, "n_positions", None) \
                or getattr(self._lm.config, "max_position_embeddings", None) \
                or 1024
            self._max_positions = int(max_pos)

    def compress(
        self,
        text: str,
        target_token: int = -1,
        dynamic_ratios: Optional[List[float]] = None,
    ) -> TokenCompressionResult:
        """
        Compress text at token level using iterative perplexity-based pruning.

        Args:
            text: The input text to compress.
            target_token: Target number of tokens after compression.
            dynamic_ratios: Per-segment dynamic compression adjustment.

        Returns:
            TokenCompressionResult with compressed text and statistics.
        """
        if not self.enabled or not text.strip():
            token_count = len(self._tiktoken.encode(text))
            return TokenCompressionResult(
                compressed_text=text,
                original_tokens=token_count,
                compressed_tokens=token_count,
                ratio=1.0,
                tokens_removed=0,
            )

        self._load_model()
        import torch

        # Tokenize with the LM tokenizer
        tokenized = self._lm_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"].to(self._device)
        n_tokens = input_ids.shape[1]

        if n_tokens == 0:
            return TokenCompressionResult(
                compressed_text=text,
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
                tokens_removed=0,
            )

        # Compute effective rate
        if target_token > 0:
            effective_rate = min(target_token / n_tokens, 1.0)
        else:
            effective_rate = self.target_ratio

        # Compute per-token losses
        token_losses = self._compute_token_losses(input_ids)

        # Compute iterative compression ratios per window
        iterative_ratios = self._get_iterative_ratios(
            n_tokens, effective_rate, dynamic_ratios
        )

        # Build keep mask using iterative windowed thresholding
        keep_mask = self._iterative_compress(
            input_ids, token_losses, iterative_ratios
        )

        # Apply force token preservation
        keep_mask = self._apply_force_tokens(input_ids, keep_mask)

        # Reconstruct compressed text
        kept_ids = input_ids[0][keep_mask].unsqueeze(0)
        compressed_text = self._lm_tokenizer.decode(
            kept_ids[0], skip_special_tokens=True
        )

        original_tiktoken = len(self._tiktoken.encode(text))
        compressed_tiktoken = len(self._tiktoken.encode(compressed_text))
        ratio = original_tiktoken / max(compressed_tiktoken, 1)

        return TokenCompressionResult(
            compressed_text=compressed_text,
            original_tokens=original_tiktoken,
            compressed_tokens=compressed_tiktoken,
            ratio=ratio,
            tokens_removed=original_tiktoken - compressed_tiktoken,
        )

    def _compute_token_losses(self, input_ids) -> "torch.Tensor":
        """Compute per-token negative log-likelihood losses.

        The forward pass is run in non-overlapping blocks of at most
        ``self._max_positions`` tokens (1024 for GPT-2). Feeding the whole
        sequence at once would (a) exceed the model's positional-embedding
        table for long contexts -- a native "bus error"/SIGBUS on macOS CPU --
        and (b) allocate a single huge ``(1, seq_len, vocab)`` logits tensor.
        Blocking keeps both bounded. The first token of every block has no
        in-block predecessor, so it is assigned an infinite loss (always kept);
        this matches the windowed spirit of LLMLingua and only preserves a
        handful of extra boundary tokens.
        """
        import torch

        n_tokens = input_ids.shape[1]
        block = max(2, int(self._max_positions))
        losses = torch.empty(n_tokens, device=input_ids.device, dtype=torch.float32)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for start in range(0, n_tokens, block):
                end = min(start + block, n_tokens)
                block_ids = input_ids[:, start:end]
                if block_ids.shape[1] < 2:
                    # Single trailing token: nothing to predict, always keep.
                    losses[start:end] = float("inf")
                    continue
                outputs = self._lm(input_ids=block_ids)
                logits = outputs.logits  # (1, block_len, vocab_size)
                shift_logits = logits[:, :-1, :]
                shift_labels = block_ids[:, 1:]
                block_losses = loss_fn(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                )
                # First token of the block is always kept (no predecessor here).
                losses[start] = float("inf")
                losses[start + 1:end] = block_losses

        # The very first token of the whole sequence is always kept too.
        if n_tokens > 0:
            losses[0] = float("inf")
        return losses

    def _get_iterative_ratios(
        self,
        n_tokens: int,
        effective_rate: float,
        dynamic_ratios: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute per-window compression ratios."""
        n_windows = max(1, (n_tokens + self.iterative_size - 1) // self.iterative_size)

        if dynamic_ratios and len(dynamic_ratios) == n_windows:
            # Adjust effective_rate per window using dynamic ratios
            return [max(0.0, min(1.0, effective_rate + dr)) for dr in dynamic_ratios]

        return [effective_rate] * n_windows

    def _iterative_compress(
        self,
        input_ids,
        token_losses,
        iterative_ratios: List[float],
    ) -> "torch.Tensor":
        """
        Iterative compression: process in windows, threshold by percentile.

        For each window, compute a threshold from the distribution of losses
        such that the target compression rate is achieved.
        """
        import torch

        n_tokens = input_ids.shape[1]
        keep_mask = torch.ones(n_tokens, dtype=torch.bool, device=input_ids.device)

        for window_idx, ratio in enumerate(iterative_ratios):
            start = window_idx * self.iterative_size
            end = min(start + self.iterative_size, n_tokens)

            if start >= n_tokens:
                break

            window_losses = token_losses[start:end]

            if ratio >= 1.0:
                # Keep all tokens in this window
                continue

            # Estimate threshold: percentile-based
            threshold = self._estimate_threshold(window_losses, ratio)

            # Mark tokens below threshold for removal
            for i in range(start, end):
                if token_losses[i] < threshold:
                    keep_mask[i] = False

        # Always keep first token
        if n_tokens > 0:
            keep_mask[0] = True

        # Keep split tokens if configured
        if self.keep_split:
            keep_mask = self._preserve_split_tokens(input_ids, keep_mask)

        return keep_mask

    def _estimate_threshold(self, losses, ratio: float) -> float:
        """
        Estimate the perplexity threshold using the distribution of losses.

        Tokens with loss below this threshold are considered redundant.
        The threshold is set at the (1-ratio) percentile of the loss distribution.
        """
        import torch

        valid_losses = losses[losses != float("inf")]
        if len(valid_losses) == 0:
            return float("-inf")

        # Target: keep `ratio` fraction of tokens
        # So remove (1-ratio) fraction → threshold at (1-ratio) percentile
        percentile = int((1 - ratio) * 100)
        percentile = max(0, min(99, percentile))

        threshold = np.percentile(valid_losses.cpu().numpy(), percentile)
        return float(threshold)

    def _preserve_split_tokens(self, input_ids, keep_mask):
        """Preserve newline/split tokens in the compressed output."""
        import torch

        split_token_id = self._lm_tokenizer.encode(
            self.split_token, add_special_tokens=False
        )
        if not split_token_id:
            return keep_mask

        split_id = split_token_id[0]
        ids_flat = input_ids[0]

        for i in range(len(ids_flat)):
            if ids_flat[i].item() == split_id:
                # Keep consecutive split tokens (paragraph breaks)
                if i > 0 and ids_flat[i - 1].item() == split_id:
                    keep_mask[i] = True
                    keep_mask[i - 1] = True
                elif i < len(ids_flat) - 1 and ids_flat[i + 1].item() == split_id:
                    keep_mask[i] = True

        return keep_mask

    def _apply_force_tokens(self, input_ids, keep_mask):
        """Force-preserve specific tokens (digits, user-specified tokens)."""
        import torch

        if not self.force_tokens and not self.force_reserve_digit:
            return keep_mask

        ids_flat = input_ids[0]
        for i in range(len(ids_flat)):
            token_text = self._lm_tokenizer.decode([ids_flat[i].item()])

            # Force-reserve digits
            if self.force_reserve_digit and any(c.isdigit() for c in token_text):
                keep_mask[i] = True

            # Force-reserve specified tokens
            if self.force_tokens:
                for ft in self.force_tokens:
                    if ft in token_text:
                        keep_mask[i] = True
                        break

        return keep_mask
