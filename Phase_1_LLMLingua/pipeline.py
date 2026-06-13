"""
LLMLingua Pipeline Orchestrator
---------------------------------
End-to-end pipeline that chains all five stages of the LLMLingua
compression algorithm:

    Stage A: Budget Controller → allocate token budgets
    Stage B: Context-Level Filter → coarse-grained context selection
    Stage C: Sentence-Level Filter → mid-grained sentence selection
    Stage D: Token-Level Compression → fine-grained iterative pruning
    Stage E: Response Recovery → assemble output & recover responses

Usage:
    from Phase_1_LLMLingua.pipeline import LLMLinguaPipeline

    pipeline = LLMLinguaPipeline(config)
    result = pipeline.compress(
        instruction="Summarize the following:",
        contexts=["Document 1...", "Document 2..."],
        question="What is the main finding?",
    )
    print(result.compressed_prompt)
    print(result.ratio)

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
"""

from typing import Dict, List, Optional, Union

import tiktoken

from .stage_a_budget_controller import BudgetController
from .stage_b_context_filter import ContextFilter
from .stage_c_sentence_filter import SentenceFilter
from .stage_d_token_compress import TokenCompressor
from .stage_e_recovery import CompressionResult, ResponseRecovery


class LLMLinguaPipeline:
    """
    End-to-end LLMLingua prompt compression pipeline.

    Implements the three-level coarse-to-fine compression strategy:
        1. Context-level: select informative demonstrations
        2. Sentence-level: filter redundant sentences
        3. Token-level: iteratively prune low-information tokens

    Args:
        config: Configuration dictionary with keys for each stage.
            Expected structure:
            {
                "model_name": "gpt2",           # Small LM for perplexity
                "device": "cpu",                 # "cpu" or "cuda"
                "rate": 0.5,                     # Global compression rate
                "target_token": -1,              # Or specify target tokens
                "budget_controller": {...},       # Stage A config
                "context_filter": {...},          # Stage B config
                "sentence_filter": {...},         # Stage C config
                "token_compressor": {...},        # Stage D config
                "recovery": {...},               # Stage E config
            }
    """

    def __init__(self, config: dict):
        self.config = config
        self._model_name = config.get("model_name", "gpt2")
        self._device = config.get("device", "cpu")
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # Inject shared model config into each stage
        shared = {"model_name": self._model_name, "device": self._device}

        # Initialize stages
        budget_cfg = {**shared, **config.get("budget_controller", {})}
        budget_cfg.setdefault("rate", config.get("rate", 0.5))
        budget_cfg.setdefault("target_token", config.get("target_token", -1))
        self.budget_controller = BudgetController(budget_cfg)

        context_cfg = {**shared, **config.get("context_filter", {})}
        self.context_filter = ContextFilter(context_cfg)

        sentence_cfg = {**shared, **config.get("sentence_filter", {})}
        self.sentence_filter = SentenceFilter(sentence_cfg)

        token_cfg = {**shared, **config.get("token_compressor", {})}
        token_cfg.setdefault("rate", config.get("rate", 0.5))
        self.token_compressor = TokenCompressor(token_cfg)

        recovery_cfg = config.get("recovery", {})
        self.recovery = ResponseRecovery(recovery_cfg)

    def compress(
        self,
        contexts: Union[str, List[str]],
        instruction: str = "",
        question: str = "",
        rate: Optional[float] = None,
        target_token: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress a prompt using the full LLMLingua pipeline.

        Args:
            contexts: Single context string or list of context strings.
            instruction: Instruction text (preserved with minimal compression).
            question: Question text (preserved with minimal compression).
            rate: Override compression rate (0 < rate <= 1.0).
            target_token: Override target token count.

        Returns:
            CompressionResult with compressed prompt and statistics.
        """
        # Normalize input
        if isinstance(contexts, str):
            contexts = [contexts]
        if not contexts:
            contexts = [" "]

        # Override rate/target if provided
        if rate is not None:
            self.budget_controller.rate = rate
            self.token_compressor.target_ratio = rate
        if target_token is not None:
            self.budget_controller.target_token = target_token

        # Compute original token count
        full_prompt = "\n\n".join([instruction] + contexts + [question]).strip()
        original_tokens = len(self._tokenizer.encode(full_prompt))

        # ─── Stage A: Budget Allocation ───
        budget = self.budget_controller.allocate(
            instruction=instruction,
            contexts=contexts,
            question=question,
        )

        # ─── Stage B: Context-Level Filter ───
        context_tokens = [len(self._tokenizer.encode(c)) for c in contexts]
        ranked_contexts, selected_indices = self.context_filter.filter(
            contexts=contexts,
            question=question,
            context_tokens=context_tokens,
            target_token=budget.context_budget,
        )
        filtered_contexts = [rc.text for rc in ranked_contexts]

        # ─── Stage C: Sentence-Level Filter ───
        filtered_contexts, _ = self.sentence_filter.filter(
            contexts=filtered_contexts,
            target_token=budget.context_budget,
            question=question,
        )

        # ─── Stage D: Token-Level Compression ───
        compressed_contexts = []
        for i, ctx in enumerate(filtered_contexts):
            per_ctx_budget = (
                budget.per_context_budget[selected_indices[i]]
                if i < len(selected_indices)
                and selected_indices[i] < len(budget.per_context_budget)
                else -1
            )
            result = self.token_compressor.compress(
                text=ctx,
                target_token=per_ctx_budget if per_ctx_budget > 0 else -1,
                dynamic_ratios=None,
            )
            compressed_contexts.append(result.compressed_text)

        # ─── Stage E: Assemble & Statistics ───
        final_result = self.recovery.assemble(
            instruction=instruction,
            compressed_contexts=compressed_contexts,
            question=question,
            original_token_count=original_tokens,
        )

        return final_result

    def compress_structured(
        self,
        structured_prompt: str,
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
    ) -> CompressionResult:
        """
        Compress a structured prompt with <llmlingua> tags.

        Each segment can specify its own compression rate:
            <llmlingua, rate=0.4>text to compress</llmlingua>
            <llmlingua, compress=False>text to keep</llmlingua>

        Args:
            structured_prompt: Prompt with <llmlingua> tags.
            instruction: Instruction text.
            question: Question text.
            rate: Default compression rate for untagged segments.

        Returns:
            CompressionResult with compressed prompt.
        """
        parsed = self.recovery.parse_structured_prompt(structured_prompt, rate)
        segments = parsed["segments"]
        rates = parsed["rates"]
        compress_flags = parsed["compress_flags"]

        original_text = "".join(segments)
        original_tokens = len(self._tokenizer.encode(
            "\n\n".join([instruction, original_text, question]).strip()
        ))

        compressed_segments = []
        for seg, seg_rate, do_compress in zip(segments, rates, compress_flags):
            if not do_compress:
                compressed_segments.append(seg)
            else:
                result = self.token_compressor.compress(
                    text=seg,
                    target_token=int(len(self._tokenizer.encode(seg)) * seg_rate),
                )
                compressed_segments.append(result.compressed_text)

        compressed_contexts = ["".join(compressed_segments)]
        return self.recovery.assemble(
            instruction=instruction,
            compressed_contexts=compressed_contexts,
            question=question,
            original_token_count=original_tokens,
        )

    def recover_response(
        self,
        original_prompt: str,
        compressed_prompt: str,
        response: str,
    ) -> str:
        """
        Recover original text from a response generated from compressed prompt.

        Args:
            original_prompt: The full original prompt.
            compressed_prompt: The compressed prompt sent to the LLM.
            response: The LLM's response.

        Returns:
            Recovered response string.
        """
        return self.recovery.recover(original_prompt, compressed_prompt, response)
