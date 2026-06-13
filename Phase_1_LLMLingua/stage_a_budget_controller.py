"""
Stage A: Budget Controller
---------------------------
Allocates the overall token compression budget across prompt components
based on their sensitivity to compression (LLMLingua, Jiang et al. 2023).

Key insight: Instructions and questions are highly sensitive to compression
and should be preserved, while demonstrations/context are less sensitive
and absorb most of the compression.

The budget controller computes per-component target tokens given:
    - A global target compression rate or target_token count
    - Token counts of instruction, context list, and question
    - Optional forced retention of specific contexts

Reference:
    "LLMLingua: Compressing Prompts for Accelerated Inference of Large
     Language Models" (EMNLP 2023)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken


@dataclass
class BudgetAllocation:
    """Result of budget allocation across prompt components."""
    instruction_tokens: int
    question_tokens: int
    context_budget: int
    total_target: int
    rate: float
    per_context_budget: List[int] = field(default_factory=list)
    dynamic_ratios: List[float] = field(default_factory=list)


class BudgetController:
    """Distributes compression budget across prompt components."""

    def __init__(self, config: dict):
        self.rate = config.get("rate", 0.5)
        self.target_token = config.get("target_token", -1)
        self.instruction_rate = config.get("instruction_rate", 1.0)
        self.question_rate = config.get("question_rate", 1.0)
        self.context_budget_expr = config.get("context_budget", "+100")
        self.dynamic_context_compression_ratio = config.get(
            "dynamic_context_compression_ratio", 0.0
        )
        self.concate_question = config.get("concate_question", True)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def allocate(
        self,
        instruction: str,
        contexts: List[str],
        question: str,
        force_context_ids: Optional[List[int]] = None,
    ) -> BudgetAllocation:
        """
        Compute token budget allocation for each prompt component.

        Instruction and question are preserved (low compression).
        Context absorbs the bulk of compression.
        """
        instruction_tokens = self._count_tokens(instruction)
        question_tokens = self._count_tokens(question)
        context_tokens = [self._count_tokens(c) for c in contexts]
        total_original = instruction_tokens + question_tokens + sum(context_tokens)

        # Determine effective target token count
        if self.target_token > 0:
            target_token = self.target_token
            effective_rate = target_token / total_original if total_original > 0 else 1.0
        else:
            effective_rate = self.rate
            target_token = int(total_original * effective_rate)

        # Instruction and question are minimally compressed
        instruction_budget = int(instruction_tokens * self.instruction_rate)
        question_budget = int(question_tokens * self.question_rate)

        # Remaining budget goes to context
        context_budget = target_token - instruction_budget
        if self.concate_question:
            context_budget -= question_budget
        context_budget = max(context_budget, 0)

        # Apply context_budget expression adjustment
        context_budget = self._apply_budget_expr(context_budget)

        # Per-context budget (proportional to original token counts)
        per_context_budget = self._distribute_context_budget(
            context_tokens, context_budget, force_context_ids
        )

        # Dynamic compression ratios (linearly varying for LongLLMLingua)
        dynamic_ratios = self._compute_dynamic_ratios(len(contexts))

        return BudgetAllocation(
            instruction_tokens=instruction_budget,
            question_tokens=question_budget,
            context_budget=context_budget,
            total_target=target_token,
            rate=effective_rate,
            per_context_budget=per_context_budget,
            dynamic_ratios=dynamic_ratios,
        )

    def _apply_budget_expr(self, budget: int) -> int:
        """Apply budget expression like '+100' or '*1.5'."""
        expr = self.context_budget_expr.strip()
        if expr.startswith("+"):
            return budget + int(expr[1:])
        elif expr.startswith("*"):
            return int(budget * float(expr[1:]))
        elif expr.startswith("-"):
            return max(0, budget - int(expr[1:]))
        return budget

    def _distribute_context_budget(
        self,
        context_tokens: List[int],
        total_budget: int,
        force_context_ids: Optional[List[int]],
    ) -> List[int]:
        """Distribute budget proportionally across contexts."""
        total_ctx_tokens = sum(context_tokens)
        if total_ctx_tokens == 0:
            return [0] * len(context_tokens)

        per_context = []
        for i, ct in enumerate(context_tokens):
            if force_context_ids and i in force_context_ids:
                per_context.append(ct)  # forced contexts get full budget
            else:
                proportion = ct / total_ctx_tokens
                per_context.append(int(total_budget * proportion))

        return per_context

    def _compute_dynamic_ratios(self, n_contexts: int) -> List[float]:
        """Compute linearly varying dynamic compression ratios (LongLLMLingua)."""
        if self.dynamic_context_compression_ratio <= 0 or n_contexts <= 1:
            return [0.0] * n_contexts

        ratio = self.dynamic_context_compression_ratio
        return [
            i * (ratio / (n_contexts - 1))
            for i in range(-(n_contexts - 1), n_contexts, 2)
        ][::-1]

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
