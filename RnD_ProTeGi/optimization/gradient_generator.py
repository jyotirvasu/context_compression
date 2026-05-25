"""
Gradient Generator
Analyzes evaluation errors to generate textual gradients (improvement suggestions).
Uses LLM to understand why prompts fail and suggest improvements.

Based on: "Automatic Prompt Optimization with Gradient Descent and Beam Search"
(Pryzant et al., EMNLP 2023)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock

from ..llm import BaseLLMProvider
from ..evaluation.evaluator import EvaluationError


@dataclass
class GradientResult:
    """Result of gradient generation."""
    gradient: str
    num_errors_analyzed: int
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"GradientResult(gradient='{self.gradient[:50]}...', errors={self.num_errors_analyzed})"


class GradientGenerator:
    """
    Generates textual gradients from evaluation errors.

    The gradient is a natural language description of why the prompt failed
    and what patterns it's missing. This guides prompt improvement.

    Analogous to numerical gradients in neural network training:
    - Forward pass = evaluate prompt on data
    - Loss = classification errors
    - Gradient = LLM-generated error analysis
    - Backward pass = edit prompt based on gradient
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        max_errors_to_analyze: int = 5,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.max_errors_to_analyze = max_errors_to_analyze
        self.temperature = temperature
        self._lock = Lock()
        self._stats = {"total_gradients": 0, "total_errors_analyzed": 0}

    def generate(
        self,
        prompt: str,
        errors: List[EvaluationError],
        task_description: Optional[str] = None,
    ) -> GradientResult:
        """
        Generate a textual gradient from evaluation errors.

        Args:
            prompt: Current prompt that produced errors
            errors: List of classification errors
            task_description: Optional task context

        Returns:
            GradientResult with improvement suggestion
        """
        if not errors:
            raise ValueError("Cannot generate gradient from empty error list")

        errors_to_analyze = errors[:self.max_errors_to_analyze]

        gradient_prompt = self._build_gradient_prompt(
            prompt, errors_to_analyze, task_description
        )

        response = self.provider.complete(
            gradient_prompt, temperature=self.temperature
        )

        gradient = response.content.strip()

        with self._lock:
            self._stats["total_gradients"] += 1
            self._stats["total_errors_analyzed"] += len(errors_to_analyze)

        return GradientResult(
            gradient=gradient,
            num_errors_analyzed=len(errors_to_analyze),
            metadata={"prompt_tokens": response.tokens_used},
        )

    def _build_gradient_prompt(
        self,
        prompt: str,
        errors: List[EvaluationError],
        task_description: Optional[str],
    ) -> str:
        """Build the prompt that asks LLM to analyze failures."""
        error_examples = []
        for i, error in enumerate(errors, 1):
            error_examples.append(
                f"{i}. Text: \"{error.item.text[:100]}\"\n"
                f"   True label: {error.item.label}\n"
                f"   Predicted: {error.predicted_label}"
            )
        error_text = "\n".join(error_examples)

        task_context = ""
        if task_description:
            task_context = f"\nTask: {task_description}\n"

        gradient_prompt = f"""You are analyzing why a classification prompt is making errors.
{task_context}
Current Prompt: "{prompt}"

The prompt failed on these examples:
{error_text}

Analyze these errors and identify the common pattern or issue. What is the prompt missing? What pattern does it fail to recognize?

Provide a concise explanation (2-3 sentences) of why the prompt failed and what it needs to detect. Focus on the specific pattern or characteristic it's missing.

Analysis:"""

        return gradient_prompt

    def get_statistics(self) -> Dict[str, int]:
        with self._lock:
            return self._stats.copy()
