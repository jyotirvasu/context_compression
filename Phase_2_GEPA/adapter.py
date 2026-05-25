"""
GEPA Adapter Interface.
Defines how the optimization loop connects to tasks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AdapterEvalResult:
    """Evaluation result from the adapter."""
    scores: List[float]
    outputs: List[Any]
    trajectories: List[str]  # Full execution traces for reflection
    num_metric_calls: int
    objective_scores: Optional[List[Dict[str, float]]] = None


class BaseAdapter(ABC):
    """
    Adapter interface connecting GEPA to an arbitrary task.

    The adapter defines:
    1. How to evaluate a candidate prompt on a batch of data
    2. How to build a reflective dataset from traces
    3. Optionally, custom proposal logic

    Users implement this to connect GEPA to their specific task.
    """

    @abstractmethod
    def evaluate(
        self,
        batch: List[Any],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> AdapterEvalResult:
        """
        Evaluate a candidate on a batch of data instances.

        Args:
            batch: List of data instances to evaluate on.
            candidate: Dict mapping component names to their text content.
            capture_traces: If True, capture detailed execution traces
                           (LLM inputs/outputs, errors, tool calls, etc.)

        Returns:
            AdapterEvalResult with scores, outputs, and traces.
        """
        ...

    @abstractmethod
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_result: AdapterEvalResult,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a reflective dataset from evaluation traces.

        This creates the "Actionable Side Information" (ASI) that the
        reflector LLM will use to diagnose failures and propose improvements.

        Args:
            candidate: The evaluated candidate.
            eval_result: Evaluation results including traces.
            components_to_update: Which components will be mutated.

        Returns:
            Dict mapping component_name -> list of trace dictionaries.
            Each trace dict should contain:
              - "input": the original input
              - "output": what the candidate produced
              - "expected": ground truth (if available)
              - "score": numeric score
              - "trace": full execution trace string
              - "error": any error message (if applicable)
        """
        ...


class DefaultAdapter(BaseAdapter):
    """
    Default adapter for simple prompt optimization tasks.

    Uses a scoring function and an LLM execution function provided at init.
    """

    def __init__(
        self,
        execute_fn=None,
        score_fn=None,
        mock_mode: bool = False,
    ):
        """
        Args:
            execute_fn: Function(input, candidate) -> (output, trace)
            score_fn: Function(output, expected) -> float score
            mock_mode: If True, use mock evaluation for testing
        """
        self.execute_fn = execute_fn
        self.score_fn = score_fn
        self.mock_mode = mock_mode

    def evaluate(
        self,
        batch: List[Any],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> AdapterEvalResult:
        scores = []
        outputs = []
        trajectories = []

        for item in batch:
            if self.mock_mode:
                output, trace, score = self._mock_evaluate(item, candidate)
            else:
                output, trace = self.execute_fn(item, candidate)
                expected = item.get("expected", item.get("label", ""))
                score = self.score_fn(output, expected)

            scores.append(score)
            outputs.append(output)
            if capture_traces:
                trajectories.append(trace)

        return AdapterEvalResult(
            scores=scores,
            outputs=outputs,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def _mock_evaluate(self, item, candidate: Dict[str, str]) -> tuple:
        """Mock evaluation for testing without LLM calls."""
        import random

        # Simulate evaluation with some randomness
        prompt_text = " ".join(candidate.values())
        input_text = str(item.get("input", ""))

        # Higher scores for longer, more detailed prompts
        base_score = min(len(prompt_text) / 200.0, 0.8)
        noise = random.uniform(-0.1, 0.2)
        score = max(0.0, min(1.0, base_score + noise))

        output = f"[Mock output for: {input_text[:50]}]"
        trace = (
            f"=== Execution Trace ===\n"
            f"Prompt: {prompt_text[:100]}...\n"
            f"Input: {input_text[:100]}\n"
            f"Output: {output}\n"
            f"Score: {score:.3f}\n"
            f"{'ERROR: Score below threshold' if score < 0.5 else 'OK'}\n"
        )
        return output, trace, score

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_result: AdapterEvalResult,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        dataset = {}
        for component in components_to_update:
            entries = []
            for i, (score, output, trace) in enumerate(
                zip(eval_result.scores, eval_result.outputs, eval_result.trajectories)
            ):
                entries.append({
                    "input": f"instance_{i}",
                    "output": output,
                    "score": score,
                    "trace": trace,
                    "current_instruction": candidate.get(component, ""),
                    "needs_improvement": score < 0.7,
                })
            dataset[component] = entries
        return dataset
