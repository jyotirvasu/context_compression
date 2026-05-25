"""
Prompt Evaluator
Evaluates classification prompts using LLM providers with caching.
"""

import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..llm import BaseLLMProvider
from .dataset import ClassificationDataset, DatasetItem
from .metrics import ClassificationMetrics, calculate_metrics


@dataclass
class EvaluationError:
    """Single evaluation error for gradient generation."""
    item: DatasetItem
    predicted_label: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.item.text,
            "true_label": self.item.label,
            "predicted_label": self.predicted_label,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt on a dataset."""
    prompt: str
    dataset_name: str
    metrics: ClassificationMetrics
    errors: List[EvaluationError]
    predictions: List[Tuple[str, str]]
    eval_time: float = 0.0
    num_api_calls: int = 0
    cache_hits: int = 0

    @property
    def score(self) -> float:
        return self.metrics.f1

    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy


class PromptEvaluator:
    """
    Evaluates classification prompts on datasets.

    Features:
    - Caching to avoid redundant API calls
    - Tracks errors for gradient generation
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache_enabled: bool = True,
        verbose: bool = False,
    ):
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.verbose = verbose
        self._cache: Dict[Tuple[str, str], str] = {}

    def evaluate(
        self,
        prompt: str,
        dataset: ClassificationDataset,
        positive_label: Optional[str] = None,
        metric: str = "f1",
    ) -> EvaluationResult:
        """
        Evaluate a prompt on the full dataset.

        Returns EvaluationResult with metrics and list of errors.
        """
        start_time = time.time()
        valid_labels = list(dataset.labels)

        true_labels = []
        predicted_labels = []
        errors = []
        api_calls = 0
        cache_hits = 0

        for item in dataset:
            # Check cache
            cache_key = (self._hash(prompt), self._hash(item.text))
            if self.cache_enabled and cache_key in self._cache:
                predicted = self._cache[cache_key]
                cache_hits += 1
            else:
                predicted = self.provider.classify(prompt, item.text, valid_labels)
                predicted = predicted.strip().lower()
                if self.cache_enabled:
                    self._cache[cache_key] = predicted
                api_calls += 1

            true_labels.append(item.label)
            predicted_labels.append(predicted)

            if predicted != item.label:
                errors.append(EvaluationError(item=item, predicted_label=predicted))

        metrics = calculate_metrics(true_labels, predicted_labels, positive_label)
        eval_time = time.time() - start_time

        if self.verbose:
            print(f"  Eval: F1={metrics.f1:.3f} Acc={metrics.accuracy:.3f} "
                  f"({api_calls} API calls, {cache_hits} cached)")

        return EvaluationResult(
            prompt=prompt,
            dataset_name=dataset.name,
            metrics=metrics,
            errors=errors,
            predictions=list(zip(true_labels, predicted_labels)),
            eval_time=eval_time,
            num_api_calls=api_calls,
            cache_hits=cache_hits,
        )

    def evaluate_score(
        self,
        prompt: str,
        dataset: ClassificationDataset,
        metric: str = "f1",
    ) -> float:
        """Evaluate and return just the score."""
        result = self.evaluate(prompt, dataset)
        if metric == "f1":
            return result.metrics.f1
        elif metric == "accuracy":
            return result.metrics.accuracy
        elif metric == "precision":
            return result.metrics.precision
        elif metric == "recall":
            return result.metrics.recall
        return result.metrics.f1

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
