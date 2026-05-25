"""
Evaluation Module
Provides tools for evaluating classification prompts on datasets.
"""

from .dataset import DatasetItem, ClassificationDataset, create_spam_dataset
from .metrics import ClassificationMetrics, calculate_metrics, print_metrics_report
from .evaluator import EvaluationError, EvaluationResult, PromptEvaluator

__all__ = [
    "DatasetItem",
    "ClassificationDataset",
    "create_spam_dataset",
    "ClassificationMetrics",
    "calculate_metrics",
    "print_metrics_report",
    "EvaluationError",
    "EvaluationResult",
    "PromptEvaluator",
]
