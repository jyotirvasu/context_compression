"""
Metrics for evaluating classification performance.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    confusion_matrix: Dict[Tuple[str, str], int]
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None


def calculate_metrics(
    true_labels: List[str],
    predicted_labels: List[str],
    positive_label: Optional[str] = None,
) -> ClassificationMetrics:
    """
    Calculate classification metrics.

    Handles both binary and multi-class classification.
    Uses macro-averaging for multi-class.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Label lists must have same length")

    all_labels = set(true_labels + predicted_labels)

    # Build confusion matrix
    confusion_matrix = defaultdict(int)
    for true_label, pred_label in zip(true_labels, predicted_labels):
        confusion_matrix[(true_label, pred_label)] += 1

    # Accuracy
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    accuracy = correct / len(true_labels)

    if len(all_labels) == 2:
        # Binary classification
        if positive_label is None:
            positive_label = sorted(all_labels)[0]
        else:
            positive_label = str(positive_label).strip().lower()

        negative_label = (all_labels - {positive_label}).pop()

        tp = confusion_matrix.get((positive_label, positive_label), 0)
        fp = confusion_matrix.get((negative_label, positive_label), 0)
        tn = confusion_matrix.get((negative_label, negative_label), 0)
        fn = confusion_matrix.get((positive_label, negative_label), 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics = None
    else:
        # Multi-class: macro-averaged metrics
        per_class_metrics = {}
        precisions, recalls, f1s = [], [], []

        for label in all_labels:
            label_tp = confusion_matrix.get((label, label), 0)
            label_fp = sum(
                confusion_matrix.get((t, label), 0)
                for t in all_labels if t != label
            )
            label_fn = sum(
                confusion_matrix.get((label, p), 0)
                for p in all_labels if p != label
            )

            label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0.0
            label_recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) > 0 else 0.0
            label_f1 = (
                2 * (label_precision * label_recall) / (label_precision + label_recall)
                if (label_precision + label_recall) > 0 else 0.0
            )

            per_class_metrics[label] = {
                "precision": label_precision,
                "recall": label_recall,
                "f1": label_f1,
                "support": sum(1 for t in true_labels if t == label),
            }
            precisions.append(label_precision)
            recalls.append(label_recall)
            f1s.append(label_f1)

        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        f1 = sum(f1s) / len(f1s) if f1s else 0.0

        tp = sum(confusion_matrix.get((l, l), 0) for l in all_labels)
        fp = sum(
            confusion_matrix.get((t, p), 0)
            for t in all_labels for p in all_labels if t != p
        )
        fn = fp
        tn = 0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        confusion_matrix=dict(confusion_matrix),
        per_class_metrics=per_class_metrics,
    )


def print_metrics_report(metrics: ClassificationMetrics) -> str:
    """Generate formatted metrics report."""
    lines = [
        "=" * 50,
        "CLASSIFICATION METRICS",
        "=" * 50,
        f"Accuracy:  {metrics.accuracy:.4f}",
        f"Precision: {metrics.precision:.4f}",
        f"Recall:    {metrics.recall:.4f}",
        f"F1 Score:  {metrics.f1:.4f}",
    ]
    if metrics.per_class_metrics:
        lines.append("\nPer-class metrics:")
        for label, m in metrics.per_class_metrics.items():
            lines.append(f"  {label}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    return "\n".join(lines)
