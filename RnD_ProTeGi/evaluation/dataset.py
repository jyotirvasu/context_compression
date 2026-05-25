"""
Dataset structures for prompt evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import random


@dataclass
class DatasetItem:
    """Single item in a classification dataset."""
    text: str
    label: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not self.label or not self.label.strip():
            raise ValueError("Label cannot be empty")
        self.label = self.label.strip().lower()

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"DatasetItem(text='{text_preview}', label='{self.label}')"


@dataclass
class ClassificationDataset:
    """Dataset for classification tasks."""
    name: str
    items: List[DatasetItem]
    description: str = ""

    def __post_init__(self):
        if not self.items:
            raise ValueError("Dataset cannot be empty")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    @property
    def labels(self) -> Set[str]:
        """Get all unique labels."""
        return {item.label for item in self.items}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def label_distribution(self) -> Dict[str, int]:
        """Get count of each label."""
        return dict(Counter(item.label for item in self.items))

    def sample(self, n: int, seed: int = 42) -> "ClassificationDataset":
        """Random sample of n items."""
        random.seed(seed)
        sampled = random.sample(self.items, min(n, len(self.items)))
        return ClassificationDataset(
            name=f"{self.name}_sample_{n}",
            items=sampled,
        )

    def split(self, train_ratio: float = 0.8, seed: int = 42):
        """Split into train/test."""
        items = self.items.copy()
        random.seed(seed)
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train = ClassificationDataset(name=f"{self.name}_train", items=items[:split_idx])
        test = ClassificationDataset(name=f"{self.name}_test", items=items[split_idx:])
        return train, test


def create_spam_dataset() -> ClassificationDataset:
    """Built-in spam detection dataset for testing."""
    return ClassificationDataset(
        name="spam_detection",
        items=[
            DatasetItem("Congratulations! You've won a $1000 gift card!", "spam"),
            DatasetItem("URGENT: Your account will be suspended. Click here.", "spam"),
            DatasetItem("Get rich quick! Make $5000 from home!", "spam"),
            DatasetItem("Your package is waiting. Confirm delivery address.", "spam"),
            DatasetItem("Limited time offer: 50% off your next purchase", "spam"),
            DatasetItem("Reminder: Your appointment is tomorrow at 3pm", "not_spam"),
            DatasetItem("Your order #12345 has shipped. Track your package.", "not_spam"),
            DatasetItem("Weekly team meeting moved to Thursday 2pm", "not_spam"),
            DatasetItem("Action required: Your password will expire in 24 hours", "not_spam"),
            DatasetItem("Your flight departs in 2 hours. Check-in now.", "not_spam"),
        ],
        description="Spam detection with tricky examples",
    )
