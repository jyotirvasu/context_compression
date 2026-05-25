"""
AutoPrompt Example: Full Pipeline Demonstration
================================================

This example demonstrates the AutoPrompt pipeline for sentiment analysis.
It can run in two modes:
1. Full mode: Downloads a BERT model and runs actual gradient-guided search
2. Mock mode: Demonstrates the algorithm logic without requiring model download

Usage:
    # Full mode (requires model download):
    python autoprompt_example.py --mode full

    # Mock mode (no downloads needed):
    python autoprompt_example.py --mode mock
"""

import sys
import json
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MOCK MODE: Demonstrates the algorithm without model downloads
# ==============================================================================

class MockVocabulary:
    """Simulates a tokenizer vocabulary for demonstration."""

    def __init__(self, size: int = 1000):
        self.size = size
        # Create a small vocabulary with meaningful tokens
        self.tokens = [
            # Special tokens
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[T]", "[P]",
            # Sentiment-relevant tokens
            "good", "great", "excellent", "wonderful", "amazing",
            "bad", "terrible", "awful", "horrible", "disgusting",
            # Neutral/common tokens
            "the", "a", "is", "was", "are", "this", "that", "it",
            "movie", "film", "book", "food", "place", "service",
            "very", "really", "quite", "somewhat", "extremely",
            "not", "never", "no", "nothing", "nobody",
            # Trigger-effective tokens (simulated)
            "absolutely", "undoubtedly", "certainly", "obviously", "clearly",
            "unfortunately", "sadly", "regrettably", "disappointingly",
        ]
        # Pad vocabulary to specified size
        while len(self.tokens) < size:
            self.tokens.append(f"token_{len(self.tokens)}")

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, 1)  # UNK=1

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return "[UNK]"


class MockEmbeddings:
    """Simulates token embeddings with sentiment-biased vectors."""

    def __init__(self, vocab: MockVocabulary, dim: int = 64):
        self.vocab = vocab
        self.dim = dim
        torch.manual_seed(42)
        self.weight = torch.randn(vocab.size, dim)

        # Bias positive tokens toward positive direction
        positive_tokens = ["good", "great", "excellent", "wonderful", "amazing",
                          "absolutely", "undoubtedly", "certainly", "obviously", "clearly"]
        negative_tokens = ["bad", "terrible", "awful", "horrible", "disgusting",
                          "unfortunately", "sadly", "regrettably", "disappointingly"]

        positive_direction = torch.randn(dim)
        positive_direction = positive_direction / positive_direction.norm()

        for token in positive_tokens:
            idx = vocab.encode(token)
            self.weight[idx] += 2.0 * positive_direction

        for token in negative_tokens:
            idx = vocab.encode(token)
            self.weight[idx] -= 2.0 * positive_direction


class MockAutoPrompt:
    """
    Demonstrates the AutoPrompt algorithm logic without a real model.

    This mock implementation shows the key steps:
    1. Template parsing
    2. Gradient computation (simulated)
    3. HotFlip candidate selection
    4. Candidate evaluation
    5. Iterative optimization
    """

    def __init__(self, vocab_size: int = 1000, emb_dim: int = 64):
        self.vocab = MockVocabulary(vocab_size)
        self.embeddings = MockEmbeddings(self.vocab, emb_dim)
        self.emb_dim = emb_dim

        # Simulated "learned" classification direction
        torch.manual_seed(42)
        pos_direction = torch.randn(emb_dim)
        self.class_direction = pos_direction / pos_direction.norm()

    def simulate_gradient(self, trigger_ids: List[int], labels: List[str]) -> torch.Tensor:
        """
        Simulate the gradient at trigger positions.

        In real AutoPrompt, this is computed via backpropagation through the model.
        Here we simulate it as the direction that would help classification.
        """
        # The gradient points in the direction that would improve classification
        # For positive examples: gradient aligns with positive direction
        # For negative examples: gradient aligns with negative direction
        num_triggers = len(trigger_ids)
        grad = torch.zeros(num_triggers, self.emb_dim)

        pos_count = labels.count("positive")
        neg_count = labels.count("negative")

        # Net gradient direction based on class balance in batch
        if pos_count > neg_count:
            direction = self.class_direction
        else:
            direction = -self.class_direction

        # Add noise to simulate real gradients
        for i in range(num_triggers):
            noise = torch.randn(self.emb_dim) * 0.3
            grad[i] = direction + noise

        return grad

    def hotflip_attack(self, grad: torch.Tensor, num_candidates: int = 5) -> List[int]:
        """
        Find tokens whose embeddings align with the gradient.
        This is the core of AutoPrompt's search.
        """
        # dot product: embedding_matrix @ grad
        scores = torch.matmul(self.embeddings.weight, grad)

        # Filter special tokens
        special_ids = [self.vocab.encode(t) for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[T]", "[P]"]]
        for idx in special_ids:
            scores[idx] = -float('inf')

        # Get top-k candidates
        _, top_k = scores.topk(num_candidates)
        return top_k.tolist()

    def evaluate_trigger(self, trigger_tokens: List[str], instances: List[Dict]) -> float:
        """
        Evaluate accuracy of current trigger tokens on instances.
        Simulates how well the trigger causes correct predictions.
        """
        correct = 0
        total = len(instances)

        # Compute trigger embedding sum (simulates trigger influence)
        trigger_embedding = torch.zeros(self.emb_dim)
        for token in trigger_tokens:
            idx = self.vocab.encode(token)
            trigger_embedding += self.embeddings.weight[idx]

        # Dot product with classification direction gives prediction score
        trigger_score = torch.dot(trigger_embedding, self.class_direction).item()

        for instance in instances:
            label = instance["label"]
            # Positive trigger score -> predicts positive
            # Negative trigger score -> predicts negative
            predicted = "positive" if trigger_score > 0 else "negative"

            # Add some instance-level noise for realism
            noise = random.gauss(0, 0.5)
            adjusted_score = trigger_score + noise
            predicted = "positive" if adjusted_score > 0 else "negative"

            if predicted == label:
                correct += 1

        return correct / total if total > 0 else 0.0

    def search(
        self,
        train_data: List[Dict],
        dev_data: List[Dict],
        num_triggers: int = 3,
        num_candidates: int = 5,
        iters: int = 10,
    ) -> Dict:
        """
        Run the mock AutoPrompt search.

        Demonstrates the algorithm:
        1. Initialize trigger tokens as [MASK]
        2. For each iteration:
           a. Compute gradient (simulated)
           b. Select candidate replacements via HotFlip
           c. Evaluate each candidate
           d. Keep best if improved
        """
        print("\n" + "=" * 70)
        print("AutoPrompt: Gradient-Guided Trigger Token Search (Mock Demo)")
        print("=" * 70)
        print(f"\nTemplate: [T] [T] [T] {{sentence}} [P] .")
        print(f"Number of trigger tokens: {num_triggers}")
        print(f"Candidates per position: {num_candidates}")
        print(f"Search iterations: {iters}")
        print(f"Train size: {len(train_data)}, Dev size: {len(dev_data)}")

        # Initialize triggers
        trigger_ids = [self.vocab.encode("[MASK]")] * num_triggers
        trigger_tokens = ["[MASK]"] * num_triggers
        best_trigger_tokens = trigger_tokens.copy()
        best_dev_acc = 0.0
        history = []

        # Initial evaluation
        dev_acc = self.evaluate_trigger(trigger_tokens, dev_data)
        print(f"\nInitial triggers: {trigger_tokens}")
        print(f"Initial dev accuracy: {dev_acc:.4f}")
        history.append((0, dev_acc))

        print(f"\n{'─' * 70}")
        print(f"{'Iter':>4} │ {'Position':>8} │ {'Old Token':>12} │ {'New Token':>12} │ {'Dev Acc':>8}")
        print(f"{'─' * 70}")

        start_time = time.time()

        for iteration in range(1, iters + 1):
            # Step 1: Get labels from a batch
            batch = random.sample(train_data, min(32, len(train_data)))
            batch_labels = [inst["label"] for inst in batch]

            # Step 2: Compute gradient (simulated)
            grad = self.simulate_gradient(trigger_ids, batch_labels)

            # Step 3: Pick random position to optimize
            pos = random.randrange(num_triggers)

            # Step 4: HotFlip attack - find candidates
            candidates = self.hotflip_attack(grad[pos], num_candidates)

            # Step 5: Evaluate candidates
            best_candidate = None
            best_score = self.evaluate_trigger(trigger_tokens, batch)

            for cand_id in candidates:
                temp_tokens = trigger_tokens.copy()
                temp_tokens[pos] = self.vocab.decode(cand_id)
                score = self.evaluate_trigger(temp_tokens, batch)
                if score > best_score:
                    best_score = score
                    best_candidate = cand_id

            # Step 6: Update if improved
            old_token = trigger_tokens[pos]
            if best_candidate is not None:
                trigger_ids[pos] = best_candidate
                trigger_tokens[pos] = self.vocab.decode(best_candidate)

                # Evaluate on dev
                dev_acc = self.evaluate_trigger(trigger_tokens, dev_data)
                history.append((iteration, dev_acc))

                print(f"{iteration:>4} │ {pos:>8} │ {old_token:>12} │ {trigger_tokens[pos]:>12} │ {dev_acc:>8.4f}")

                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_trigger_tokens = trigger_tokens.copy()

        elapsed = time.time() - start_time

        print(f"{'─' * 70}")
        print(f"\nSearch completed in {elapsed:.2f}s")
        print(f"\nFinal trigger tokens: {trigger_tokens}")
        print(f"Best trigger tokens:  {best_trigger_tokens}")
        print(f"Best dev accuracy:    {best_dev_acc:.4f}")

        # Show final prompt example
        print(f"\nExample generated prompt:")
        example = dev_data[0]
        prompt = f"  [CLS] {' '.join(best_trigger_tokens)} {example['sentence']} [MASK] . [SEP]"
        print(f"  {prompt}")
        print(f"  True label: {example['label']}")

        return {
            'best_tokens': best_trigger_tokens,
            'best_dev_metric': best_dev_acc,
            'history': history,
            'elapsed_time': elapsed,
        }


def run_mock_demo():
    """Run the mock demonstration of AutoPrompt."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " AutoPrompt Pipeline - Mock Demonstration ".center(68) + "║")
    print("║" + " (No model download required) ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Create synthetic data
    positive_sentences = [
        "This movie is absolutely wonderful",
        "I really loved this film",
        "An excellent experience overall",
        "The quality is outstanding",
        "Highly recommended to everyone",
        "Best thing I have ever seen",
        "Truly remarkable work",
        "A masterpiece of cinema",
        "Brilliantly crafted story",
        "Superb acting throughout",
    ]
    negative_sentences = [
        "This movie is absolutely terrible",
        "I really hated this film",
        "A horrible experience overall",
        "The quality is atrocious",
        "Would never recommend this",
        "Worst thing I have ever seen",
        "Truly dreadful work",
        "A disaster of cinema",
        "Poorly crafted story",
        "Terrible acting throughout",
    ]

    train_data = []
    for _ in range(50):
        train_data.append({"sentence": random.choice(positive_sentences), "label": "positive"})
        train_data.append({"sentence": random.choice(negative_sentences), "label": "negative"})

    dev_data = []
    for _ in range(25):
        dev_data.append({"sentence": random.choice(positive_sentences), "label": "positive"})
        dev_data.append({"sentence": random.choice(negative_sentences), "label": "negative"})

    # Run mock search
    searcher = MockAutoPrompt(vocab_size=1000, emb_dim=64)
    result = searcher.search(
        train_data=train_data,
        dev_data=dev_data,
        num_triggers=3,
        num_candidates=5,
        iters=15,
    )

    # Print algorithm explanation
    print("\n" + "=" * 70)
    print("ALGORITHM EXPLANATION")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    AutoPrompt Algorithm Overview                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input:  Template "[T] [T] [T] {sentence} [P] ."                    │
│          Training data with labels                                    │
│          Pretrained MLM (BERT/RoBERTa)                               │
│                                                                       │
│  1. INITIALIZE                                                        │
│     trigger_tokens = [MASK, MASK, MASK]                              │
│                                                                       │
│  2. FOR each iteration:                                              │
│     ┌──────────────────────────────────────────────────────────┐     │
│     │ a) Forward pass: predict at [P] position                 │     │
│     │    loss = -log P(label_token | input + triggers)          │     │
│     │                                                           │     │
│     │ b) Backward pass: get ∇ embedding at trigger positions   │     │
│     │                                                           │     │
│     │ c) HotFlip: score(token) = embedding(token) · ∇          │     │
│     │    candidates = top-k tokens by score                     │     │
│     │                                                           │     │
│     │ d) Evaluate each candidate on training batch              │     │
│     │    keep best if it improves accuracy                      │     │
│     └──────────────────────────────────────────────────────────┘     │
│                                                                       │
│  3. RETURN best trigger tokens found                                 │
│                                                                       │
│  Key Insight: The gradient tells us which DIRECTION in embedding     │
│  space would decrease loss. We find discrete tokens closest to       │
│  that direction via dot product with the embedding matrix.           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

    return result


def run_full_mode():
    """Run with actual model (requires transformers + model download)."""
    try:
        from RnD_Autoprompt.trigger_search import AutoPromptSearcher, SearchConfig
        from RnD_Autoprompt.data import create_synthetic_dataset
    except ImportError:
        print("Error: Cannot import autoprompt modules. Run from project root.")
        sys.exit(1)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " AutoPrompt Pipeline - Full Mode ".center(68) + "║")
    print("║" + " (Requires BERT model download) ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Create synthetic sentiment data
    train_path, dev_path = create_synthetic_dataset(
        task="sentiment", num_train=200, num_dev=50, output_dir="data"
    )

    # Configure search
    config = SearchConfig(
        model_name="bert-base-uncased",
        template="[T] [T] [T] {sentence} [P] .",
        label_map={"positive": "good", "negative": "bad"},
        label_field="label",
        num_candidates=10,
        accumulation_steps=10,
        iters=30,
        batch_size=32,
        eval_size=256,
        patience=5,
        seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Template: {config.template}")
    print(f"  Label map: {config.label_map}")
    print(f"  Iterations: {config.iters}")
    print(f"  Candidates: {config.num_candidates}")

    # Run search
    searcher = AutoPromptSearcher(config)
    result = searcher.search(train_path, dev_path)

    print(f"\nFinal Results:")
    print(f"  Best triggers: {result['best_tokens']}")
    print(f"  Dev accuracy: {result['best_dev_metric']:.4f}")
    print(f"  Time: {result['elapsed_time']:.1f}s")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AutoPrompt Example")
    parser.add_argument(
        '--mode', type=str, default='mock',
        choices=['mock', 'full'],
        help='Run mode: mock (no downloads) or full (requires BERT)'
    )
    args = parser.parse_args()

    if args.mode == 'mock':
        run_mock_demo()
    else:
        run_full_mode()
