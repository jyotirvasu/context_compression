"""
AutoPrompt Pipeline Runner.
Main entry point for running the full AutoPrompt pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .trigger_search import AutoPromptSearcher, SearchConfig
from .data import create_synthetic_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_sentiment(args):
    """Run sentiment analysis task with AutoPrompt."""
    config = SearchConfig(
        model_name=args.model_name,
        template="[T] [T] [T] {sentence} [P] .",
        label_map={"positive": "good", "negative": "bad"},
        label_field="label",
        num_candidates=args.num_cand,
        accumulation_steps=args.accumulation_steps,
        iters=args.iters,
        batch_size=args.bsz,
        eval_size=args.eval_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    if args.train and args.dev:
        train_path = Path(args.train)
        dev_path = Path(args.dev)
    else:
        logger.info("No data paths provided. Creating synthetic dataset...")
        train_path, dev_path = create_synthetic_dataset(
            task="sentiment", num_train=200, num_dev=50, output_dir="data"
        )

    searcher = AutoPromptSearcher(config)
    result = searcher.search(train_path, dev_path)
    return result


def run_nli(args):
    """Run Natural Language Inference task with AutoPrompt."""
    config = SearchConfig(
        model_name=args.model_name,
        template="{premise} [T] [T] [T] [P] {hypothesis}",
        label_map={"entailment": "yes", "contradiction": "no"},
        label_field="label",
        num_candidates=args.num_cand,
        accumulation_steps=args.accumulation_steps,
        iters=args.iters,
        batch_size=args.bsz,
        eval_size=args.eval_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    if args.train and args.dev:
        train_path = Path(args.train)
        dev_path = Path(args.dev)
    else:
        logger.info("No data paths provided. Creating synthetic dataset...")
        train_path, dev_path = create_synthetic_dataset(
            task="nli", num_train=200, num_dev=50, output_dir="data"
        )

    searcher = AutoPromptSearcher(config)
    result = searcher.search(train_path, dev_path)
    return result


def run_fact_retrieval(args):
    """Run fact retrieval (LAMA-style) task with AutoPrompt."""
    if not args.train or not args.dev:
        logger.error("Fact retrieval requires --train and --dev data paths")
        sys.exit(1)

    config = SearchConfig(
        model_name=args.model_name,
        template="{sub_label} [T] [T] [T] [T] [T] [P] .",
        label_map=None,  # Open vocabulary for fact retrieval
        label_field="obj_label",
        tokenize_labels=True,
        num_candidates=args.num_cand,
        accumulation_steps=args.accumulation_steps,
        iters=args.iters,
        batch_size=args.bsz,
        eval_size=args.eval_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        use_ctx=args.use_ctx,
    )

    searcher = AutoPromptSearcher(config)
    result = searcher.search(Path(args.train), Path(args.dev))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AutoPrompt: Gradient-guided trigger token search for MLMs"
    )

    # Task selection
    parser.add_argument(
        '--task', type=str, default='sentiment',
        choices=['sentiment', 'nli', 'fact_retrieval'],
        help='Task to run'
    )

    # Data paths
    parser.add_argument('--train', type=str, default=None, help='Training data path')
    parser.add_argument('--dev', type=str, default=None, help='Dev data path')

    # Model
    parser.add_argument(
        '--model-name', type=str, default='bert-base-uncased',
        help='HuggingFace model name'
    )

    # Search hyperparameters
    parser.add_argument('--num-cand', type=int, default=10, help='Candidates per position')
    parser.add_argument('--accumulation-steps', type=int, default=10, help='Gradient accumulation steps')
    parser.add_argument('--iters', type=int, default=50, help='Search iterations')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval batch size')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--use-ctx', action='store_true', help='Use context (relation extraction)')

    args = parser.parse_args()

    task_runners = {
        'sentiment': run_sentiment,
        'nli': run_nli,
        'fact_retrieval': run_fact_retrieval,
    }

    logger.info(f"Running AutoPrompt for task: {args.task}")
    result = task_runners[args.task](args)

    # Print results
    print("\n" + "=" * 60)
    print("AUTOPROMPT RESULTS")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Best trigger tokens: {result['best_tokens']}")
    print(f"Best dev metric: {result['best_dev_metric']:.4f}")
    print(f"Search time: {result['elapsed_time']:.1f}s")
    print(f"Iterations: {len(result['history'])}")
    print("=" * 60)


if __name__ == '__main__':
    main()
