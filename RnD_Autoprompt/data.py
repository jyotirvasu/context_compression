"""
Data loading utilities for AutoPrompt.
Supports TSV and JSONL formats for classification and relation extraction tasks.
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from .template import TriggerTemplatizer, encode_label

logger = logging.getLogger(__name__)

MAX_CONTEXT_LEN = 50


def pad_squeeze_sequence(sequence, batch_first=True, padding_value=0):
    """Squeeze fake batch dimension added by tokenizer before padding."""
    return pad_sequence(
        [x.squeeze(0) for x in sequence],
        batch_first=batch_first,
        padding_value=padding_value,
    )


class Collator:
    """
    Collates transformer outputs into batched tensors with proper padding.
    Used as collate_fn for DataLoader.
    """

    def __init__(self, pad_token_id: int = 0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        model_inputs, labels = list(zip(*features))

        # Determine keys from first input
        keys = list(model_inputs[0].keys())
        padded_inputs = {}

        for key in keys:
            padding_value = self._pad_token_id if key == 'input_ids' else 0
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(
                sequence, batch_first=True, padding_value=padding_value
            )
            padded_inputs[key] = padded

        labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=0)
        return padded_inputs, labels


def load_tsv(fname):
    """Load a TSV file as an iterator of dicts."""
    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def load_jsonl(fname):
    """Load a JSONL file as an iterator of dicts."""
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


LOADERS = {
    '.tsv': load_tsv,
    '.jsonl': load_jsonl,
}


def load_trigger_dataset(
    fname: Path,
    templatizer: TriggerTemplatizer,
    use_ctx: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple]:
    """
    Load dataset and process through templatizer for trigger search.

    Each instance is converted to (model_inputs, label_id) pairs using
    the templatizer's template.

    Args:
        fname: Path to data file (.tsv or .jsonl)
        templatizer: Template processor
        use_ctx: Whether to use context sentences (for relation extraction)
        limit: Maximum number of instances to return

    Returns:
        List of (model_inputs_dict, label_tensor) tuples
    """
    fname = Path(fname)
    loader = LOADERS[fname.suffix]
    instances = []

    for x in loader(fname):
        try:
            if use_ctx:
                # Handle relation extraction with context sentences
                if 'evidences' not in x:
                    logger.warning(f'Skipping sample without context: {x}')
                    continue

                evidences = x['evidences']
                obj_surface, masked_sent = random.choice([
                    (ev['obj_surface'], ev['masked_sentence'])
                    for ev in evidences
                ])
                words = masked_sent.split()
                if len(words) > MAX_CONTEXT_LEN:
                    masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])

                context = masked_sent.replace('[MASK]', obj_surface)
                x['context'] = context

            model_inputs, label_id = templatizer(x)
        except ValueError as e:
            logger.warning(f'Error "{e}" processing "{x}". Skipping.')
            continue
        else:
            instances.append((model_inputs, label_id))

    if limit and len(instances) > limit:
        return random.sample(instances, limit)
    return instances


def load_classification_dataset(
    fname: Path,
    tokenizer,
    input_field_a: str,
    input_field_b: Optional[str] = None,
    label_field: str = 'label',
    label_map: Optional[Dict] = None,
    limit: Optional[int] = None,
):
    """
    Load a classification dataset (for label token search).

    Args:
        fname: Path to data file
        tokenizer: HuggingFace tokenizer
        input_field_a: Primary text field name
        input_field_b: Optional second text field (for NLI)
        label_field: Name of the label field
        label_map: Optional pre-defined label-to-index mapping
        limit: Max instances

    Returns:
        (instances, label_map) tuple
    """
    fname = Path(fname)
    instances = []
    label_map = label_map or {}
    loader = LOADERS[fname.suffix]

    for instance in loader(fname):
        model_inputs = tokenizer.encode_plus(
            instance[input_field_a],
            instance[input_field_b] if input_field_b else None,
            add_special_tokens=True,
            return_tensors='pt',
        )
        label = instance[label_field]
        if label not in label_map:
            label_map[label] = len(label_map)
        label_id = torch.tensor([[label_map[label]]])
        instances.append((model_inputs, label_id))

    if limit and len(instances) > limit:
        instances = random.sample(instances, limit)

    return instances, label_map


def create_synthetic_dataset(
    task: str = "sentiment",
    num_train: int = 100,
    num_dev: int = 50,
    output_dir: str = "data",
) -> Tuple[Path, Path]:
    """
    Create a synthetic dataset for testing the pipeline.

    Args:
        task: Task type ("sentiment" or "nli")
        num_train: Number of training examples
        num_dev: Number of dev examples
        output_dir: Directory to write data files

    Returns:
        (train_path, dev_path) tuple
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if task == "sentiment":
        positive_phrases = [
            "This is wonderful", "I love this", "Absolutely fantastic",
            "Great experience", "Highly recommended", "Best ever",
            "So good", "Amazing quality", "Truly excellent", "Outstanding work",
            "Brilliant performance", "Wonderful time", "Loved every moment",
            "Superb craftsmanship", "Exceptional service", "Delightful",
            "Remarkable achievement", "Thoroughly enjoyed", "Perfect in every way",
            "Incredibly satisfying",
        ]
        negative_phrases = [
            "This is terrible", "I hate this", "Absolutely awful",
            "Horrible experience", "Never again", "Worst ever",
            "So bad", "Poor quality", "Truly dreadful", "Disappointing work",
            "Terrible performance", "Wasted my time", "Hated every moment",
            "Shoddy craftsmanship", "Awful service", "Disgusting",
            "Complete failure", "Thoroughly unpleasant", "Worst in every way",
            "Incredibly frustrating",
        ]

        def generate_instances(n):
            instances = []
            for i in range(n):
                if i % 2 == 0:
                    sent = random.choice(positive_phrases)
                    label = "positive"
                else:
                    sent = random.choice(negative_phrases)
                    label = "negative"
                instances.append({"sentence": sent, "label": label})
            return instances

    elif task == "nli":
        entailment_pairs = [
            ("A dog runs in the park", "An animal is outside"),
            ("She is singing a song", "She is making music"),
            ("The cat sleeps on the bed", "The feline is resting"),
            ("He drives to work", "He commutes by car"),
            ("Children play in the garden", "Kids are outdoors"),
        ]
        contradiction_pairs = [
            ("A dog runs in the park", "No animals are outside"),
            ("She is singing a song", "She is completely silent"),
            ("The cat sleeps on the bed", "The cat is running"),
            ("He drives to work", "He walks everywhere"),
            ("Children play in the garden", "The garden is empty"),
        ]

        def generate_instances(n):
            instances = []
            for i in range(n):
                if i % 2 == 0:
                    premise, hypothesis = random.choice(entailment_pairs)
                    label = "entailment"
                else:
                    premise, hypothesis = random.choice(contradiction_pairs)
                    label = "contradiction"
                instances.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                })
            return instances
    else:
        raise ValueError(f"Unknown task: {task}")

    train_data = generate_instances(num_train)
    dev_data = generate_instances(num_dev)

    train_path = output_dir / f"{task}_train.jsonl"
    dev_path = output_dir / f"{task}_dev.jsonl"

    for path, data in [(train_path, train_data), (dev_path, dev_data)]:
        with open(path, 'w', encoding='utf-8') as f:
            for instance in data:
                f.write(json.dumps(instance) + '\n')

    logger.info(f"Created {task} dataset: {len(train_data)} train, {len(dev_data)} dev")
    return train_path, dev_path
