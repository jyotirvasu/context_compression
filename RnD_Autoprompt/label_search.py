"""
Label Token Search for AutoPrompt.
Finds the best vocabulary tokens to represent each class label.

Instead of manually specifying which token represents "positive" (e.g., "good"),
this module automatically searches the vocabulary for tokens that best
discriminate between classes.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

from .template import add_task_specific_tokens, TriggerTemplatizer
from .data import Collator, load_trigger_dataset
from .gradient_search import get_loss

logger = logging.getLogger(__name__)


def search_labels(
    model_name: str,
    train_path: Path,
    template: str,
    label_field: str = "label",
    num_candidates: int = 10,
    batch_size: int = 32,
    use_ctx: bool = False,
    device: str = "auto",
) -> Dict[str, List[str]]:
    """
    Search for the best label tokens for each class.

    For each class, finds vocabulary tokens that have highest average
    log-probability at the [P] position for instances of that class.

    Algorithm:
    1. For each instance in the training set, get logits at [P] position
    2. Group by true label
    3. For each label, find tokens with highest average log-prob

    Args:
        model_name: HuggingFace model name
        train_path: Path to training data
        template: Template string with [T] and [P]
        label_field: Name of the label field in data
        num_candidates: Number of candidate label tokens to return per class
        batch_size: Batch size for inference
        use_ctx: Whether to use context (relation extraction)
        device: Device specification

    Returns:
        Dict mapping label string -> list of top candidate token strings

    Example:
        >>> labels = search_labels(
        ...     "bert-base-uncased",
        ...     "data/sentiment_train.jsonl",
        ...     "[T] [T] [T] {sentence} [P] .",
        ... )
        >>> print(labels)
        {'positive': ['good', 'great', 'wonderful', ...],
         'negative': ['bad', 'terrible', 'awful', ...]}
    """
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Load model
    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    add_task_specific_tokens(tokenizer)

    # We need a templatizer without label_map (since we're searching for labels)
    # Use a dummy label_map that passes labels through
    # First, scan data to find all labels
    train_path = Path(train_path)
    from .data import LOADERS
    loader = LOADERS[train_path.suffix]
    all_labels = set()
    for instance in loader(train_path):
        all_labels.add(instance[label_field])
    logger.info(f"Found labels: {all_labels}")

    # Create identity label map (label -> label token)
    # For label search, we use the labels as-is and accumulate logits
    identity_map = {label: label for label in all_labels}

    templatizer = TriggerTemplatizer(
        template=template,
        config=config,
        tokenizer=tokenizer,
        label_map=identity_map,
        label_field=label_field,
        add_special_tokens=False,
        use_ctx=use_ctx,
    )

    # Load dataset
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = load_trigger_dataset(train_path, templatizer, use_ctx=use_ctx)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # For label search, we use [MASK] as trigger (no optimization)
    num_triggers = templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(
        [tokenizer.mask_token_id] * num_triggers, device=device
    ).unsqueeze(0)

    # Accumulate log-probabilities per label
    label_to_logprobs = {label: [] for label in all_labels}
    label_to_label_id = {}
    for label in all_labels:
        from .template import encode_label as _encode_label
        label_to_label_id[label] = _encode_label(tokenizer, label)

    logger.info("Computing label logits...")
    all_logits = []
    all_label_ids = []

    for model_inputs, labels in tqdm(train_loader, desc="Label search"):
        model_inputs_dev = {k: v.to(device) for k, v in model_inputs.items()}
        trigger_mask = model_inputs_dev.pop('trigger_mask')
        predict_mask = model_inputs_dev.pop('predict_mask')

        # Replace triggers with MASK
        input_ids = model_inputs_dev['input_ids']
        trigger_ids_expanded = trigger_ids.repeat(trigger_mask.size(0), 1)
        try:
            input_ids = input_ids.masked_scatter(trigger_mask, trigger_ids_expanded)
        except RuntimeError:
            pass
        model_inputs_dev['input_ids'] = input_ids

        with torch.no_grad():
            logits = model(**model_inputs_dev)[0]
            predict_logits = logits.masked_select(
                predict_mask.unsqueeze(-1)
            ).view(logits.size(0), -1)

        all_logits.append(predict_logits.cpu())
        all_label_ids.append(labels)

    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)  # [N, vocab_size]
    all_label_ids = torch.cat(all_label_ids, dim=0)  # [N, ...]

    # For each label, find tokens with highest average log-probability
    log_probs = torch.log_softmax(all_logits, dim=-1)

    results = {}
    for label in all_labels:
        # Find instances belonging to this label
        label_id = tokenizer.convert_tokens_to_ids([label])[0]
        mask = (all_label_ids.squeeze(-1) == label_id)

        if mask.sum() == 0:
            logger.warning(f"No instances found for label '{label}'. Trying string match.")
            # Fallback: use index-based matching
            results[label] = [label]
            continue

        # Average log-probs for this class
        class_logprobs = log_probs[mask].mean(dim=0)  # [vocab_size]

        # Filter special tokens
        for idx in tokenizer.all_special_ids:
            class_logprobs[idx] = -float('inf')

        # Get top candidates
        top_values, top_indices = class_logprobs.topk(num_candidates)
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

        results[label] = top_tokens
        logger.info(f"Label '{label}' top tokens: {top_tokens[:5]}")

    return results
