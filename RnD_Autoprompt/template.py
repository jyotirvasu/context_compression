"""
Template system for AutoPrompt.
Handles the conversion of templates with [T] and [P] tokens into model inputs.
"""

import torch
from typing import Dict, Optional, Any
from transformers import PreTrainedTokenizer


def add_task_specific_tokens(tokenizer: PreTrainedTokenizer):
    """
    Add special trigger [T], predict [P], and label [Y] tokens to tokenizer.

    These tokens act as placeholders:
    - [T]: Trigger token positions (will be replaced during optimization)
    - [P]: Prediction position (replaced with [MASK] for MLM prediction)
    - [Y]: Used for LAMA-style label formatting
    """
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]', '[Y]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_y_id = tokenizer.convert_tokens_to_ids('[Y]')


def encode_label(tokenizer: PreTrainedTokenizer, label, tokenize: bool = False) -> torch.Tensor:
    """
    Encode a label into token IDs.

    Args:
        tokenizer: HuggingFace tokenizer
        label: String token, list of tokens, or integer ID
        tokenize: If True, tokenize the label string first

    Returns:
        Tensor of shape [1, num_tokens]
    """
    if isinstance(label, str):
        if tokenize:
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" maps to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" maps to UNK.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    else:
        raise ValueError(f"Unsupported label type: {type(label)}")
    return encoded


class TriggerTemplatizer:
    """
    Converts a template string with placeholders into model-ready inputs.

    Template format:
        "[T] [T] [T] {sentence} [P] ."

    Where:
        [T] = Trigger token placeholder (optimized by AutoPrompt)
        [P] = Prediction placeholder (becomes [MASK] for MLM)
        {field} = Data field placeholder (filled from dataset instance)

    Example:
        template = "[T] [T] [T] {sentence} [P] ."
        instance = {"sentence": "This movie is great", "label": "positive"}
        -> Tokenized: [CLS] [T] [T] [T] This movie is great [MASK] . [SEP]
        -> trigger_mask marks [T] positions
        -> predict_mask marks [MASK] position
    """

    def __init__(
        self,
        template: str,
        config,
        tokenizer: PreTrainedTokenizer,
        label_field: str = 'label',
        label_map: Optional[Dict[str, str]] = None,
        tokenize_labels: bool = False,
        add_special_tokens: bool = False,
        use_ctx: bool = False,
    ):
        if not hasattr(tokenizer, 'predict_token') or not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens. '
                'Use `add_task_specific_tokens()` to add them.'
            )
        self._template = template
        self._config = config
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._tokenize_labels = tokenize_labels
        self._add_special_tokens = add_special_tokens
        self._use_ctx = use_ctx

    @property
    def num_trigger_tokens(self) -> int:
        """Count number of [T] placeholders in template."""
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs: Dict[str, Any]):
        """
        Process a data instance through the template.

        Args:
            format_kwargs: Dict with data fields and label

        Returns:
            (model_inputs, label_id) tuple
        """
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop(self._label_field)
        if label is None:
            raise ValueError(f'Missing label in instance')

        # Fill in the template with data fields
        text = self._template.format(**format_kwargs)

        # Tokenize
        model_inputs = self._tokenizer.encode_plus(
            text,
            add_special_tokens=self._add_special_tokens,
            return_tensors='pt'
        )

        input_ids = model_inputs['input_ids']

        # Create masks for trigger and predict positions
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)

        # Replace predict token with actual [MASK] for MLM
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        # Handle token_type_ids for two-sentence inputs (BERT)
        if self._use_ctx and self._config.model_type == 'bert':
            sep_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)
            sep_indices = (input_ids.squeeze(0) == sep_token_id).nonzero().flatten()
            if len(sep_indices) >= 2:
                sequence_b_indices = torch.arange(
                    sep_indices[0], sep_indices[1] + 1
                ).long().unsqueeze(0)
                model_inputs['token_type_ids'].scatter_(1, sequence_b_indices, 1)

        # Encode label
        if self._label_map is not None:
            label = self._label_map[label]
        label_id = encode_label(
            tokenizer=self._tokenizer,
            label=label,
            tokenize=self._tokenize_labels
        )

        return model_inputs, label_id
