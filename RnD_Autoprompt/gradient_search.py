"""
Gradient-based utilities for AutoPrompt.
Implements GradientStorage and the HotFlip attack for candidate token selection.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class GradientStorage:
    """
    Stores intermediate gradients of the embedding layer output.

    During backpropagation, PyTorch doesn't retain gradients for intermediate
    tensors by default. This hooks into the embedding module to capture them.

    Usage:
        embeddings = model.bert.embeddings.word_embeddings
        gradient_storage = GradientStorage(embeddings)
        loss.backward()
        grad = gradient_storage.get()  # [batch, seq_len, emb_dim]
    """

    def __init__(self, module: torch.nn.Module):
        self._stored_gradient = None
        module.register_backward_hook(self._hook)

    def _hook(self, module, grad_in, grad_out):
        """Backward hook to capture gradient flowing through embedding layer."""
        self._stored_gradient = grad_out[0]

    def get(self) -> Optional[torch.Tensor]:
        """Returns the stored gradient tensor."""
        return self._stored_gradient

    def reset(self):
        """Clear stored gradient."""
        self._stored_gradient = None


def hotflip_attack(
    averaged_grad: torch.Tensor,
    embedding_matrix: torch.Tensor,
    increase_loss: bool = False,
    num_candidates: int = 1,
    token_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    HotFlip attack: find tokens whose embeddings maximize gradient dot product.

    This is the core of AutoPrompt's gradient-guided discrete search.
    For each trigger position, we compute:
        score(token) = embedding(token) · gradient

    The tokens with highest scores are the best candidates for replacement
    (they move the loss in the steepest descent direction).

    Args:
        averaged_grad: Averaged gradient at trigger position [emb_dim]
        embedding_matrix: Full embedding matrix [vocab_size, emb_dim]
        increase_loss: If True, find tokens that increase loss (for adversarial)
        num_candidates: Number of top candidates to return
        token_filter: Tensor of shape [vocab_size] with large negative values
                      for tokens to exclude (special tokens, proper nouns, etc.)

    Returns:
        top_k_ids: Token IDs of top candidates [num_candidates]
    """
    with torch.no_grad():
        # Compute dot product: how much does each token's embedding
        # align with the gradient direction?
        gradient_dot_embedding = torch.matmul(
            embedding_matrix, averaged_grad
        )

        # Apply filter to exclude unwanted tokens
        if token_filter is not None:
            gradient_dot_embedding += token_filter

        # For standard optimization, we want tokens that DECREASE loss
        # (negative gradient direction). Negate to find via topk.
        if not increase_loss:
            gradient_dot_embedding *= -1

        _, top_k_ids = gradient_dot_embedding.topk(num_candidates)

    return top_k_ids


def get_loss(predict_logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute negative log-probability of target labels.

    For multi-token labels, uses logsumexp over valid token positions.

    Args:
        predict_logits: Logits at prediction position [batch, vocab_size]
        label_ids: Target token IDs [batch, num_label_tokens]
                   (padded with 0 for variable-length labels)

    Returns:
        Negative log-probabilities [batch]
    """
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    # Mask out padding (label_id == 0)
    target_logp = target_logp - 1e32 * label_ids.eq(0).float()
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def get_embeddings(model, config):
    """Extract the word embedding module from a HuggingFace model."""
    base_model = getattr(model, config.model_type)
    return base_model.embeddings.word_embeddings
