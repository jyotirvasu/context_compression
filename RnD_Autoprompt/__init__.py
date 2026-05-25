"""
AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts
=========================================================================================

Implementation based on:
    Shin et al., "AutoPrompt: Eliciting Knowledge from Language Models with
    Automatically Generated Prompts" (EMNLP 2020)
    Paper: https://arxiv.org/abs/2010.15980
    Original code: https://github.com/ucinlp/autoprompt

Core Idea:
    Uses gradient-guided search to find discrete "trigger tokens" that, when
    prepended/appended to inputs, cause masked language models to perform
    classification tasks without any fine-tuning.

Algorithm:
    1. Define a template: "{sentence} [T] [T] [T] [P]"
       where [T] = trigger tokens (shared across all inputs)
             [P] = prediction token (replaced with [MASK])
    2. Initialize triggers as [MASK] tokens
    3. For each iteration:
       a. Forward pass -> compute loss
       b. Backward pass -> get gradient at embedding layer
       c. HotFlip attack: find tokens whose embeddings have highest dot product
          with the gradient (steepest descent direction)
       d. Evaluate candidates, keep best replacement
    4. Return optimized trigger tokens

Usage:
    python -m autoprompt.run --task sentiment --model bert-base-uncased
"""

from .gradient_search import GradientStorage, hotflip_attack
from .template import TriggerTemplatizer, add_task_specific_tokens
from .data import load_trigger_dataset, load_classification_dataset
from .trigger_search import AutoPromptSearcher, SearchConfig
