"""
ProTeGi: Prompt Optimization with Textual Gradients
Based on "Automatic Prompt Optimization with Gradient Descent and Beam Search"
(Pryzant et al., EMNLP 2023)
"""

from .llm import ProviderFactory, LLMConfig, BaseLLMProvider, LLMResponse
from .evaluation import (
    ClassificationDataset,
    DatasetItem,
    PromptEvaluator,
    create_spam_dataset,
)
from .optimization import (
    GradientGenerator,
    PromptEditor,
    BanditBeamSearch,
    BanditBeamConfig,
)
