"""
Base LLM Provider Interface
Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Request timed out."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Authentication failed."""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement the `complete` method.
    """

    def __init__(self, api_key: str, config: Optional[LLMConfig] = None):
        self.api_key = api_key
        self.config = config or LLMConfig(model="gpt-3.5-turbo")

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    def classify(
        self,
        prompt: str,
        text: str,
        valid_labels: List[str],
    ) -> str:
        """
        Classify text using the prompt.

        Args:
            prompt: Classification instruction prompt
            text: Text to classify
            valid_labels: Valid label options

        Returns:
            Predicted label string
        """
        pass
