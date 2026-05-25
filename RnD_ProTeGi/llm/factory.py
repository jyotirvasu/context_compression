"""
LLM Provider Factory
"""

from typing import Optional
from .base import BaseLLMProvider, LLMConfig
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .mock_provider import MockProvider


class ProviderFactory:
    """Factory for creating LLM providers."""

    PROVIDERS = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "claude": AnthropicProvider,
        "gpt": OpenAIProvider,
        "mock": MockProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str,
        config: Optional[LLMConfig] = None,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_name: "anthropic", "openai", "claude", or "gpt"
            api_key: API key for the provider
            config: Optional LLMConfig

        Returns:
            BaseLLMProvider instance
        """
        provider_name = provider_name.lower().strip()
        if provider_name not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {available}"
            )
        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(api_key=api_key, config=config)
