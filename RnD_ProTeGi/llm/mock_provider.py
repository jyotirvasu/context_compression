"""
Mock LLM Provider for testing without API keys.
Returns simulated responses based on pattern matching.
"""

import random
from typing import Optional, List
from .base import BaseLLMProvider, LLMConfig, LLMResponse


class MockProvider(BaseLLMProvider):
    """
    Mock provider for testing the pipeline without real API calls.
    Simulates classification, gradient generation, and prompt editing.
    """

    def __init__(self, api_key: str = "mock", config: Optional[LLMConfig] = None):
        super().__init__(api_key=api_key, config=config or LLMConfig())

    def complete(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Simulate LLM completion based on prompt content."""
        prompt_lower = prompt.lower()

        # Gradient generation (error analysis)
        if "analyze" in prompt_lower and "error" in prompt_lower:
            return LLMResponse(
                content="The prompt lacks specific category names and doesn't instruct "
                        "the model to choose from predefined labels. It needs explicit "
                        "output format instructions.",
                model="mock",
                tokens_used=50,
            )

        # Prompt editing
        if "rewrite" in prompt_lower or "improv" in prompt_lower:
            variations = [
                "Classify the following customer message into one of these categories: "
                "refund, technical_support, billing, general_inquiry. "
                "Respond with ONLY the category name.",

                "Read the customer's message and determine their intent. "
                "Choose exactly one label from: [refund, technical_support, billing, general_inquiry]. "
                "Output only the label.",

                "You are a customer support classifier. Given a message, output the matching category: "
                "refund, technical_support, billing, or general_inquiry. No explanation needed.",
            ]
            return LLMResponse(
                content=random.choice(variations),
                model="mock",
                tokens_used=40,
            )

        # Default: return a simple response
        return LLMResponse(content="general_inquiry", model="mock", tokens_used=10)

    def classify(self, prompt: str, text: str, valid_labels: List[str] = None) -> str:
        """Simulate classification with basic keyword matching."""
        text_lower = text.lower()

        # Simple keyword-based classification
        if any(w in text_lower for w in ["refund", "money back", "return", "defective"]):
            label = "refund"
        elif any(w in text_lower for w in ["crash", "error", "log in", "bug", "can't"]):
            label = "technical_support"
        elif any(w in text_lower for w in ["charge", "payment", "credit card", "bill"]):
            label = "billing"
        else:
            label = "general_inquiry"

        # Add some noise to simulate imperfect classification
        if random.random() < 0.15:
            labels = valid_labels or ["refund", "technical_support", "billing", "general_inquiry"]
            label = random.choice(labels)

        return label
