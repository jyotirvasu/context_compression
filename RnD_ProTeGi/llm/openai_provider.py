"""
OpenAI Provider Implementation
"""

import time
from typing import Optional, List

from .base import (
    BaseLLMProvider,
    LLMResponse,
    LLMConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMAuthenticationError,
)


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using OpenAI's API."""

    def __init__(self, api_key: str, config: Optional[LLMConfig] = None):
        super().__init__(api_key, config or LLMConfig(model="gpt-3.5-turbo"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    temperature=temp,
                    max_tokens=tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                choice = response.choices[0]
                return LLMResponse(
                    content=choice.message.content,
                    model=self.config.model,
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    metadata={"finish_reason": choice.finish_reason},
                )
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    raise LLMRateLimitError(f"Rate limited: {e}")
                elif "auth" in error_str or "401" in error_str:
                    raise LLMAuthenticationError(f"Auth failed: {e}")
                elif "timeout" in error_str:
                    raise LLMTimeoutError(f"Timeout: {e}")
                else:
                    raise LLMProviderError(f"OpenAI error: {e}")

    def classify(
        self,
        prompt: str,
        text: str,
        valid_labels: List[str],
    ) -> str:
        """Classify text using OpenAI."""
        labels_str = ", ".join(valid_labels)
        full_prompt = (
            f"{prompt}\n\n"
            f"Text: \"{text}\"\n\n"
            f"Valid labels: [{labels_str}]\n"
            f"Respond with ONLY the label, nothing else."
        )
        response = self.complete(full_prompt, temperature=0.0)
        predicted = response.content.strip().lower()

        for label in valid_labels:
            if label in predicted:
                return label
        return predicted
