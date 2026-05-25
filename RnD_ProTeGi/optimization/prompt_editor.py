"""
Prompt Editor
Edits prompts based on textual gradients to generate improved variants.
Uses LLM to rewrite prompts, applying diversity through temperature variation.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import concurrent.futures
from threading import Lock

from ..llm import BaseLLMProvider


@dataclass
class EditResult:
    """Result of prompt editing."""
    original_prompt: str
    gradient: str
    edited_prompts: List[str]
    temperatures_used: List[float]
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f"EditResult(original='{self.original_prompt[:30]}...', "
            f"variants={len(self.edited_prompts)})"
        )


class PromptEditor:
    """
    Edits prompts based on textual gradients.

    Takes a prompt and a gradient (error analysis) and produces
    improved prompt variants using temperature-based diversity.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        base_temperature: float = 0.7,
        temperature_range: float = 0.3,
        max_workers: int = 4,
    ):
        self.provider = provider
        self.base_temperature = base_temperature
        self.temperature_range = temperature_range
        self.max_workers = max_workers
        self._lock = Lock()
        self._stats = {"total_edits": 0, "total_variants": 0}

    def edit(
        self,
        prompt: str,
        gradient: str,
        num_variants: int = 3,
        preserve_format: bool = True,
    ) -> EditResult:
        """
        Edit a prompt based on a gradient to produce improved variants.

        Args:
            prompt: Original prompt to improve
            gradient: Textual gradient (error analysis)
            num_variants: Number of variants to generate
            preserve_format: Whether to maintain prompt structure

        Returns:
            EditResult with list of edited prompt variants
        """
        temperatures = self._generate_temperatures(num_variants)

        if self.max_workers <= 1:
            edited_prompts = [
                self._edit_single(prompt, gradient, temp, preserve_format)
                for temp in temperatures
            ]
        else:
            edited_prompts = self._edit_parallel(
                prompt, gradient, temperatures, preserve_format
            )

        with self._lock:
            self._stats["total_edits"] += 1
            self._stats["total_variants"] += len(edited_prompts)

        return EditResult(
            original_prompt=prompt,
            gradient=gradient,
            edited_prompts=edited_prompts,
            temperatures_used=temperatures,
        )

    def _edit_single(
        self,
        prompt: str,
        gradient: str,
        temperature: float,
        preserve_format: bool,
    ) -> str:
        """Edit a single prompt with given temperature."""
        edit_prompt = self._build_edit_prompt(prompt, gradient, preserve_format)
        response = self.provider.complete(edit_prompt, temperature=temperature)
        edited = response.content.strip()
        return self._clean_edited_prompt(edited)

    def _edit_parallel(
        self,
        prompt: str,
        gradient: str,
        temperatures: List[float],
        preserve_format: bool,
    ) -> List[str]:
        """Edit prompt in parallel with different temperatures."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._edit_single, prompt, gradient, temp, preserve_format)
                for temp in temperatures
            ]
            edited_prompts = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    edited_prompts.append(future.result())
                except Exception as e:
                    print(f"Error in parallel edit: {e}")
                    edited_prompts.append(prompt)
        return edited_prompts

    def _generate_temperatures(self, num_variants: int) -> List[float]:
        """Generate evenly spaced temperatures for diversity."""
        if num_variants == 1:
            return [self.base_temperature]

        temps = []
        for i in range(num_variants):
            offset = (i / (num_variants - 1) - 0.5) * 2 * self.temperature_range
            temp = max(0.0, min(1.5, self.base_temperature + offset))
            temps.append(temp)
        return temps

    def _build_edit_prompt(
        self, prompt: str, gradient: str, preserve_format: bool
    ) -> str:
        """Build the meta-prompt for editing."""
        format_instruction = ""
        if preserve_format:
            format_instruction = "\n4. Maintain a similar length/style as the original."

        return f"""You are improving a classification prompt based on feedback.

Current Prompt: "{prompt}"

Issue Identified: {gradient}

Task: Rewrite the prompt to address this issue. The improved prompt should:
1. Fix the identified problem
2. Be clear and specific
3. Guide the model to recognize the missing pattern{format_instruction}

Provide ONLY the improved prompt, nothing else.

Improved Prompt:"""

    def _clean_edited_prompt(self, edited: str) -> str:
        """Clean up LLM-generated prompt."""
        # Remove surrounding quotes
        edited = edited.strip('"\'')

        # Remove common prefixes
        prefixes = [
            "Improved Prompt:",
            "Improved prompt:",
            "New prompt:",
            "Here is the improved prompt:",
            "Here's the improved prompt:",
        ]
        for prefix in prefixes:
            if edited.startswith(prefix):
                edited = edited[len(prefix):].strip()

        # Remove quotes again after prefix removal
        edited = edited.strip('"\'')
        return edited.strip()

    def get_statistics(self) -> Dict[str, int]:
        with self._lock:
            return self._stats.copy()
