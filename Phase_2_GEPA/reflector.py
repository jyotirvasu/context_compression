"""
GEPA Reflector: LLM-based trace analysis and failure diagnosis.

The Reflector is the key differentiator of GEPA from RL approaches.
Instead of collapsing execution traces into scalar rewards, the Reflector
reads full traces (errors, reasoning, tool outputs) and produces
natural-language diagnoses of what went wrong and how to fix it.

This produces "Actionable Side Information" (ASI) — the text-optimization
analogue of a gradient signal.
"""

from typing import Any, Dict, List, Optional


# Default reflection prompt template
DEFAULT_REFLECTION_TEMPLATE = """You are an expert prompt engineer analyzing execution traces to improve a prompt.

## Current Instruction
{current_instruction}

## Execution Traces (with scores)
{traces}

## Task
Analyze these execution traces carefully. For instances with low scores:
1. Identify specific failure patterns (what went wrong)
2. Diagnose root causes (why the current instruction fails)
3. Propose a concrete, improved instruction that addresses these failures

Your improved instruction should:
- Fix identified failure modes
- Preserve behaviors that work well (high-scoring instances)
- Be clear, specific, and actionable
- Not be overly verbose (conciseness matters)

## Improved Instruction
"""


class Reflector:
    """
    Reads execution traces and proposes improved instructions.

    The reflection process:
    1. Receives execution traces from evaluating a candidate
    2. Identifies failure patterns in low-scoring instances
    3. Diagnoses root causes by reading the full trace
    4. Proposes a targeted fix (new instruction text)

    This is what makes GEPA fundamentally different from RL:
    - RL: collapse traces to scalar reward → policy gradient update
    - GEPA: LLM reads traces → natural language diagnosis → targeted mutation
    """

    def __init__(
        self,
        lm=None,
        prompt_template: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """
        Args:
            lm: Language model callable (e.g., litellm.completion).
                Should accept messages and return a response.
            prompt_template: Custom reflection prompt template.
                Must contain {current_instruction} and {traces} placeholders.
            mock_mode: If True, use mock reflection for testing.
        """
        self.lm = lm
        self.prompt_template = prompt_template or DEFAULT_REFLECTION_TEMPLATE
        self.mock_mode = mock_mode
        self._reflection_count = 0

    def reflect_and_propose(
        self,
        component_name: str,
        current_instruction: str,
        reflective_dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze traces and propose an improved instruction.

        Args:
            component_name: Name of the component being optimized.
            current_instruction: Current text of the instruction.
            reflective_dataset: List of trace dicts from adapter.make_reflective_dataset().

        Returns:
            Dict with keys:
              - "new_instruction": proposed improved text
              - "diagnosis": natural language explanation of what's wrong
              - "prompt_used": the reflection prompt sent to the LLM
              - "raw_output": raw LLM output
        """
        self._reflection_count += 1

        # Format traces for the reflection prompt
        traces_text = self._format_traces(reflective_dataset)

        # Build reflection prompt
        prompt = self.prompt_template.format(
            current_instruction=current_instruction,
            traces=traces_text,
        )

        if self.mock_mode:
            return self._mock_reflect(component_name, current_instruction, reflective_dataset)

        # Call LLM for reflection
        response = self._call_lm(prompt)
        new_instruction = self._parse_response(response)

        return {
            "new_instruction": new_instruction,
            "diagnosis": self._extract_diagnosis(response),
            "prompt_used": prompt,
            "raw_output": response,
        }

    def _format_traces(self, dataset: List[Dict[str, Any]]) -> str:
        """Format execution traces for the reflection prompt."""
        lines = []
        for i, entry in enumerate(dataset):
            score = entry.get("score", 0.0)
            trace = entry.get("trace", "")
            needs_fix = entry.get("needs_improvement", score < 0.7)

            lines.append(f"--- Instance {i+1} (Score: {score:.3f}) {'[NEEDS FIX]' if needs_fix else '[OK]'} ---")
            lines.append(trace.strip())
            lines.append("")

        return "\n".join(lines)

    def _call_lm(self, prompt: str) -> str:
        """Call the language model with the reflection prompt."""
        if self.lm is None:
            raise ValueError("Language model (lm) must be provided when not in mock mode.")

        # Support both litellm-style and raw callable
        if callable(self.lm):
            messages = [{"role": "user", "content": prompt}]
            response = self.lm(messages=messages)
            # Handle litellm response format
            if hasattr(response, "choices"):
                return response.choices[0].message.content
            return str(response)
        raise TypeError(f"Unsupported LM type: {type(self.lm)}")

    def _parse_response(self, response: str) -> str:
        """Extract the improved instruction from LLM response."""
        # The response should be the improved instruction directly
        # Strip any markdown formatting
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return text

    def _extract_diagnosis(self, response: str) -> str:
        """Extract diagnosis section from the response (if structured)."""
        # Simple heuristic: everything before the last paragraph is diagnosis
        paragraphs = response.strip().split("\n\n")
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:-1])
        return ""

    def _mock_reflect(
        self,
        component_name: str,
        current_instruction: str,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Mock reflection for testing without LLM API calls."""
        # Analyze failure patterns from scores
        scores = [d.get("score", 0.0) for d in dataset]
        avg_score = sum(scores) / max(len(scores), 1)
        low_score_count = sum(1 for s in scores if s < 0.5)
        total = len(scores)

        # Simulate progressive improvement
        iteration = self._reflection_count

        # Mock diagnosis
        if low_score_count > total * 0.5:
            diagnosis = (
                f"Major issues detected: {low_score_count}/{total} instances scored below 0.5. "
                f"The current instruction lacks specificity and fails to guide the model "
                f"on edge cases. Need to add explicit constraints and examples."
            )
            improvement = "Add explicit constraints, step-by-step structure, and edge case handling"
        elif low_score_count > 0:
            diagnosis = (
                f"Partial failures: {low_score_count}/{total} instances need improvement. "
                f"The instruction handles common cases well but misses nuanced scenarios. "
                f"Need to add clarification for ambiguous inputs."
            )
            improvement = "Add clarification for ambiguous cases and output format specification"
        else:
            diagnosis = (
                f"All {total} instances scoring well (avg: {avg_score:.3f}). "
                f"Minor refinements possible through conciseness and precision."
            )
            improvement = "Refine wording for precision and conciseness"

        # Generate improved instruction (simulates iterative improvement)
        base_improvements = [
            "Be specific and provide step-by-step reasoning.",
            "Consider edge cases and ambiguous inputs carefully.",
            "Format your response clearly with explicit structure.",
            "Verify your answer against the constraints before responding.",
            "Use the exact output format specified.",
        ]

        idx = (iteration - 1) % len(base_improvements)
        new_instruction = (
            f"{current_instruction}\n\n"
            f"[Iteration {iteration} improvement]: {base_improvements[idx]} "
            f"{improvement}."
        )

        return {
            "new_instruction": new_instruction.strip(),
            "diagnosis": diagnosis,
            "prompt_used": f"[Mock reflection prompt for {component_name}]",
            "raw_output": f"[Mock] Diagnosis: {diagnosis}\nImproved: {new_instruction}",
        }
