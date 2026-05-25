"""
GEPA Merge Proposer.
Combines strengths of two Pareto-optimal candidates.

The merge operation is GEPA's way of doing crossover in the genetic algorithm
analogy. When two candidates on the Pareto front excel at different subsets
of tasks, the merger tries to combine their strengths into a single candidate
that performs well across both subsets.
"""

from typing import Any, Dict, List, Optional
from .state import GEPAState, Candidate


# Default merge prompt template
DEFAULT_MERGE_TEMPLATE = """You are an expert prompt engineer. You need to merge two high-performing instructions into one that combines the strengths of both.

## Candidate A (excels at subset A of tasks)
{instruction_a}

## Candidate B (excels at subset B of tasks)
{instruction_b}

## Performance Summary
- Candidate A average score: {score_a:.3f}
- Candidate B average score: {score_b:.3f}

## Task
Create a merged instruction that:
1. Preserves the strengths of Candidate A
2. Incorporates the strengths of Candidate B
3. Resolves any contradictions between them
4. Is coherent and not simply a concatenation

## Merged Instruction
"""


class MergeProposer:
    """
    Proposes merged candidates from two Pareto-optimal parents.

    The merge step is triggered after a successful reflective mutation
    finds a new program on the Pareto front. It attempts to combine
    two Pareto-optimal candidates that excel on different task subsets.

    Key insight from the paper: System-aware merge is critical —
    naive concatenation doesn't work. The LLM must understand what
    each candidate does well and synthesize a coherent combination.
    """

    def __init__(
        self,
        lm=None,
        prompt_template: Optional[str] = None,
        max_merge_attempts: int = 5,
        mock_mode: bool = False,
    ):
        """
        Args:
            lm: Language model for generating merged candidates.
            prompt_template: Custom merge prompt template.
            max_merge_attempts: Maximum number of merge operations allowed.
            mock_mode: If True, use mock merging for testing.
        """
        self.lm = lm
        self.prompt_template = prompt_template or DEFAULT_MERGE_TEMPLATE
        self.max_merge_attempts = max_merge_attempts
        self.mock_mode = mock_mode
        self.total_merges_tested = 0
        self.merges_due = 0
        self.last_iter_found_new_program = False

    def should_merge(self, state: GEPAState) -> bool:
        """Determine if a merge should be attempted this iteration."""
        if self.total_merges_tested >= self.max_merge_attempts:
            return False
        if len(state.pareto_front_ids) < 2:
            return False
        if not self.last_iter_found_new_program:
            return False
        return self.merges_due > 0

    def select_parents(self, state: GEPAState) -> tuple:
        """
        Select two Pareto-optimal candidates to merge.

        Strategy: pick two candidates that have the most complementary
        strengths (excel on different subsets of validation instances).
        """
        import random

        pareto_ids = state.pareto_front_ids
        if len(pareto_ids) < 2:
            return None, None

        # Find the pair with maximum complementarity
        best_pair = None
        best_complementarity = -1

        for i in range(len(pareto_ids)):
            for j in range(i + 1, len(pareto_ids)):
                c_i = state.candidates[pareto_ids[i]]
                c_j = state.candidates[pareto_ids[j]]
                comp = self._complementarity(c_i, c_j)
                if comp > best_complementarity:
                    best_complementarity = comp
                    best_pair = (pareto_ids[i], pareto_ids[j])

        if best_pair is None:
            # Fallback: random pair
            pair = random.sample(pareto_ids, 2)
            return pair[0], pair[1]

        return best_pair

    def _complementarity(self, a: Candidate, b: Candidate) -> float:
        """
        Measure how complementary two candidates are.
        Higher = they excel on different instances.
        """
        common_ids = set(a.scores.keys()) & set(b.scores.keys())
        if not common_ids:
            return 0.0

        # Count instances where one is clearly better than the other
        a_better = 0
        b_better = 0
        for vid in common_ids:
            if a.scores[vid] > b.scores[vid] + 0.1:
                a_better += 1
            elif b.scores[vid] > a.scores[vid] + 0.1:
                b_better += 1

        # Complementarity = how evenly distributed are their strengths
        total_diff = a_better + b_better
        if total_diff == 0:
            return 0.0
        balance = min(a_better, b_better) / max(a_better, b_better) if max(a_better, b_better) > 0 else 0
        return balance * total_diff / len(common_ids)

    def propose_merge(
        self,
        state: GEPAState,
        parent_a_idx: int,
        parent_b_idx: int,
    ) -> Optional[Dict[str, str]]:
        """
        Generate a merged candidate from two parents.

        Returns:
            Merged candidate dict, or None if merge fails.
        """
        self.total_merges_tested += 1
        self.merges_due = max(0, self.merges_due - 1)

        parent_a = state.candidates[parent_a_idx]
        parent_b = state.candidates[parent_b_idx]

        merged = {}
        for component in parent_a.text.keys():
            text_a = parent_a.text.get(component, "")
            text_b = parent_b.text.get(component, "")

            if text_a == text_b:
                merged[component] = text_a
                continue

            if self.mock_mode:
                merged[component] = self._mock_merge(
                    text_a, text_b,
                    parent_a.average_score, parent_b.average_score
                )
            else:
                merged[component] = self._lm_merge(
                    text_a, text_b,
                    parent_a.average_score, parent_b.average_score
                )

        return merged

    def _lm_merge(
        self,
        text_a: str,
        text_b: str,
        score_a: float,
        score_b: float,
    ) -> str:
        """Use LLM to merge two instructions."""
        prompt = self.prompt_template.format(
            instruction_a=text_a,
            instruction_b=text_b,
            score_a=score_a,
            score_b=score_b,
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.lm(messages=messages)

        if hasattr(response, "choices"):
            return response.choices[0].message.content.strip()
        return str(response).strip()

    def _mock_merge(
        self,
        text_a: str,
        text_b: str,
        score_a: float,
        score_b: float,
    ) -> str:
        """Mock merge for testing."""
        # Simulate intelligent merging by combining key parts
        # In reality, the LLM would do sophisticated synthesis
        lines_a = text_a.strip().split("\n")
        lines_b = text_b.strip().split("\n")

        # Take the base from the higher-scoring candidate
        if score_a >= score_b:
            base = lines_a[0] if lines_a else text_a
            supplement_lines = lines_b
        else:
            base = lines_b[0] if lines_b else text_b
            supplement_lines = lines_a

        # Add unique content from the other candidate
        merged_parts = [base]
        for line in supplement_lines:
            if line.strip() and line not in merged_parts:
                merged_parts.append(line)

        return "\n".join(merged_parts[:5])  # Limit length
