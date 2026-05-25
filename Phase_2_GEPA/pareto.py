"""
Pareto Front Management for GEPA.
Implements multi-objective Pareto selection for candidate diversity.
"""

from typing import Dict, List, Set
from .state import Candidate, GEPAState


class ParetoFront:
    """
    Manages the Pareto frontier of candidates.

    In GEPA, the Pareto front tracks candidates that excel on different
    subsets of the validation set. This ensures diversity — instead of
    converging to a single "best" prompt, GEPA maintains candidates that
    are complementary (good at different things).

    A candidate is Pareto-optimal if no other candidate dominates it on
    ALL evaluation instances. This allows the merge step to combine
    strengths from different Pareto-optimal candidates.
    """

    def __init__(self, frontier_type: str = "linear"):
        """
        Args:
            frontier_type: "linear" (single-objective, average score) or
                          "pareto" (multi-objective, per-instance scores)
        """
        self.frontier_type = frontier_type

    def update(self, state: GEPAState) -> List[int]:
        """
        Recompute the Pareto front after adding a new candidate.

        Returns:
            Updated list of Pareto-optimal candidate indices.
        """
        if self.frontier_type == "linear":
            state.pareto_front_ids = self._linear_front(state)
        else:
            state.pareto_front_ids = self._pareto_front(state)
        return state.pareto_front_ids

    def _linear_front(self, state: GEPAState) -> List[int]:
        """
        Simple frontier: keep top-K candidates by average score.
        Used when only a single scalar objective exists.
        """
        if not state.candidates:
            return []

        # Sort by average score, keep top candidates
        sorted_candidates = sorted(
            state.candidates, key=lambda c: c.average_score, reverse=True
        )
        # Keep top 5 or all if fewer
        k = min(5, len(sorted_candidates))
        return [c.idx for c in sorted_candidates[:k]]

    def _pareto_front(self, state: GEPAState) -> List[int]:
        """
        Multi-objective Pareto front: a candidate is Pareto-optimal if
        no other candidate dominates it on all validation instances.

        Dominance: candidate A dominates B if A scores >= B on ALL
        instances AND strictly > on at least one.
        """
        if not state.candidates:
            return []

        # Get all validation IDs across all candidates
        all_val_ids: Set[int] = set()
        for c in state.candidates:
            all_val_ids.update(c.scores.keys())

        if not all_val_ids:
            return [0] if state.candidates else []

        pareto_ids = []
        n = len(state.candidates)

        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(state.candidates[j], state.candidates[i], all_val_ids):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_ids.append(i)

        return pareto_ids if pareto_ids else [state.best_candidate_idx]

    @staticmethod
    def _dominates(a: Candidate, b: Candidate, val_ids: Set[int]) -> bool:
        """Check if candidate A dominates candidate B."""
        all_geq = True
        any_greater = False

        for vid in val_ids:
            score_a = a.scores.get(vid, 0.0)
            score_b = b.scores.get(vid, 0.0)

            if score_a < score_b:
                all_geq = False
                break
            if score_a > score_b:
                any_greater = True

        return all_geq and any_greater

    def select_candidate(self, state: GEPAState) -> int:
        """
        Select a candidate from the Pareto front for mutation.

        Strategy: weighted random selection favoring candidates with
        fewer descendants (to encourage exploration of underexplored regions).
        """
        import random

        if not state.pareto_front_ids:
            return 0

        # Count how many times each Pareto candidate has been selected as parent
        parent_counts: Dict[int, int] = {pid: 0 for pid in state.pareto_front_ids}
        for c in state.candidates:
            for pid in c.parent_ids:
                if pid in parent_counts:
                    parent_counts[pid] += 1

        # Inverse weighting: less-explored candidates get higher weight
        weights = []
        for pid in state.pareto_front_ids:
            weight = 1.0 / (1.0 + parent_counts[pid])
            weights.append(weight)

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(state.pareto_front_ids, weights=weights, k=1)[0]
