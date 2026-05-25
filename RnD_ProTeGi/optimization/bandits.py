"""
Bandit Algorithms
UCB (Upper Confidence Bound) and Successive Rejects for smart resource allocation.
"""

from typing import List, Dict
import math
from dataclasses import dataclass

from .candidate import Candidate


@dataclass
class BanditStats:
    """Statistics for bandit algorithms."""
    total_budget: int
    allocations: Dict[str, int]
    pulls: Dict[str, int]


class UCB:
    """
    Upper Confidence Bound algorithm.

    Balances exploitation (high-scoring candidates) with exploration
    (under-evaluated candidates).

    UCB score = mean_score + c * sqrt(log(T) / n)
    where T = total pulls, n = candidate pulls, c = exploration constant
    """

    def __init__(self, exploration_constant: float = math.sqrt(2)):
        self.exploration_constant = exploration_constant

    def compute_ucb(self, candidate: Candidate, total_pulls: int) -> float:
        """Compute UCB score for a candidate."""
        if candidate.num_trials == 0:
            return float('inf')

        exploration_bonus = self.exploration_constant * math.sqrt(
            math.log(total_pulls) / candidate.num_trials
        )
        return candidate.mean_score + exploration_bonus

    def allocate_budget(
        self,
        candidates: List[Candidate],
        total_budget: int,
        min_allocation: int = 1,
    ) -> Dict[str, int]:
        """
        Allocate evaluation budget across candidates proportionally to UCB scores.

        Higher UCB = more budget (more variants generated).
        """
        if not candidates:
            return {}

        # Give minimum allocation to each
        allocations = {c.prompt: min_allocation for c in candidates}
        remaining_budget = total_budget - min_allocation * len(candidates)

        if remaining_budget <= 0:
            return allocations

        # Compute UCB scores
        total_pulls = max(1, sum(c.num_trials for c in candidates))
        ucb_scores = {
            c.prompt: self.compute_ucb(c, total_pulls) for c in candidates
        }

        # Handle all-infinite scores (all unobserved)
        if all(math.isinf(s) for s in ucb_scores.values()):
            per_candidate = remaining_budget // len(candidates)
            for c in candidates:
                allocations[c.prompt] += per_candidate
            return allocations

        # Proportional allocation based on finite UCB scores
        finite_scores = {k: v for k, v in ucb_scores.items() if not math.isinf(v)}
        total_ucb = sum(finite_scores.values())

        if total_ucb > 0:
            for c in candidates:
                score = ucb_scores[c.prompt]
                if math.isinf(score):
                    proportion = 1.0 / len(candidates)
                else:
                    proportion = finite_scores.get(c.prompt, 0) / total_ucb
                additional = int(remaining_budget * proportion)
                allocations[c.prompt] += additional

        return allocations

    def select_next(self, candidates: List[Candidate]) -> Candidate:
        """Select next candidate to evaluate using UCB."""
        total_pulls = max(1, sum(c.num_trials for c in candidates))
        ucb_scores = [self.compute_ucb(c, total_pulls) for c in candidates]
        max_idx = max(range(len(candidates)), key=lambda i: ucb_scores[i])
        return candidates[max_idx]


class SuccessiveRejects:
    """
    Successive Rejects algorithm for pruning.

    Eliminates the worst candidate after each round of evaluation,
    allocating more budget to remaining candidates in later rounds.
    """

    def __init__(self, confidence: float = 0.1):
        self.confidence = confidence

    def prune(
        self,
        candidates: List[Candidate],
        target_size: int,
        evaluator_fn=None,
    ) -> List[Candidate]:
        """
        Prune candidates down to target_size using successive rejects.

        If evaluator_fn is None, uses existing scores.
        """
        if len(candidates) <= target_size:
            return candidates

        remaining = candidates.copy()

        while len(remaining) > target_size:
            # Remove candidate with lowest mean score
            worst = min(remaining, key=lambda c: c.mean_score)
            remaining.remove(worst)

        return remaining


def allocate_budget_proportional(
    candidates: List[Candidate],
    total_budget: int,
) -> Dict[str, int]:
    """Simple proportional budget allocation based on scores."""
    if not candidates:
        return {}

    scores = [max(c.mean_score, 0.01) for c in candidates]
    total_score = sum(scores)

    allocations = {}
    for c, score in zip(candidates, scores):
        allocations[c.prompt] = max(1, int(total_budget * score / total_score))

    return allocations
