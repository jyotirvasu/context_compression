"""
GEPA State Management.
Tracks candidates, scores, Pareto front, and optimization history.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Candidate:
    """A single candidate in the optimization pool."""
    idx: int
    text: Dict[str, str]  # component_name -> text content
    parent_ids: List[int]
    scores: Dict[int, float] = field(default_factory=dict)  # val_id -> score
    average_score: float = 0.0
    iteration_created: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def components(self) -> List[str]:
        return list(self.text.keys())


@dataclass
class EvaluationResult:
    """Result from evaluating a candidate on a batch."""
    scores: List[float]
    outputs: List[Any]
    trajectories: List[str]  # execution traces / reasoning logs
    objective_scores: Optional[List[Dict[str, float]]] = None


@dataclass
class GEPAState:
    """
    Full state of the GEPA optimization process.

    Maintains:
    - All candidates discovered so far
    - Pareto front tracking
    - Evaluation cache
    - Optimization history
    """
    candidates: List[Candidate] = field(default_factory=list)
    pareto_front_ids: List[int] = field(default_factory=list)
    iteration: int = 0
    total_evals: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_candidate_idx: int = 0
    best_score: float = 0.0

    # Evaluation cache: (candidate_hash, data_id) -> score
    _eval_cache: Dict[Tuple[str, int], float] = field(default_factory=dict)

    def add_candidate(
        self,
        text: Dict[str, str],
        parent_ids: List[int],
        scores: Dict[int, float],
        iteration: int,
    ) -> int:
        """Add a new candidate and return its index."""
        idx = len(self.candidates)
        avg_score = sum(scores.values()) / max(len(scores), 1)
        candidate = Candidate(
            idx=idx,
            text=text,
            parent_ids=parent_ids,
            scores=scores,
            average_score=avg_score,
            iteration_created=iteration,
        )
        self.candidates.append(candidate)

        # Update best
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_candidate_idx = idx

        return idx

    def get_candidate(self, idx: int) -> Candidate:
        return self.candidates[idx]

    def get_best_candidate(self) -> Candidate:
        return self.candidates[self.best_candidate_idx]

    def get_pareto_candidates(self) -> List[Candidate]:
        return [self.candidates[i] for i in self.pareto_front_ids]

    def cache_key(self, candidate: Dict[str, str], data_id: int) -> Tuple[str, int]:
        """Create a hashable cache key for evaluation results."""
        text_hash = str(sorted(candidate.items()))
        return (text_hash, data_id)

    def get_cached_score(self, candidate: Dict[str, str], data_id: int) -> Optional[float]:
        key = self.cache_key(candidate, data_id)
        return self._eval_cache.get(key)

    def cache_score(self, candidate: Dict[str, str], data_id: int, score: float):
        key = self.cache_key(candidate, data_id)
        self._eval_cache[key] = score
