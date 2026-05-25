"""
Candidate Tracking
Tracks prompt candidates during beam search optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import statistics


@dataclass
class Candidate:
    """
    A prompt candidate in the beam.

    Tracks a prompt along with its evaluation history and statistics.
    """
    prompt: str
    scores: List[float] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def num_trials(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def std_score(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return statistics.stdev(self.scores)

    @property
    def best_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores)

    def add_score(self, score: float):
        """Add a new evaluation score."""
        self.scores.append(score)

    def __repr__(self) -> str:
        prompt_preview = self.prompt[:40] + "..." if len(self.prompt) > 40 else self.prompt
        return (
            f"Candidate(prompt='{prompt_preview}', "
            f"mean={self.mean_score:.3f}, trials={self.num_trials})"
        )

    def __lt__(self, other: "Candidate") -> bool:
        return self.mean_score < other.mean_score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return False
        return self.prompt == other.prompt

    def __hash__(self) -> int:
        return hash(self.prompt)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "mean_score": self.mean_score,
            "best_score": self.best_score,
            "num_trials": self.num_trials,
            "scores": self.scores,
            "metadata": self.metadata,
        }


@dataclass
class BeamState:
    """Snapshot of beam at a given iteration."""
    iteration: int
    candidates: List[Candidate]
    best_score: float

    def __repr__(self) -> str:
        return f"BeamState(iter={self.iteration}, best={self.best_score:.3f}, size={len(self.candidates)})"
