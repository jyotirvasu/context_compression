"""
Optimization Module
Implements ProTeGi optimization algorithms:
- Textual gradients (error analysis and prompt improvement)
- Beam search (multi-path exploration)
- Bandit algorithms (smart resource allocation)
"""

from .gradient_generator import GradientGenerator, GradientResult
from .prompt_editor import PromptEditor, EditResult
from .candidate import Candidate, BeamState
from .bandits import UCB, SuccessiveRejects, BanditStats, allocate_budget_proportional
from .bandit_beam_search import BanditBeamSearch, BanditBeamConfig, BeamSearchConfig

__all__ = [
    "GradientGenerator",
    "GradientResult",
    "PromptEditor",
    "EditResult",
    "Candidate",
    "BeamState",
    "UCB",
    "SuccessiveRejects",
    "BanditStats",
    "allocate_budget_proportional",
    "BanditBeamSearch",
    "BanditBeamConfig",
    "BeamSearchConfig",
]
