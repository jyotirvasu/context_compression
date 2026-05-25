"""
GEPA: Genetic-Pareto Reflective Prompt Evolution
=================================================

Implementation based on:
    Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
    Reinforcement Learning" (ICLR 2026 Oral)
    Paper: https://arxiv.org/abs/2507.19457
    Original code: https://github.com/gepa-ai/gepa

Core Idea:
    Unlike RL methods that collapse execution traces into scalar rewards,
    GEPA uses LLMs to read full execution traces (errors, reasoning logs,
    tool outputs) to diagnose WHY a candidate failed and propose targeted fixes.
    Through iterative reflection, mutation, and Pareto-aware selection,
    GEPA evolves high-performing prompts with minimal evaluations.

Algorithm (5 steps per iteration):
    1. SELECT a candidate from the Pareto frontier
    2. EXECUTE on a minibatch, capturing full execution traces
    3. REFLECT — an LLM reads traces and diagnoses failures
    4. MUTATE — generate improved candidate informed by reflection
    5. ACCEPT — add to pool if improved, update Pareto front

Key Concepts:
    - Actionable Side Information (ASI): diagnostic feedback from evaluators
      that serves as the text-optimization analogue of a gradient
    - Pareto frontier: candidates excelling on different task subsets
    - Merge: combining strengths of two Pareto-optimal candidates

Usage:
    from gepa import GEPAOptimizer, GEPAConfig
    optimizer = GEPAOptimizer(config)
    result = optimizer.optimize(seed_prompt, train_data, val_data)
"""

from .engine import GEPAEngine, GEPAConfig
from .state import GEPAState, Candidate
from .reflector import Reflector
from .pareto import ParetoFront
from .adapter import BaseAdapter, DefaultAdapter
from .merge import MergeProposer
