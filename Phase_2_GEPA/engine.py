"""
GEPA Engine: Main optimization orchestrator.

Implements the core optimization loop:
    1. Select candidate from Pareto frontier
    2. Execute on minibatch with trace capture
    3. Reflect on traces to diagnose failures
    4. Mutate candidate based on reflection
    5. Accept if improved, update Pareto front
    + Periodic merge of complementary Pareto-optimal candidates
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapter import BaseAdapter, DefaultAdapter, AdapterEvalResult
from .merge import MergeProposer
from .pareto import ParetoFront
from .reflector import Reflector
from .state import GEPAState


@dataclass
class GEPAConfig:
    """Configuration for the GEPA optimization engine."""

    # Budget
    max_iterations: int = 50
    max_metric_calls: int = 150

    # Minibatch
    minibatch_size: int = 5

    # Pareto
    frontier_type: str = "pareto"  # "pareto" or "linear"

    # Merge
    use_merge: bool = True
    max_merge_attempts: int = 5
    merge_after_improvement: bool = True

    # Scores
    perfect_score: Optional[float] = 1.0
    skip_perfect_score: bool = True

    # Random seed
    seed: int = 42

    # Mock mode (no LLM calls)
    mock_mode: bool = False

    # Logging
    verbose: bool = True

    # Component selection
    components_per_iteration: int = 1  # How many components to mutate per iteration


@dataclass
class OptimizationResult:
    """Result of a GEPA optimization run."""
    best_candidate: Dict[str, str]
    best_score: float
    best_candidate_idx: int
    total_iterations: int
    total_metric_calls: int
    pareto_front: List[Dict[str, str]]
    history: List[Dict[str, Any]]
    state: GEPAState


class GEPAEngine:
    """
    Main optimization engine implementing the GEPA algorithm.

    Usage:
        config = GEPAConfig(max_iterations=50, mock_mode=True)
        engine = GEPAEngine(config, adapter, reflector)
        result = engine.optimize(seed_candidate, train_data, val_data)
    """

    def __init__(
        self,
        config: GEPAConfig,
        adapter: Optional[BaseAdapter] = None,
        reflector: Optional[Reflector] = None,
        merge_proposer: Optional[MergeProposer] = None,
    ):
        self.config = config
        random.seed(config.seed)

        # Create defaults if not provided
        self.adapter = adapter or DefaultAdapter(mock_mode=config.mock_mode)
        self.reflector = reflector or Reflector(mock_mode=config.mock_mode)
        self.merge_proposer = merge_proposer or (
            MergeProposer(mock_mode=config.mock_mode, max_merge_attempts=config.max_merge_attempts)
            if config.use_merge else None
        )
        self.pareto = ParetoFront(frontier_type=config.frontier_type)
        self.state = GEPAState()

    def optimize(
        self,
        seed_candidate: Dict[str, str],
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
    ) -> OptimizationResult:
        """
        Run the GEPA optimization loop.

        Args:
            seed_candidate: Initial prompt candidate.
                Dict mapping component names to text, e.g.:
                {"system_prompt": "You are a helpful assistant."}
            train_data: Training data for minibatch evaluation.
                List of dicts with at least "input" and optionally "expected".
            val_data: Validation data for full evaluation.
                Same format as train_data.

        Returns:
            OptimizationResult with the best candidate and optimization history.
        """
        self._log(f"Starting GEPA optimization")
        self._log(f"  Seed candidate components: {list(seed_candidate.keys())}")
        self._log(f"  Train size: {len(train_data)}, Val size: {len(val_data)}")
        self._log(f"  Budget: {self.config.max_iterations} iterations, {self.config.max_metric_calls} metric calls")
        self._log("")

        # Step 0: Evaluate seed candidate on full validation set
        self._log("Evaluating seed candidate on validation set...")
        seed_scores = self._evaluate_on_valset(seed_candidate, val_data)
        seed_idx = self.state.add_candidate(
            text=seed_candidate,
            parent_ids=[],
            scores=seed_scores,
            iteration=0,
        )
        self.pareto.update(self.state)
        seed_avg = sum(seed_scores.values()) / max(len(seed_scores), 1)
        self._log(f"  Seed score: {seed_avg:.4f}")
        self._log("")

        # Main optimization loop
        for iteration in range(1, self.config.max_iterations + 1):
            if self.state.total_evals >= self.config.max_metric_calls:
                self._log(f"\nBudget exhausted ({self.state.total_evals} metric calls). Stopping.")
                break

            self.state.iteration = iteration
            self._log(f"--- Iteration {iteration} ---")

            # 1) Try merge first (if conditions met)
            merged = False
            if self.merge_proposer and self.merge_proposer.should_merge(self.state):
                merged = self._attempt_merge(val_data, iteration)
                if merged:
                    continue

            # 2) SELECT: Pick candidate from Pareto front
            selected_idx = self.pareto.select_candidate(self.state)
            selected = self.state.get_candidate(selected_idx)
            self._log(f"  Selected candidate {selected_idx} (score: {selected.average_score:.4f})")

            # 3) EXECUTE: Evaluate on minibatch with trace capture
            minibatch = self._sample_minibatch(train_data)
            eval_result = self.adapter.evaluate(
                batch=minibatch,
                candidate=selected.text,
                capture_traces=True,
            )
            self.state.total_evals += len(minibatch)
            subsample_score = sum(eval_result.scores) / max(len(eval_result.scores), 1)
            self._log(f"  Minibatch score: {subsample_score:.4f}")

            # Check if all perfect → skip
            if (
                self.config.skip_perfect_score
                and self.config.perfect_score is not None
                and all(s >= self.config.perfect_score for s in eval_result.scores)
            ):
                self._log(f"  All scores perfect on minibatch. Skipping iteration.")
                continue

            # 4) REFLECT + MUTATE: Use reflector to diagnose and propose fix
            components_to_update = self._select_components(selected.text)
            reflective_dataset = self.adapter.make_reflective_dataset(
                selected.text, eval_result, components_to_update
            )

            new_texts = {}
            for component in components_to_update:
                result = self.reflector.reflect_and_propose(
                    component_name=component,
                    current_instruction=selected.text[component],
                    reflective_dataset=reflective_dataset.get(component, []),
                )
                new_texts[component] = result["new_instruction"]
                self._log(f"  Reflected on '{component}': {result['diagnosis'][:80]}...")

            # Build new candidate
            new_candidate = selected.text.copy()
            for component, text in new_texts.items():
                new_candidate[component] = text

            # 5) ACCEPT: Evaluate new candidate on same minibatch
            eval_after = self.adapter.evaluate(
                batch=minibatch,
                candidate=new_candidate,
                capture_traces=False,
            )
            self.state.total_evals += len(minibatch)
            new_subsample_score = sum(eval_after.scores) / max(len(eval_after.scores), 1)
            self._log(f"  New minibatch score: {new_subsample_score:.4f}")

            # Acceptance criterion: new must be strictly better (with tolerance)
            if new_subsample_score > subsample_score + 1e-6:
                self._log(f"  ACCEPTED (improvement: +{new_subsample_score - subsample_score:.4f})")

                # Full evaluation on validation set
                val_scores = self._evaluate_on_valset(new_candidate, val_data)
                new_idx = self.state.add_candidate(
                    text=new_candidate,
                    parent_ids=[selected_idx],
                    scores=val_scores,
                    iteration=iteration,
                )
                self.pareto.update(self.state)

                val_avg = sum(val_scores.values()) / max(len(val_scores), 1)
                self._log(f"  Full val score: {val_avg:.4f}")

                # Trigger merge after improvement
                if self.merge_proposer and self.config.merge_after_improvement:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_attempts:
                        self.merge_proposer.merges_due += 1
            else:
                self._log(f"  REJECTED (no improvement: {new_subsample_score:.4f} <= {subsample_score:.4f})")

            # Record history
            self.state.history.append({
                "iteration": iteration,
                "selected_idx": selected_idx,
                "subsample_score_before": subsample_score,
                "subsample_score_after": new_subsample_score,
                "accepted": new_subsample_score > subsample_score,
                "total_evals": self.state.total_evals,
            })

            self._log("")

        # Final results
        best = self.state.get_best_candidate()
        pareto_candidates = self.state.get_pareto_candidates()

        self._log("=" * 60)
        self._log("OPTIMIZATION COMPLETE")
        self._log(f"  Total iterations: {self.state.iteration}")
        self._log(f"  Total metric calls: {self.state.total_evals}")
        self._log(f"  Best candidate idx: {best.idx}")
        self._log(f"  Best score: {best.average_score:.4f}")
        self._log(f"  Pareto front size: {len(pareto_candidates)}")
        self._log(f"  Total candidates explored: {len(self.state.candidates)}")

        return OptimizationResult(
            best_candidate=best.text,
            best_score=best.average_score,
            best_candidate_idx=best.idx,
            total_iterations=self.state.iteration,
            total_metric_calls=self.state.total_evals,
            pareto_front=[c.text for c in pareto_candidates],
            history=self.state.history,
            state=self.state,
        )

    def _attempt_merge(self, val_data: List[Dict[str, Any]], iteration: int) -> bool:
        """Attempt to merge two Pareto-optimal candidates."""
        self._log(f"  Attempting merge...")

        parent_a_idx, parent_b_idx = self.merge_proposer.select_parents(self.state)
        if parent_a_idx is None or parent_b_idx is None:
            self._log(f"  Cannot find merge parents. Skipping.")
            return False

        merged_text = self.merge_proposer.propose_merge(self.state, parent_a_idx, parent_b_idx)
        if merged_text is None:
            self._log(f"  Merge proposal failed.")
            return False

        # Evaluate merged candidate
        val_scores = self._evaluate_on_valset(merged_text, val_data)
        merged_avg = sum(val_scores.values()) / max(len(val_scores), 1)

        parent_a_score = self.state.candidates[parent_a_idx].average_score
        parent_b_score = self.state.candidates[parent_b_idx].average_score
        max_parent = max(parent_a_score, parent_b_score)

        if merged_avg >= max_parent:
            self._log(f"  MERGE ACCEPTED: {merged_avg:.4f} >= max({parent_a_score:.4f}, {parent_b_score:.4f})")
            self.state.add_candidate(
                text=merged_text,
                parent_ids=[parent_a_idx, parent_b_idx],
                scores=val_scores,
                iteration=iteration,
            )
            self.pareto.update(self.state)
            return True
        else:
            self._log(f"  MERGE REJECTED: {merged_avg:.4f} < max({parent_a_score:.4f}, {parent_b_score:.4f})")
            return False

    def _evaluate_on_valset(
        self, candidate: Dict[str, str], val_data: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """Evaluate candidate on full validation set, using cache."""
        scores = {}
        uncached = []

        for i, item in enumerate(val_data):
            cached = self.state.get_cached_score(candidate, i)
            if cached is not None:
                scores[i] = cached
            else:
                uncached.append((i, item))

        if uncached:
            batch = [item for _, item in uncached]
            eval_result = self.adapter.evaluate(batch, candidate, capture_traces=False)
            self.state.total_evals += len(batch)

            for (idx, _), score in zip(uncached, eval_result.scores):
                scores[idx] = score
                self.state.cache_score(candidate, idx, score)

        return scores

    def _sample_minibatch(self, train_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sample a random minibatch from training data."""
        size = min(self.config.minibatch_size, len(train_data))
        return random.sample(train_data, size)

    def _select_components(self, candidate: Dict[str, str]) -> List[str]:
        """Select which components to mutate this iteration."""
        components = list(candidate.keys())
        k = min(self.config.components_per_iteration, len(components))
        return random.sample(components, k)

    def _log(self, msg: str):
        """Log message if verbose."""
        if self.config.verbose:
            print(msg)
