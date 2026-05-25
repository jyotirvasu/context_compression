"""
Bandit-Enhanced Beam Search
Beam search with smart resource allocation using bandit algorithms.

This is the core ProTeGi optimization loop:
1. Evaluate current prompt → find errors
2. Generate textual gradient from errors (why it failed)
3. Edit prompt using gradient → produce variants
4. Use bandits for smart allocation/pruning
5. Repeat
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..evaluation.evaluator import PromptEvaluator
from ..evaluation.dataset import ClassificationDataset
from .candidate import Candidate, BeamState
from .gradient_generator import GradientGenerator
from .prompt_editor import PromptEditor
from .bandits import UCB, SuccessiveRejects, BanditStats


@dataclass
class BeamSearchConfig:
    """Configuration for beam search."""
    beam_width: int = 3
    num_iterations: int = 5
    variants_per_candidate: int = 3
    early_stop_patience: int = 2
    early_stop_threshold: float = 0.01


@dataclass
class BanditBeamConfig(BeamSearchConfig):
    """Configuration for bandit-enhanced beam search."""
    use_ucb: bool = True
    use_successive_rejects: bool = True
    ucb_exploration: float = 1.414


class BanditBeamSearch:
    """
    ProTeGi: Prompt Optimization with Textual Gradients.

    Combines beam search with bandit algorithms to efficiently
    optimize prompts through iterative gradient-based editing.

    Algorithm:
        1. Start with initial prompt
        2. For each iteration:
            a. Evaluate each beam candidate
            b. Generate gradients (error analysis)
            c. Edit prompts (apply gradient)
            d. Use UCB to allocate budget
            e. Use Successive Rejects to prune beam
        3. Return best candidate
    """

    def __init__(
        self,
        evaluator: PromptEvaluator,
        gradient_generator: GradientGenerator,
        prompt_editor: PromptEditor,
        config: Optional[BanditBeamConfig] = None,
    ):
        self.evaluator = evaluator
        self.generator = gradient_generator
        self.editor = prompt_editor
        self.config = config or BanditBeamConfig()

        # Bandit algorithms
        self.ucb = UCB(exploration_constant=self.config.ucb_exploration)
        self.successive_rejects = SuccessiveRejects()

        # Tracking
        self._history: List[BeamState] = []
        self._stats: Dict[str, Any] = {
            "total_iterations": 0,
            "total_candidates": 0,
            "ucb_savings": 0,
        }

    def optimize(
        self,
        initial_prompt: str,
        dataset: ClassificationDataset,
        metric: str = "f1",
    ) -> Candidate:
        """
        Optimize a prompt using ProTeGi.

        Args:
            initial_prompt: Starting prompt to optimize
            dataset: Dataset to evaluate on
            metric: Metric to optimize ("f1", "accuracy")

        Returns:
            Best Candidate found during optimization
        """
        # Initialize beam with the starting prompt
        initial_candidate = Candidate(prompt=initial_prompt, metadata={"generation": 0})
        initial_score = self._evaluate_candidate(initial_candidate, dataset, metric)
        initial_candidate.add_score(initial_score)

        beam = [initial_candidate]
        self._save_state(0, beam)

        best_score_history = [initial_score]

        # Optimization loop
        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n🔄 Iteration {iteration}/{self.config.num_iterations}")
            print(f"   Current beam: {len(beam)} candidates")
            print(f"   Best score: {max(c.mean_score for c in beam):.3f}")

            # Expand beam: generate new variants
            all_variants = self._expand_beam(beam, dataset, iteration, metric)
            print(f"   Generated: {len(all_variants) - len(beam)} new variants")

            # Prune beam to beam_width
            if self.config.use_successive_rejects:
                beam = self.successive_rejects.prune(
                    all_variants, self.config.beam_width
                )
            else:
                # Simple top-k pruning
                beam = sorted(all_variants, key=lambda c: c.mean_score, reverse=True)
                beam = beam[:self.config.beam_width]

            current_best = max(c.mean_score for c in beam)
            print(f"   Pruned to: {len(beam)} candidates")
            print(f"   New best: {current_best:.3f}")

            self._save_state(iteration, beam)
            best_score_history.append(current_best)

            # Early stopping
            if self._should_stop(best_score_history):
                print(f"   ⚠️ Early stopping: converged")
                break

        self._stats["total_iterations"] = iteration
        return max(beam, key=lambda c: c.mean_score)

    def _expand_beam(
        self,
        beam: List[Candidate],
        dataset: ClassificationDataset,
        iteration: int,
        metric: str,
    ) -> List[Candidate]:
        """Expand beam by generating variants for each candidate."""
        all_variants = list(beam)  # Keep existing candidates

        # Determine budget allocation per candidate
        if self.config.use_ucb:
            total_budget = self.config.variants_per_candidate * len(beam)
            allocations = self.ucb.allocate_budget(beam, total_budget, min_allocation=1)
        else:
            allocations = {c.prompt: self.config.variants_per_candidate for c in beam}

        for candidate in beam:
            num_variants = allocations.get(candidate.prompt, 1)
            variants = self._generate_variants(
                candidate, dataset, iteration, metric, num_variants
            )
            all_variants.extend(variants)

        return all_variants

    def _generate_variants(
        self,
        candidate: Candidate,
        dataset: ClassificationDataset,
        iteration: int,
        metric: str,
        num_variants: int,
    ) -> List[Candidate]:
        """Generate improved variants of a candidate."""
        # Evaluate to get errors
        result = self.evaluator.evaluate(candidate.prompt, dataset)

        if not result.errors:
            return []  # No errors = perfect score, nothing to improve

        # Generate gradient (textual analysis of errors)
        gradient_result = self.generator.generate(
            candidate.prompt, result.errors
        )

        # Edit prompt based on gradient
        edit_result = self.editor.edit(
            candidate.prompt,
            gradient_result.gradient,
            num_variants=num_variants,
        )

        # Evaluate each variant
        variants = []
        for i, variant_prompt in enumerate(edit_result.edited_prompts):
            if variant_prompt == candidate.prompt:
                continue  # Skip if identical to parent

            variant = Candidate(
                prompt=variant_prompt,
                metadata={
                    "generation": iteration,
                    "parent": candidate.prompt[:50],
                    "gradient": gradient_result.gradient[:50] + "...",
                    "temperature": edit_result.temperatures_used[i],
                },
            )

            score = self._evaluate_candidate(variant, dataset, metric)
            variant.add_score(score)
            variants.append(variant)
            self._stats["total_candidates"] += 1

        return variants

    def _evaluate_candidate(
        self, candidate: Candidate, dataset: ClassificationDataset, metric: str
    ) -> float:
        """Evaluate a candidate and return its score."""
        return self.evaluator.evaluate_score(candidate.prompt, dataset, metric)

    def _should_stop(self, score_history: List[float]) -> bool:
        """Check early stopping condition."""
        patience = self.config.early_stop_patience
        threshold = self.config.early_stop_threshold

        if len(score_history) < patience + 1:
            return False

        recent = score_history[-patience:]
        improvement = max(recent) - min(recent)
        return improvement < threshold

    def _save_state(self, iteration: int, beam: List[Candidate]):
        """Save beam state for history tracking."""
        best_score = max(c.mean_score for c in beam) if beam else 0.0
        self._history.append(BeamState(
            iteration=iteration,
            candidates=beam.copy(),
            best_score=best_score,
        ))

    def get_history(self) -> List[BeamState]:
        return self._history

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()
