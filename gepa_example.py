"""
GEPA Example: Prompt optimization with mock mode.
Demonstrates the full GEPA pipeline without requiring LLM API keys.

This example optimizes a "question answering" prompt through
reflective evolution on a mock task.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Phase_2_GEPA import GEPAEngine, GEPAConfig


def main():
    print("=" * 70)
    print("GEPA: Genetic-Pareto Reflective Prompt Evolution")
    print("Paper: Agrawal et al. (ICLR 2026 Oral)")
    print("https://arxiv.org/abs/2507.19457")
    print("=" * 70)
    print()

    # === Define the seed candidate ===
    # In a real scenario, this would be your initial prompt(s)
    seed_candidate = {
        "system_prompt": "You are a helpful assistant that answers questions accurately.",
        "output_format": "Provide a clear, concise answer.",
    }

    # === Define training data (used for minibatch evaluation) ===
    train_data = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is 2 + 2?", "expected": "4"},
        {"input": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
        {"input": "What is the chemical symbol for water?", "expected": "H2O"},
        {"input": "What year did WW2 end?", "expected": "1945"},
        {"input": "What is the largest planet?", "expected": "Jupiter"},
        {"input": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
        {"input": "What is the speed of light?", "expected": "299,792,458 m/s"},
        {"input": "What is the powerhouse of the cell?", "expected": "Mitochondria"},
        {"input": "How many continents are there?", "expected": "7"},
        {"input": "What is the boiling point of water in Celsius?", "expected": "100"},
        {"input": "Who discovered gravity?", "expected": "Newton"},
        {"input": "What is the largest ocean?", "expected": "Pacific Ocean"},
        {"input": "What language is spoken in Brazil?", "expected": "Portuguese"},
        {"input": "What is DNA short for?", "expected": "Deoxyribonucleic acid"},
    ]

    # === Define validation data (used for full evaluation / acceptance) ===
    val_data = [
        {"input": "What is the smallest prime number?", "expected": "2"},
        {"input": "Who was the first person on the moon?", "expected": "Neil Armstrong"},
        {"input": "What is the capital of Japan?", "expected": "Tokyo"},
        {"input": "What is the formula for energy?", "expected": "E=mc^2"},
        {"input": "How many bones in the human body?", "expected": "206"},
        {"input": "What is the largest mammal?", "expected": "Blue whale"},
        {"input": "Who invented the telephone?", "expected": "Alexander Graham Bell"},
        {"input": "What is the hardest natural substance?", "expected": "Diamond"},
        {"input": "What planet is known as the Red Planet?", "expected": "Mars"},
        {"input": "What is the main gas in Earth's atmosphere?", "expected": "Nitrogen"},
    ]

    # === Configure GEPA ===
    config = GEPAConfig(
        max_iterations=20,
        max_metric_calls=200,
        minibatch_size=5,
        frontier_type="pareto",
        use_merge=True,
        max_merge_attempts=3,
        mock_mode=True,  # No LLM API keys needed
        seed=42,
        verbose=True,
        components_per_iteration=1,
    )

    # === Run optimization ===
    engine = GEPAEngine(config)
    result = engine.optimize(
        seed_candidate=seed_candidate,
        train_data=train_data,
        val_data=val_data,
    )

    # === Display results ===
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nSeed candidate score: {result.state.candidates[0].average_score:.4f}")
    print(f"Best candidate score: {result.best_score:.4f}")
    print(f"Improvement: +{result.best_score - result.state.candidates[0].average_score:.4f}")
    print(f"\nTotal iterations: {result.total_iterations}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"Candidates explored: {len(result.state.candidates)}")
    print(f"Pareto front size: {len(result.pareto_front)}")

    print("\n--- Best Candidate ---")
    for component, text in result.best_candidate.items():
        print(f"\n[{component}]:")
        # Show first 200 chars
        display = text[:200] + "..." if len(text) > 200 else text
        print(f"  {display}")

    print("\n--- Pareto Front ---")
    for i, candidate in enumerate(result.pareto_front):
        c = result.state.get_pareto_candidates()[i]
        print(f"  Candidate {c.idx} (score: {c.average_score:.4f})")

    # Show optimization trajectory
    print("\n--- Optimization Trajectory ---")
    accepted_count = sum(1 for h in result.history if h["accepted"])
    rejected_count = sum(1 for h in result.history if not h["accepted"])
    print(f"  Proposals accepted: {accepted_count}")
    print(f"  Proposals rejected: {rejected_count}")
    print(f"  Acceptance rate: {accepted_count / max(accepted_count + rejected_count, 1) * 100:.1f}%")

    # Show score progression of accepted candidates
    print("\n--- Score Progression (accepted) ---")
    for h in result.history:
        if h["accepted"]:
            print(f"  Iter {h['iteration']:3d}: {h['subsample_score_before']:.4f} -> {h['subsample_score_after']:.4f} "
                  f"(+{h['subsample_score_after'] - h['subsample_score_before']:.4f})")


if __name__ == "__main__":
    main()
