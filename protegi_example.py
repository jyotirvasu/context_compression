"""
ProTeGi Intent Classification Example
======================================
Demonstrates automatic prompt optimization for customer intent classification.

Setup:
    1. pip install anthropic  (or openai)
    2. Set API key: set ANTHROPIC_API_KEY=your_key  (or OPENAI_API_KEY)
    3. Run: python protegi_example.py

Based on: "Automatic Prompt Optimization with Gradient Descent and Beam Search"
(Pryzant et al., EMNLP 2023)
GitHub: https://github.com/pree-dew/protegi
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from RnD_ProTeGi.llm import ProviderFactory, LLMConfig
from RnD_ProTeGi.evaluation import ClassificationDataset, DatasetItem, PromptEvaluator
from RnD_ProTeGi.optimization import (
    GradientGenerator,
    PromptEditor,
    BanditBeamSearch,
    BanditBeamConfig,
)


def create_intent_dataset() -> ClassificationDataset:
    """Create a customer support intent classification dataset."""
    return ClassificationDataset(
        name="customer_intents",
        items=[
            # Refund requests
            DatasetItem("I want my money back", "refund"),
            DatasetItem("Can I get a refund for this purchase?", "refund"),
            DatasetItem("This product is defective, I need a return", "refund"),
            # Technical support
            DatasetItem("The app keeps crashing on my phone", "technical_support"),
            DatasetItem("I can't log into my account", "technical_support"),
            DatasetItem("The website is showing an error", "technical_support"),
            # Billing
            DatasetItem("My credit card was charged twice", "billing"),
            DatasetItem("Need to update my payment method", "billing"),
            DatasetItem("What's this charge on my statement?", "billing"),
            # General inquiries
            DatasetItem("What are your shipping options?", "general_inquiry"),
            DatasetItem("Do you have this in different colors?", "general_inquiry"),
            DatasetItem("When will new products be available?", "general_inquiry"),
        ],
    )


def main():
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Determine provider
    if os.getenv("ANTHROPIC_API_KEY"):
        provider_name = "anthropic"
        config = LLMConfig(model="claude-3-5-haiku-20241022")
    elif os.getenv("OPENAI_API_KEY"):
        provider_name = "openai"
        config = LLMConfig(model="gpt-3.5-turbo")
    else:
        # No API key → use mock provider for testing
        provider_name = "mock"
        api_key = "mock"
        config = LLMConfig(model="mock")

    provider = ProviderFactory.create(provider_name, api_key=api_key, config=config)
    dataset = create_intent_dataset()

    # Initial simple prompt
    initial_prompt = "What is the customer asking for?"

    print("=" * 60)
    print("ProTeGi - Prompt Optimization with Textual Gradients")
    print("=" * 60)
    print(f"\nProvider: {provider_name}")
    print(f"Model: {config.model}")
    print(f"Dataset: {dataset.name} ({len(dataset)} items, {dataset.num_labels} labels)")
    print(f"Labels: {sorted(dataset.labels)}")

    # Evaluate initial prompt
    evaluator = PromptEvaluator(provider, verbose=True)
    initial_result = evaluator.evaluate(initial_prompt, dataset)
    print(f"\nInitial prompt: '{initial_prompt}'")
    print(f"Initial F1 score: {initial_result.metrics.f1:.3f}")
    print(f"Initial accuracy: {initial_result.metrics.accuracy:.3f}")

    # Setup ProTeGi optimization
    generator = GradientGenerator(provider)
    editor = PromptEditor(provider)
    opt_config = BanditBeamConfig(
        beam_width=2,
        num_iterations=2,
        variants_per_candidate=2,
    )

    protegi = BanditBeamSearch(evaluator, generator, editor, opt_config)

    # Optimize prompt
    print("\nOptimizing prompt...")
    best = protegi.optimize(initial_prompt, dataset, metric="f1")

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Optimized prompt: '{best.prompt}'")
    print(f"Optimized F1 score: {best.mean_score:.3f}")
    print(f"Improvement: {((best.mean_score - initial_result.metrics.f1) / max(initial_result.metrics.f1, 0.001)) * 100:.1f}%")

    # Show optimization history
    print(f"\nOptimization History:")
    for state in protegi.get_history():
        print(f"  Iter {state.iteration}: best={state.best_score:.3f}, beam_size={len(state.candidates)}")


def dry_run():
    """Show algorithm structure without API calls."""
    print("""
┌──────────────────────────────────────────────────────────────┐
│               ProTeGi Algorithm Flow                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INITIALIZE                                               │
│     prompt₀ = "What is the customer asking for?"             │
│     score₀ = evaluate(prompt₀, dataset) → F1 score          │
│                                                              │
│  2. FOR each iteration:                                      │
│                                                              │
│     a. EVALUATE → Find errors                                │
│        errors = classify(prompt, each_item) → wrong ones     │
│                                                              │
│     b. GENERATE GRADIENT (textual)                           │
│        gradient = LLM("Why did this prompt fail on these     │
│                        examples? What pattern is missing?")  │
│        → "The prompt doesn't specify output categories..."   │
│                                                              │
│     c. EDIT PROMPT (apply gradient)                          │
│        variants = LLM("Rewrite prompt to fix: {gradient}")   │
│        → Multiple variants at different temperatures         │
│                                                              │
│     d. UCB BUDGET ALLOCATION                                 │
│        UCB(candidate) = mean + c·√(log(T)/n)                 │
│        → Allocate more variants to promising candidates      │
│                                                              │
│     e. SUCCESSIVE REJECTS (prune beam)                       │
│        → Keep top beam_width candidates                      │
│                                                              │
│  3. RETURN best candidate                                    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Example improvement:                                        │
│  "What is the customer asking for?"                          │
│     → F1: 0.062                                              │
│                                                              │
│  "Identify the customer's request by selecting from these    │
│   predefined categories: [refund, technical_support,         │
│   billing, general_inquiry]. Respond with the EXACT          │
│   matching category name."                                   │
│     → F1: 0.760  (1135% improvement)                        │
└──────────────────────────────────────────────────────────────┘

Project Structure:
    protegi/
    ├── __init__.py
    ├── llm/
    │   ├── base.py              # BaseLLMProvider, LLMConfig, LLMResponse
    │   ├── anthropic_provider.py
    │   ├── openai_provider.py
    │   └── factory.py           # ProviderFactory.create("anthropic", key)
    ├── evaluation/
    │   ├── dataset.py           # DatasetItem, ClassificationDataset
    │   ├── metrics.py           # F1, precision, recall, confusion matrix
    │   └── evaluator.py         # PromptEvaluator (with caching)
    └── optimization/
        ├── gradient_generator.py # LLM-based error analysis → "gradient"
        ├── prompt_editor.py      # LLM-based prompt rewriting
        ├── candidate.py          # Candidate tracking (scores, trials)
        ├── bandits.py            # UCB + Successive Rejects
        └── bandit_beam_search.py # Main ProTeGi optimization loop
""")


if __name__ == "__main__":
    main()
