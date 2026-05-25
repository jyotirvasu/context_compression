# GEPA: Genetic-Pareto Reflective Prompt Evolution

Implementation based on:
> Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"  
> ICLR 2026 (Oral)  
> Paper: https://arxiv.org/abs/2507.19457  
> Original code: https://github.com/gepa-ai/gepa

## Overview

GEPA is a prompt optimization algorithm that uses **reflective evolution** instead of RL.
The key insight: rather than collapsing execution traces into scalar rewards (like GRPO/PPO),
GEPA uses an LLM to read full execution traces, diagnose failures, and propose targeted fixes.

### Results (from paper)
- **Outperforms GRPO** by 6% average across benchmarks
- **90x cheaper** than RL methods  
- **35x fewer rollouts** needed
- **Outperforms MIPROv2** (best previous prompt optimizer) by 10%+

## Algorithm

Each iteration follows 5 steps:

```
1. SELECT  → Pick candidate from Pareto frontier (exploration-aware)
2. EXECUTE → Run on minibatch, capturing full execution traces  
3. REFLECT → LLM reads traces, diagnoses failures (produces ASI)
4. MUTATE  → Generate improved candidate informed by reflection
5. ACCEPT  → If improved, add to pool and update Pareto front
```

Plus periodic **System-Aware Merge**: combine strengths of two
Pareto-optimal candidates that excel on different task subsets.

## Key Concepts

- **Actionable Side Information (ASI)**: Natural language diagnostics from 
  trace analysis — the text-optimization analogue of a gradient
- **Pareto frontier**: Maintains diversity by keeping candidates that excel 
  on different subsets (no single "best" — complementary candidates)
- **Reflective mutation**: LLM-based diagnosis + targeted fix (vs. random perturbation)
- **System-aware merge**: Intelligent crossover combining Pareto-optimal strengths

## Module Structure

```
gepa/
├── __init__.py          # Package exports
├── engine.py            # GEPAEngine - main optimization loop
├── state.py             # GEPAState - candidates, scores, cache
├── pareto.py            # ParetoFront - multi-objective selection
├── reflector.py         # Reflector - LLM trace analysis & mutation
├── merge.py             # MergeProposer - Pareto candidate fusion
├── adapter.py           # BaseAdapter - task interface
└── README.md            # This file
```

## Usage

### Mock Mode (no API keys needed)

```python
from gepa import GEPAEngine, GEPAConfig

config = GEPAConfig(
    max_iterations=20,
    max_metric_calls=200,
    mock_mode=True,
)

engine = GEPAEngine(config)
result = engine.optimize(
    seed_candidate={"system_prompt": "You are helpful."},
    train_data=[{"input": "...", "expected": "..."}],
    val_data=[{"input": "...", "expected": "..."}],
)

print(f"Best score: {result.best_score}")
print(f"Best prompt: {result.best_candidate}")
```

### With Real LLM (litellm)

```python
import litellm
from gepa import GEPAEngine, GEPAConfig
from gepa.reflector import Reflector
from gepa.adapter import DefaultAdapter

# Custom adapter with real LLM evaluation
def execute_fn(item, candidate):
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": candidate["system_prompt"]},
            {"role": "user", "content": item["input"]},
        ]
    )
    output = response.choices[0].message.content
    trace = f"Input: {item['input']}\nOutput: {output}"
    return output, trace

def score_fn(output, expected):
    return 1.0 if expected.lower() in output.lower() else 0.0

adapter = DefaultAdapter(execute_fn=execute_fn, score_fn=score_fn)
reflector = Reflector(lm=litellm.completion)

config = GEPAConfig(max_iterations=30, max_metric_calls=150)
engine = GEPAEngine(config, adapter=adapter, reflector=reflector)
result = engine.optimize(seed_candidate, train_data, val_data)
```

## Architecture Comparison

| Feature | RL (GRPO/PPO) | GEPA |
|---------|---------------|------|
| Signal type | Scalar reward | Full trace + NL diagnosis |
| Update mechanism | Policy gradient | Reflective mutation |
| Diversity | None (single policy) | Pareto frontier |
| Cost | High (many rollouts) | Low (35x fewer) |
| Interpretability | Low | High (NL explanations) |
