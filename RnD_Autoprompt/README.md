# AutoPrompt: Gradient-Guided Trigger Token Search

## Overview

AutoPrompt automatically discovers **trigger tokens** that, when added to a template,
cause masked language models (BERT, RoBERTa) to perform classification tasks
**without any fine-tuning**.

```
Template:  [T] [T] [T] {sentence} [P] .
                ↓ optimized triggers
Result:    "absolutely certainly clearly {sentence} [MASK] ."
                                                      ↓
                                            MLM predicts: "good" → positive
                                                          "bad"  → negative
```

## Key Idea

Instead of fine-tuning model parameters, AutoPrompt searches for discrete tokens
in the vocabulary that steer the model's predictions:

1. **Template**: Define positions for trigger tokens `[T]` and prediction `[P]`
2. **Gradient Signal**: Compute gradient of loss w.r.t. trigger embeddings
3. **HotFlip Attack**: Find vocab tokens whose embeddings align with gradient
4. **Greedy Search**: Evaluate candidates, keep best replacement

## Project Structure

```
autoprompt/
├── __init__.py          # Package exports
├── gradient_search.py   # GradientStorage, HotFlip attack, loss functions
├── template.py          # TriggerTemplatizer, token encoding
├── data.py              # Data loading (TSV/JSONL), collation, synthetic data
├── trigger_search.py    # Main AutoPromptSearcher class + SearchConfig
├── label_search.py      # Automatic label token discovery
└── run.py               # CLI entry point

autoprompt_example.py    # Standalone demo (mock + full modes)
```

## Quick Start

### Mock Mode (No Downloads Required)

```bash
python autoprompt_example.py --mode mock
```

### Full Mode (Requires BERT Model)

```bash
# Install dependencies
pip install torch transformers tqdm

# Run with synthetic data
python -m autoprompt.run --task sentiment --model-name bert-base-uncased

# Run with custom data
python -m autoprompt.run \
    --task sentiment \
    --train data/sentiment_train.jsonl \
    --dev data/sentiment_dev.jsonl \
    --model-name bert-base-uncased \
    --iters 50 \
    --num-cand 10
```

### Programmatic Usage

```python
from autoprompt.trigger_search import AutoPromptSearcher, SearchConfig

config = SearchConfig(
    model_name="bert-base-uncased",
    template="[T] [T] [T] {sentence} [P] .",
    label_map={"positive": "good", "negative": "bad"},
    num_candidates=10,
    iters=50,
)

searcher = AutoPromptSearcher(config)
result = searcher.search("data/train.jsonl", "data/dev.jsonl")

print(f"Best triggers: {result['best_tokens']}")
print(f"Dev accuracy: {result['best_dev_metric']:.4f}")
```

## Algorithm Details

### 1. Template Format

```
[T] [T] [T] {sentence} [P] .
 │   │   │      │       │
 └───┴───┘      │       └── Prediction position (→ [MASK])
      │         │
  Trigger    Data field
  positions  (from dataset)
```

### 2. Gradient-Guided Search (HotFlip)

For each trigger position `i`:
```
score(token) = embedding(token) · ∇_embedding L
```

The gradient `∇_embedding L` tells us which direction in embedding space
would decrease the loss. We find the discrete token closest to that direction.

### 3. Iterative Optimization

```
For each iteration:
    1. Pick random trigger position
    2. Accumulate gradients over training batches
    3. Find top-k candidates via HotFlip
    4. Evaluate each candidate (forward pass only)
    5. Keep best if it improves training metric
    6. Evaluate on dev set
```

## Supported Tasks

| Task | Template | Label Map |
|------|----------|-----------|
| Sentiment (SST-2) | `[T] [T] [T] {sentence} [P] .` | positive→"good", negative→"bad" |
| NLI (SICK/SNLI) | `{premise} [T] [T] [T] [P] {hypothesis}` | entailment→"yes", contradiction→"no" |
| Fact Retrieval | `{sub_label} [T] [T] [T] [T] [T] [P] .` | Open vocabulary |
| Relation Extraction | `{sub_label} [T] [T] [T] [T] [T] [P] .` + context | Open vocabulary |

## Data Format

### JSONL (Sentiment)
```json
{"sentence": "This movie is great", "label": "positive"}
{"sentence": "Terrible acting", "label": "negative"}
```

### JSONL (NLI)
```json
{"premise": "A dog runs", "hypothesis": "An animal moves", "label": "entailment"}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | bert-base-uncased | HuggingFace model |
| `num_candidates` | 10 | Candidates per position |
| `accumulation_steps` | 10 | Gradient accumulation batches |
| `iters` | 50 | Search iterations |
| `batch_size` | 32 | Training batch size |
| `patience` | 5 | Early stopping patience |
| `filter_special` | True | Filter special tokens |
| `filter_labels` | True | Filter label tokens |
