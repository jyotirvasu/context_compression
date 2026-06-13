# Phase 1: LLMLingua — Prompt Compression Pipeline

Implementation of the LLMLingua prompt compression algorithm based on:

- **LLMLingua** (EMNLP 2023): Compressing Prompts for Accelerated Inference of Large Language Models
- **LongLLMLingua** (ACL 2024): Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
- Source: https://github.com/microsoft/LLMLingua

## Architecture

The pipeline implements a **coarse-to-fine** three-level compression strategy:

```
┌────────────────────────────────────────────────────────────────┐
│                    LLMLingua Pipeline                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Stage A: Budget Controller                                    │
│    └─ Allocates token budgets across instruction/context/      │
│       question based on compression sensitivity                │
│                                                                │
│  Stage B: Context-Level Filter (Coarse)                        │
│    └─ Ranks contexts by perplexity from small LM               │
│    └─ Selects most informative contexts within budget          │
│    └─ Supports: llmlingua, longllmlingua, bm25 ranking         │
│                                                                │
│  Stage C: Sentence-Level Filter (Medium)                       │
│    └─ Ranks sentences within selected contexts by PPL          │
│    └─ Preserves first/last sentences with priority bonus       │
│                                                                │
│  Stage D: Token-Level Compression (Fine)                       │
│    └─ Iterative windowed compression (iterative_size=200)      │
│    └─ Per-token NLL loss from small LM                         │
│    └─ Percentile-based threshold estimation                    │
│    └─ Force-preserves digits and specified tokens              │
│                                                                │
│  Stage E: Response Recovery & Assembly                         │
│    └─ Assembles compressed prompt                              │
│    └─ Computes compression statistics                          │
│    └─ Recovers original text from LLM responses               │
│    └─ Parses structured <llmlingua> tag prompts                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from Phase_1_LLMLingua import LLMLinguaPipeline

config = {
    "model_name": "gpt2",          # Small LM (gpt2, microsoft/phi-2, etc.)
    "device": "cpu",               # "cpu" or "cuda"
    "rate": 0.5,                   # Target compression rate
    "context_filter": {
        "enabled": True,
        "rank_method": "llmlingua",   # or "longllmlingua", "bm25"
    },
    "sentence_filter": {
        "enabled": False,             # Enable for sentence-level filtering
    },
    "token_compressor": {
        "enabled": True,
        "iterative_size": 200,
        "keep_split": False,
        "force_reserve_digit": True,
    },
    "recovery": {
        "concate_question": True,
        "add_instruction": False,
    },
}

pipeline = LLMLinguaPipeline(config)

result = pipeline.compress(
    instruction="Answer the question based on the context.",
    contexts=[
        "Document 1: The capital of France is Paris...",
        "Document 2: Python is a programming language...",
        "Document 3: Machine learning uses algorithms...",
    ],
    question="What is the capital of France?",
)

print(result.compressed_prompt)
print(f"Compression: {result.ratio} ({result.rate})")
print(f"Tokens: {result.origin_tokens} → {result.compressed_tokens}")
```

## Structured Compression

```python
structured = """<llmlingua, compress=False>Speaker 1:</llmlingua>\
<llmlingua, rate=0.4> Long content that should be heavily compressed.</llmlingua>\
<llmlingua, compress=False>Speaker 2:</llmlingua>\
<llmlingua, rate=0.6> Less compression needed here.</llmlingua>"""

result = pipeline.compress_structured(structured, rate=0.5)
```

## Module Structure

| File | Class | Role |
|------|-------|------|
| `stage_a_budget_controller.py` | `BudgetController` | Token budget allocation |
| `stage_b_context_filter.py` | `ContextFilter` | Coarse context-level ranking |
| `stage_c_sentence_filter.py` | `SentenceFilter` | Sentence-level filtering |
| `stage_d_token_compress.py` | `TokenCompressor` | Iterative token pruning |
| `stage_e_recovery.py` | `ResponseRecovery` | Assembly & response recovery |
| `pipeline.py` | `LLMLinguaPipeline` | End-to-end orchestrator |

## Key Concepts

- **Perplexity as information signal**: Tokens/sentences/contexts with higher perplexity carry more unique information and are preserved
- **Budget allocation**: Instructions and questions are highly sensitive to compression → preserved; contexts absorb most compression
- **Iterative windowed compression**: Process tokens in windows of `iterative_size`, estimating removal thresholds from the loss distribution
- **Dynamic compression ratio**: In LongLLMLingua, contexts closer to the question get higher preservation ratios
- **Force tokens**: Digits, newlines, and user-specified tokens are always preserved

## Dependencies

- `tiktoken` — Token counting (cl100k_base)
- `transformers` — Small LM loading (GPT-2 / LLaMA / phi-2)
- `torch` — Model inference
- `numpy` — Statistical threshold computation
- `nltk` — Sentence tokenization
- `rank_bm25` — BM25 ranking (optional)
