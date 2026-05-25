# Context Compression Pipeline v1

A modular, research-oriented baseline for **context compression** in LLM pipelines.  
Implements a five-stage pipeline from raw documents to optimally-packed compressed context.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT COMPRESSION PIPELINE v1                    │
├─────────┬──────────┬────────────┬──────────────┬────────────────────┤
│ Stage A │ Stage B  │  Stage C   │   Stage D    │      Stage E       │
│Chunking │ Cleanup  │ Retrieval  │ Compression  │  Position Packing  │
│         │          │            │              │                    │
│ Split   │ Normalize│ Embedding  │ Selective    │ Lost-in-the-Middle │
│ docs    │ & filter │ + BM25     │ Context      │ aware ordering     │
│ into    │ chunks   │ top-N      │ (GPT-2 self- │                    │
│ chunks  │          │ ranking    │  information)│ Edges get highest  │
│         │          │            │              │ relevance chunks   │
└─────────┴──────────┴────────────┴──────────────┴────────────────────┘
```

## Stages

| Stage | Description | Key Config |
|-------|-------------|-----------|
| **A** | Chunking (sentence/paragraph/fixed/sliding window) | `method`, `max_chunk_tokens` |
| **B** | Cleanup (URL removal, dedup, length filter, unicode) | `remove_urls`, `min_chunk_length` |
| **C** | Retrieval via dense embeddings, BM25, or hybrid fusion | `method`, `top_n`, `hybrid_alpha` |
| **D** | Selective Context compression (self-information filtering) | `model_type`, `reduce_ratio` |
| **E** | Position-aware packing (edges_first / round_robin) | `strategy`, `max_context_tokens` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the demo
python pipeline.py

# 3. Use in your code
from pipeline import ContextCompressionPipeline

pipe = ContextCompressionPipeline("config.yaml")
result = pipe.run(document_text, query="Your question here")
print(result.compressed_context)
print(f"Compression: {result.compression_ratio:.1%}")
```

## Configuration

All parameters are in `config.yaml`. Key knobs:

- **Chunking method**: `sentence` | `paragraph` | `fixed_token` | `sliding_window`
- **Retrieval method**: `embedding` | `bm25` | `hybrid`
- **Compression ratio**: `0.0` (no compression) to `1.0` (maximum)
- **Packing strategy**: `edges_first` | `decreasing` | `round_robin`

## Stage D: Selective Context (Plug-and-Play)

Computes self-information per lexical unit using GPT-2 and drops low-information units.

```python
# Standalone usage
from selective_context import SelectiveContext
sc = SelectiveContext(model_type='gpt2', lang='en')
compressed, removed = sc(text, reduce_ratio=0.5)
```

Falls back to simple truncation if the library is unavailable.

## Stage E: Position-Aware Packing

LLMs attend most to the **beginning** and **end** of context. Performance degrades significantly for information in the **middle**.

The `edges_first` strategy places the highest-relevance chunks at the start and end of the packed context, relegating less important chunks to the middle (low-attention zone).

## Project Structure

```
context_compression/
├── pipeline.py              # Main orchestrator
├── config.yaml              # All configuration
├── requirements.txt         # Dependencies
├── stages/
│   ├── __init__.py
│   ├── stage_a_chunking.py  # Document chunking
│   ├── stage_b_cleanup.py   # Text cleanup/filtering
│   ├── stage_c_retrieval.py # Embedding + BM25 retrieval
│   ├── stage_d_compression.py # Selective Context wrapper
│   └── stage_e_packing.py   # Position-aware packing
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Token counting, config loading
└── examples/
    └── demo.py              # Extended usage examples
```
