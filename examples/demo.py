"""
Extended demo showing individual stage usage and full pipeline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ContextCompressionPipeline
from Phase_1_CC_PA.stage_a_chunking import Chunker
from Phase_1_CC_PA.stage_b_cleanup import Cleaner
from Phase_1_CC_PA.stage_c_retrieval import Retriever
from Phase_1_CC_PA.stage_e_packing import PositionAwarePacker
from utils.helpers import load_config, count_tokens


SAMPLE_DOCUMENT = """
The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
has revolutionized natural language processing. It relies entirely on self-attention mechanisms
to compute representations of its input and output without using sequence-aligned RNNs or
convolution. The key innovation is the multi-head attention mechanism, which allows the model
to jointly attend to information from different representation subspaces at different positions.

BERT (Bidirectional Encoder Representations from Transformers) was introduced by Devlin et al.
in 2018. It is designed to pre-train deep bidirectional representations from unlabeled text by
jointly conditioning on both left and right context in all layers. BERT can be fine-tuned with
just one additional output layer to create state-of-the-art models for a wide range of tasks.

GPT (Generative Pre-trained Transformer) models use a left-to-right autoregressive approach.
Starting with GPT-1, the series progressed through GPT-2, GPT-3, and GPT-4, each significantly
larger than its predecessor. GPT-3, with 175 billion parameters, demonstrated remarkable
few-shot learning capabilities without task-specific fine-tuning.

Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generative
approaches. Instead of relying solely on the model's parametric knowledge, RAG retrieves
relevant documents from an external knowledge base and conditions the generation on both the
input query and the retrieved documents. This approach reduces hallucination and enables
knowledge updates without retraining.

Context window limitations remain a significant challenge for large language models. While
recent models like Claude and GPT-4 support context windows of 100K+ tokens, efficiently
utilizing this context remains an open research problem. Studies have shown that models
struggle with information in the middle of long contexts, leading to research on context
compression and position-aware strategies.

Prompt compression techniques aim to reduce the number of tokens sent to an LLM while
preserving the essential information. Methods include extractive approaches (selecting key
sentences), abstractive approaches (summarizing), and information-theoretic approaches
(removing low-information-content tokens based on self-information or entropy).
"""


def demo_individual_stages():
    """Demonstrate each stage independently."""
    config = load_config(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    )

    print("=" * 70)
    print("STAGE A: Chunking Demo")
    print("=" * 70)
    chunker = Chunker(config["chunking"])
    chunks = chunker.chunk(SAMPLE_DOCUMENT)
    print(f"Created {len(chunks)} chunks using '{config['chunking']['method']}' method")
    for c in chunks[:3]:
        print(f"  Chunk {c.index}: {c.token_count} tokens | '{c.text[:60]}...'")

    print(f"\n{'=' * 70}")
    print("STAGE B: Cleanup Demo")
    print("=" * 70)
    cleaner = Cleaner(config["cleanup"])
    cleaned = cleaner.clean(chunks)
    print(f"Chunks after cleanup: {len(cleaned)} (from {len(chunks)})")

    print(f"\n{'=' * 70}")
    print("STAGE C: Retrieval Demo")
    print("=" * 70)
    query = "How does RAG reduce hallucination in language models?"
    retriever = Retriever(config["retrieval"])
    ranked = retriever.retrieve(cleaned, query)
    print(f"Query: '{query}'")
    print(f"Top-{len(ranked)} chunks retrieved:")
    for chunk, score in ranked[:5]:
        print(f"  Score {score:.4f}: '{chunk.text[:60]}...'")

    print(f"\n{'=' * 70}")
    print("STAGE E: Position-Aware Packing Demo")
    print("=" * 70)
    packer = PositionAwarePacker(config["packing"])
    # Use retrieved chunks directly for packing demo
    retrieved_chunks = [c for c, _ in ranked]
    packed = packer.pack(retrieved_chunks)
    print(f"Packed context: {count_tokens(packed)} tokens")
    print(f"Strategy: {config['packing']['strategy']}")
    pos_map = packer.get_position_map(retrieved_chunks)
    for idx, info in pos_map.items():
        print(f"  Chunk {idx}: {info['attention_zone']}")


def demo_full_pipeline():
    """Run the complete pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE DEMO")
    print("=" * 70)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    pipeline = ContextCompressionPipeline(config_path)

    query = "What are context compression techniques for LLMs?"
    result = pipeline.run(SAMPLE_DOCUMENT, query)

    print(f"\nQuery: '{query}'")
    print(f"\nOriginal tokens:     {result.original_token_count}")
    print(f"Compressed tokens:   {result.compressed_token_count}")
    print(f"Compression ratio:   {result.compression_ratio:.1%}")
    print(f"Chunks created:      {result.num_chunks_created}")
    print(f"After cleanup:       {result.num_chunks_after_cleanup}")
    print(f"Retrieved (top-N):   {result.num_chunks_retrieved}")
    print(f"After compression:   {result.num_chunks_after_compression}")
    print(f"\n{'─' * 70}")
    print("Final Compressed Context:")
    print(f"{'─' * 70}")
    print(result.compressed_context[:500] + "..." if len(result.compressed_context) > 500 else result.compressed_context)


if __name__ == "__main__":
    demo_individual_stages()
    demo_full_pipeline()
