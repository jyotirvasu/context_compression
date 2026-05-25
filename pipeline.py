"""
Context Compression Pipeline v1
================================
End-to-end orchestrator that chains all five stages:
    A. Chunking -> B. Cleanup -> C. Retrieval -> D. Compression -> E. Packing

Usage:
    from pipeline import ContextCompressionPipeline
    pipe = ContextCompressionPipeline("config.yaml")
    result = pipe.run(document_text, query="What is X?")
"""

from dataclasses import dataclass
from typing import List, Optional

from utils.helpers import load_config, count_tokens, compute_compression_ratio
from Phase_1_CC_PA import Chunker, Cleaner, Retriever, Compressor, PositionAwarePacker
from Phase_1_CC_PA.stage_a_chunking import Chunk


@dataclass
class PipelineResult:
    """Container for pipeline output and diagnostics."""
    compressed_context: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    num_chunks_created: int
    num_chunks_after_cleanup: int
    num_chunks_retrieved: int
    num_chunks_after_compression: int
    position_map: dict


class ContextCompressionPipeline:
    """Main pipeline orchestrating all five stages."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # Initialize stages
        self.chunker = Chunker(self.config.get("chunking", {}))
        self.cleaner = Cleaner(self.config.get("cleanup", {}))
        self.retriever = Retriever(self.config.get("retrieval", {}))
        self.compressor = Compressor(self.config.get("compression", {}))
        self.packer = PositionAwarePacker(self.config.get("packing", {}))

    def run(self, document: str, query: str) -> PipelineResult:
        """Execute the full compression pipeline.

        Args:
            document: The full input document/context to compress.
            query: The user query to guide relevance-based retrieval.

        Returns:
            PipelineResult with compressed context and diagnostics.
        """
        original_tokens = count_tokens(document)

        # Stage A: Chunking
        chunks = self.chunker.chunk(document)
        num_chunks_created = len(chunks)

        # Stage B: Cleanup
        chunks = self.cleaner.clean(chunks)
        num_chunks_after_cleanup = len(chunks)

        # Stage C: Retrieval (top-N by relevance)
        ranked_chunks = self.retriever.retrieve(chunks, query)
        num_chunks_retrieved = len(ranked_chunks)

        # Stage D: Selective Context Compression
        compressed_chunks = self.compressor.compress(ranked_chunks, query)
        num_chunks_after_compression = len(compressed_chunks)

        # Stage E: Position-Aware Packing
        packed_context = self.packer.pack(compressed_chunks)
        position_map = self.packer.get_position_map(compressed_chunks)

        compressed_tokens = count_tokens(packed_context)
        ratio = compute_compression_ratio(document, packed_context)

        return PipelineResult(
            compressed_context=packed_context,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=ratio,
            num_chunks_created=num_chunks_created,
            num_chunks_after_cleanup=num_chunks_after_cleanup,
            num_chunks_retrieved=num_chunks_retrieved,
            num_chunks_after_compression=num_chunks_after_compression,
            position_map=position_map,
        )

    def run_stages_individually(
        self, document: str, query: str
    ) -> dict:
        """Run each stage separately for debugging/analysis.

        Returns a dict with intermediate results from each stage.
        """
        results = {}

        # Stage A
        chunks = self.chunker.chunk(document)
        results["stage_a_chunks"] = chunks

        # Stage B
        cleaned = self.cleaner.clean(chunks)
        results["stage_b_cleaned"] = cleaned

        # Stage C
        ranked = self.retriever.retrieve(cleaned, query)
        results["stage_c_ranked"] = ranked

        # Stage D
        compressed = self.compressor.compress(ranked, query)
        results["stage_d_compressed"] = compressed

        # Stage E
        packed = self.packer.pack(compressed)
        results["stage_e_packed"] = packed
        results["position_map"] = self.packer.get_position_map(compressed)

        return results


if __name__ == "__main__":
    import sys

    # Quick demo with sample text
    sample_doc = """
    Machine learning is a subset of artificial intelligence that focuses on building
    systems that learn from data. Unlike traditional programming where rules are
    explicitly coded, machine learning algorithms build models based on sample data,
    known as training data, to make predictions or decisions without being explicitly
    programmed to do so.

    Deep learning is a subset of machine learning that uses neural networks with many
    layers. These deep neural networks have been particularly successful in areas such
    as computer vision, natural language processing, and speech recognition. The key
    advantage of deep learning is its ability to automatically learn hierarchical
    representations from raw data.

    Transfer learning is a technique where a model trained on one task is repurposed
    for a second related task. This is particularly useful when the second task has
    limited training data. Pre-trained models like BERT, GPT, and ResNet serve as
    powerful starting points for many downstream tasks.

    Reinforcement learning is an area of machine learning where an agent learns to
    make decisions by interacting with an environment. The agent receives rewards or
    penalties based on its actions and learns to maximize cumulative reward over time.
    Applications include game playing, robotics, and autonomous driving.

    Natural language processing (NLP) is a field at the intersection of computer
    science, artificial intelligence, and linguistics. It focuses on the interaction
    between computers and humans through natural language. Recent advances in NLP have
    been driven by large language models that are pre-trained on massive text corpora.
    """

    query = "What is transfer learning and how does it relate to pre-trained models?"

    print("=" * 70)
    print("Context Compression Pipeline v1 - Demo")
    print("=" * 70)

    pipeline = ContextCompressionPipeline()
    result = pipeline.run(sample_doc, query)

    print(f"\nOriginal tokens:    {result.original_token_count}")
    print(f"Compressed tokens:  {result.compressed_token_count}")
    print(f"Compression ratio:  {result.compression_ratio:.1%}")
    print(f"\nChunks created:     {result.num_chunks_created}")
    print(f"After cleanup:      {result.num_chunks_after_cleanup}")
    print(f"After retrieval:    {result.num_chunks_retrieved}")
    print(f"After compression:  {result.num_chunks_after_compression}")
    print(f"\n{'─' * 70}")
    print("Compressed Context:")
    print(f"{'─' * 70}")
    print(result.compressed_context)
    print(f"{'─' * 70}")
    print("\nPosition Map:")
    for idx, info in result.position_map.items():
        print(f"  Chunk {idx}: {info['attention_zone']} (pos={info['packed_position']})")
