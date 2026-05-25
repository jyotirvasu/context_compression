"""Evaluation metrics for Context Compression Pipeline."""

import time
import re
from pipeline import ContextCompressionPipeline
from utils.helpers import count_tokens

# Test documents of varying sizes
docs = {
    "short": (
        "Large language models like GPT-4 and Claude use transformer architectures with billions of parameters. "
        "They are trained on massive text corpora using self-supervised learning objectives. "
        "Fine-tuning adapts pre-trained models to specific downstream tasks using labeled data. "
        "Prompt engineering involves crafting input text to elicit desired model behavior without retraining. "
        "Context window limitations restrict how much text can be processed in a single forward pass. "
        "Retrieval-augmented generation retrieves relevant documents to ground model responses in facts."
    ),
    "medium": (
        "The Transformer architecture, introduced in Attention Is All You Need (Vaswani et al., 2017), "
        "has revolutionized natural language processing. It relies entirely on self-attention mechanisms "
        "to compute representations of its input and output without using sequence-aligned RNNs or "
        "convolution. The key innovation is the multi-head attention mechanism, which allows the model "
        "to jointly attend to information from different representation subspaces at different positions.\n\n"
        "BERT (Bidirectional Encoder Representations from Transformers) was introduced by Devlin et al. "
        "in 2018. It is designed to pre-train deep bidirectional representations from unlabeled text by "
        "jointly conditioning on both left and right context in all layers. BERT can be fine-tuned with "
        "just one additional output layer to create state-of-the-art models for a wide range of tasks.\n\n"
        "GPT (Generative Pre-trained Transformer) models use a left-to-right autoregressive approach. "
        "Starting with GPT-1, the series progressed through GPT-2, GPT-3, and GPT-4, each significantly "
        "larger than its predecessor. GPT-3, with 175 billion parameters, demonstrated remarkable "
        "few-shot learning capabilities without task-specific fine-tuning.\n\n"
        "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generative "
        "approaches. Instead of relying solely on the model's parametric knowledge, RAG retrieves "
        "relevant documents from an external knowledge base and conditions the generation on both the "
        "input query and the retrieved documents. This approach reduces hallucination and enables "
        "knowledge updates without retraining.\n\n"
        "Context window limitations remain a significant challenge for large language models. While "
        "recent models like Claude and GPT-4 support context windows of 100K+ tokens, efficiently "
        "utilizing this context remains an open research problem. Studies have shown that models "
        "struggle with information in the middle of long contexts, leading to research on context "
        "compression and position-aware strategies.\n\n"
        "Prompt compression techniques aim to reduce the number of tokens sent to an LLM while "
        "preserving the essential information. Methods include extractive approaches (selecting key "
        "sentences), abstractive approaches (summarizing), and information-theoretic approaches "
        "(removing low-information-content tokens based on self-information or entropy)."
    ),
    "large": (
        "The Transformer architecture, introduced in Attention Is All You Need (Vaswani et al., 2017), "
        "has revolutionized natural language processing. It relies entirely on self-attention mechanisms "
        "to compute representations of its input and output without using sequence-aligned RNNs or "
        "convolution. The key innovation is the multi-head attention mechanism, which allows the model "
        "to jointly attend to information from different representation subspaces at different positions.\n\n"
        "BERT (Bidirectional Encoder Representations from Transformers) was introduced by Devlin et al. "
        "in 2018. It is designed to pre-train deep bidirectional representations from unlabeled text by "
        "jointly conditioning on both left and right context in all layers. BERT can be fine-tuned with "
        "just one additional output layer to create state-of-the-art models for a wide range of tasks.\n\n"
        "GPT (Generative Pre-trained Transformer) models use a left-to-right autoregressive approach. "
        "Starting with GPT-1, the series progressed through GPT-2, GPT-3, and GPT-4, each significantly "
        "larger than its predecessor. GPT-3, with 175 billion parameters, demonstrated remarkable "
        "few-shot learning capabilities without task-specific fine-tuning.\n\n"
        "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generative "
        "approaches. Instead of relying solely on the model's parametric knowledge, RAG retrieves "
        "relevant documents from an external knowledge base and conditions the generation on both the "
        "input query and the retrieved documents. This approach reduces hallucination and enables "
        "knowledge updates without retraining.\n\n"
        "Context window limitations remain a significant challenge for large language models. While "
        "recent models like Claude and GPT-4 support context windows of 100K+ tokens, efficiently "
        "utilizing this context remains an open research problem. Studies have shown that models "
        "struggle with information in the middle of long contexts, leading to research on context "
        "compression and position-aware strategies.\n\n"
        "Prompt compression techniques aim to reduce the number of tokens sent to an LLM while "
        "preserving the essential information. Methods include extractive approaches (selecting key "
        "sentences), abstractive approaches (summarizing), and information-theoretic approaches "
        "(removing low-information-content tokens based on self-information or entropy).\n\n"
        "Vector databases store embeddings of documents in high-dimensional spaces, enabling fast "
        "approximate nearest neighbor search. Popular implementations include FAISS, Pinecone, Weaviate, "
        "and ChromaDB. These systems support semantic search by comparing query embeddings to stored "
        "document embeddings using metrics like cosine similarity or dot product.\n\n"
        "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model. "
        "The student learns to mimic the teacher's output distribution rather than just the hard labels, "
        "preserving more nuanced information about class relationships. This technique enables deployment "
        "of smaller, faster models that retain much of the original model's capability.\n\n"
        "Mixture of Experts (MoE) architectures activate only a subset of model parameters for each input, "
        "enabling efficient scaling to very large parameter counts. Models like Switch Transformer and "
        "Mixtral use routing networks to direct each token to the most relevant expert subnetworks. "
        "This approach achieves better performance per compute than dense models of equivalent size.\n\n"
        "Reinforcement Learning from Human Feedback (RLHF) aligns language models with human preferences. "
        "The process involves training a reward model on human preference data, then optimizing the language "
        "model using proximal policy optimization (PPO) to maximize the learned reward. RLHF has been "
        "crucial for making models like ChatGPT helpful, harmless, and honest.\n\n"
        "Constitutional AI (CAI) extends alignment by having the model critique and revise its own outputs "
        "based on a set of principles. This reduces the need for human feedback while maintaining alignment. "
        "The model is trained to identify and correct harmful, biased, or unhelpful content in its responses "
        "using self-supervised critiques guided by constitutional principles."
    ),
}

queries = {
    "short": "How do LLMs handle context limitations?",
    "medium": "What are context compression techniques for LLMs?",
    "large": "How does RLHF align language models with human preferences?",
}

# GPT-4 pricing (per 1K tokens)
INPUT_COST_PER_1K = 0.03


def keyword_retention(original, compressed, query):
    orig_words = set(re.findall(r"\b[a-z]{4,}\b", original.lower()))
    comp_words = set(re.findall(r"\b[a-z]{4,}\b", compressed.lower()))
    retained = orig_words.intersection(comp_words)
    return len(retained) / len(orig_words) * 100 if orig_words else 0


def main():
    pipe = ContextCompressionPipeline("config.yaml")

    print("=" * 80)
    print("CONTEXT COMPRESSION PIPELINE - EVALUATION METRICS")
    print("=" * 80)

    results = {}
    latencies = {}

    for name, doc in docs.items():
        start = time.perf_counter()
        result = pipe.run(doc, queries[name])
        end = time.perf_counter()
        results[name] = result
        latencies[name] = (end - start) * 1000

    # Table header
    header = f"{'Metric':<30} {'Short':<18} {'Medium':<18} {'Large':<18}"
    print(header)
    print("-" * 80)

    # Input Token Count
    row = f"{'Input Tokens':<30} "
    row += f"{results['short'].original_token_count:<18} "
    row += f"{results['medium'].original_token_count:<18} "
    row += f"{results['large'].original_token_count:<18}"
    print(row)

    # Output Token Count
    row = f"{'Output Tokens':<30} "
    row += f"{results['short'].compressed_token_count:<18} "
    row += f"{results['medium'].compressed_token_count:<18} "
    row += f"{results['large'].compressed_token_count:<18}"
    print(row)

    # Tokens Saved
    saved = {n: results[n].original_token_count - results[n].compressed_token_count for n in docs}
    row = f"{'Tokens Saved':<30} "
    row += f"{saved['short']:<18} "
    row += f"{saved['medium']:<18} "
    row += f"{saved['large']:<18}"
    print(row)

    # Token Reduction %
    reduction = {n: (saved[n] / results[n].original_token_count * 100) for n in docs}
    row = f"{'Token Reduction %':<30} "
    row += f"{reduction['short']:<18.1f} "
    row += f"{reduction['medium']:<18.1f} "
    row += f"{reduction['large']:<18.1f}"
    print(row)

    # Compression Ratio
    row = f"{'Compression Ratio':<30} "
    row += f"{results['short'].compression_ratio:<18.1%} "
    row += f"{results['medium'].compression_ratio:<18.1%} "
    row += f"{results['large'].compression_ratio:<18.1%}"
    print(row)

    # Latency
    row = f"{'Latency (ms)':<30} "
    row += f"{latencies['short']:<18.2f} "
    row += f"{latencies['medium']:<18.2f} "
    row += f"{latencies['large']:<18.2f}"
    print(row)

    # Accuracy (keyword retention)
    retention = {n: keyword_retention(docs[n], results[n].compressed_context, queries[n]) for n in docs}
    row = f"{'Keyword Retention %':<30} "
    row += f"{retention['short']:<18.1f} "
    row += f"{retention['medium']:<18.1f} "
    row += f"{retention['large']:<18.1f}"
    print(row)

    # Cost Savings
    print()
    print("-" * 80)
    print("INFERENCE COST SAVINGS (GPT-4 pricing: $0.03/1K input tokens)")
    print("-" * 80)

    for name in docs:
        r = results[name]
        orig_cost = (r.original_token_count / 1000) * INPUT_COST_PER_1K
        comp_cost = (r.compressed_token_count / 1000) * INPUT_COST_PER_1K
        cost_saved = orig_cost - comp_cost
        pct = (cost_saved / orig_cost * 100) if orig_cost > 0 else 0
        print(f"  {name.capitalize():<10} Original: ${orig_cost:.6f}  "
              f"Compressed: ${comp_cost:.6f}  Saved: ${cost_saved:.6f} ({pct:.1f}%)")

    # Scale to 1M requests
    print()
    print("-" * 80)
    print("PROJECTED SAVINGS AT SCALE (1M requests)")
    print("-" * 80)
    for name in docs:
        r = results[name]
        tokens_saved_per_req = r.original_token_count - r.compressed_token_count
        total_saved = tokens_saved_per_req * 1_000_000
        cost_saved = (total_saved / 1000) * INPUT_COST_PER_1K
        print(f"  {name.capitalize():<10} Tokens saved/req: {tokens_saved_per_req:<8} "
              f"Total saved: {total_saved:>12,} tokens  Cost saved: ${cost_saved:,.2f}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    avg_ratio = sum(results[n].compression_ratio for n in docs) / len(docs)
    avg_latency = sum(latencies[n] for n in docs) / len(docs)
    avg_retention = sum(retention[n] for n in docs) / len(docs)
    print(f"  Avg Compression Ratio:    {avg_ratio:.1%}")
    print(f"  Avg Token Reduction:      {sum(reduction.values())/3:.1f}%")
    print(f"  Avg Latency:              {avg_latency:.2f} ms")
    print(f"  Avg Keyword Retention:    {avg_retention:.1f}%")
    print(f"  Retrieval Method:         BM25")
    print(f"  Compression Method:       Truncation (fallback)")
    print(f"  Packing Strategy:         edges_first (position-aware)")


if __name__ == "__main__":
    main()
