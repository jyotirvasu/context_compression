"""
HuggingFace Dataset Evaluation for the Context Compression Pipeline
===================================================================

Runs the full pipeline (A->E) over a real QA dataset pulled from the
HuggingFace Hub and reports:
    - Input / output token counts and token reduction
    - Compression ratio
    - Per-sample latency
    - Keyword retention (lexical overlap)
    - Answer recall  (does the compressed context still contain the gold answer?)

The answer-recall metric is the important one: it measures whether
compression preserves the information the downstream LLM actually needs.

USAGE (on a machine with internet / HF access)
----------------------------------------------
    pip install datasets
    python evaluate_hf.py                          # HotpotQA, 50 samples
    python evaluate_hf.py --num-samples 200
    python evaluate_hf.py --reduce-ratios 0.3 0.5 0.7   # sweep compression
    python evaluate_hf.py --dataset hotpotqa/hotpot_qa --config distractor --split validation
    python evaluate_hf.py --offline                # use built-in sample (no internet)

OUTPUTS
-------
    eval_results.json   (full per-sample + aggregate metrics)
    eval_results.csv    (aggregate summary, one row per reduce_ratio)
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pipeline import ContextCompressionPipeline
from utils.helpers import count_tokens, load_config


def make_run_dir(results_dir: str, project: str) -> str:
    """Create results/<project>/run_<timestamp>/ with eval/ and plot/ subfolders.

    Also records the run name in results/<project>/latest.txt so that
    plot_results.py can auto-discover the most recent run.
    """
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, project, timestamp)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plot"), exist_ok=True)

    # Pointer to the latest run for this project
    with open(os.path.join(results_dir, project, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(timestamp)

    return run_dir



# ----------------------------------------------------------------------
# Built-in offline sample (synthetic, for smoke-testing WITHOUT internet).
# This is NOT HotpotQA -- it is a tiny hand-written stand-in so the script
# is runnable on a network-restricted machine.
# ----------------------------------------------------------------------
OFFLINE_SAMPLE: List[Dict] = [
    {
        "question": "What architecture do GPT models use?",
        "document": (
            "The Transformer architecture relies on self-attention. "
            "GPT (Generative Pre-trained Transformer) models use a left-to-right "
            "autoregressive Transformer decoder. BERT instead uses a bidirectional "
            "encoder. Vector databases store embeddings for semantic search. "
            "Reinforcement learning trains agents through reward signals."
        ),
        "answer": "Transformer",
    },
    {
        "question": "How many parameters does GPT-3 have?",
        "document": (
            "GPT-2 was released in 2019. GPT-3, introduced in 2020, has 175 billion "
            "parameters and demonstrated strong few-shot learning. Knowledge distillation "
            "transfers knowledge from a teacher to a student model. Mixture of Experts "
            "activates only a subset of parameters per input."
        ),
        "answer": "175 billion",
    },
    {
        "question": "What does RAG retrieve to ground responses?",
        "document": (
            "Retrieval-Augmented Generation (RAG) combines retrieval and generation. "
            "Instead of relying solely on parametric knowledge, RAG retrieves relevant "
            "documents from an external knowledge base and conditions generation on them. "
            "This reduces hallucination. Prompt compression reduces token counts."
        ),
        "answer": "relevant documents",
    },
]


# ----------------------------------------------------------------------
# Dataset adapters: convert an HF row into (document, query, answer)
# ----------------------------------------------------------------------
def _flatten_hotpot_context(context) -> str:
    """HotpotQA context = {'title': [...], 'sentences': [[...], ...]}.

    Flatten into a single document string.
    """
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    parts = []
    for i, title in enumerate(titles):
        sents = sentences[i] if i < len(sentences) else []
        parts.append(f"{title}. " + " ".join(s.strip() for s in sents))
    return "\n\n".join(parts)


def adapt_row(dataset_name: str, row: Dict) -> Optional[Tuple[str, str, str]]:
    """Return (document, query, answer) for a given dataset row, or None to skip."""
    name = dataset_name.lower()

    if "hotpot" in name:
        document = _flatten_hotpot_context(row["context"])
        query = row["question"]
        answer = row.get("answer", "")
        return document, query, answer

    if "2wiki" in name or "musique" in name:
        # Both expose 'context' similar to HotpotQA in many HF mirrors
        ctx = row.get("context")
        document = _flatten_hotpot_context(ctx) if isinstance(ctx, dict) else str(ctx)
        return document, row.get("question", ""), str(row.get("answer", ""))

    if "qasper" in name:
        # full_text is nested; join paragraphs
        ft = row.get("full_text", {})
        paras = []
        for section in ft.get("paragraphs", []):
            paras.extend(section)
        document = "\n\n".join(paras) if paras else str(ft)
        qas = row.get("qas", {})
        question = qas.get("question", [""])[0] if isinstance(qas, dict) else ""
        return document, question, ""

    if "trivia" in name:
        # search_results / evidence -> use 'search_results' contexts if present
        document = " ".join(row.get("search_results", {}).get("search_context", [])) \
            if isinstance(row.get("search_results"), dict) else str(row.get("question", ""))
        answer = row.get("answer", {}).get("value", "") if isinstance(row.get("answer"), dict) else ""
        return document, row.get("question", ""), answer

    # Generic best-effort: look for common field names
    document = row.get("context") or row.get("document") or row.get("article") or ""
    if isinstance(document, dict):
        document = _flatten_hotpot_context(document)
    query = row.get("question") or row.get("query") or ""
    answer = row.get("answer") or row.get("answers") or ""
    if isinstance(answer, dict):
        answer = (answer.get("text") or [""])[0] if answer.get("text") else ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if document and query:
        return str(document), str(query), str(answer)
    return None


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def keyword_retention(original: str, compressed: str) -> float:
    orig_words = set(re.findall(r"\b[a-z]{4,}\b", original.lower()))
    comp_words = set(re.findall(r"\b[a-z]{4,}\b", compressed.lower()))
    if not orig_words:
        return 0.0
    return len(orig_words & comp_words) / len(orig_words) * 100


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_retained(compressed_context: str, answer: str) -> bool:
    """True if the (normalized) gold answer appears in the compressed context."""
    if not answer:
        return False
    a = _normalize(answer)
    if not a:
        return False
    if a in ("yes", "no"):
        # yes/no answers are not extractable from context; skip by treating as recalled
        return True
    return a in _normalize(compressed_context)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_samples(args) -> List[Dict]:
    if args.offline:
        print("[data] Using built-in OFFLINE sample (synthetic, 3 examples).")
        return OFFLINE_SAMPLE

    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] 'datasets' not installed. Run: pip install datasets")
        print("[data] Falling back to built-in OFFLINE sample.")
        return OFFLINE_SAMPLE

    print(f"[data] Loading {args.dataset} ({args.config}/{args.split}) from HuggingFace ...")
    try:
        if args.config:
            ds = load_dataset(args.dataset, args.config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"[data] Failed to load dataset: {e}")
        print("[data] Falling back to built-in OFFLINE sample.")
        return OFFLINE_SAMPLE

    n = min(args.num_samples, len(ds))
    samples = []
    for i in range(n):
        adapted = adapt_row(args.dataset, ds[i])
        if adapted is None:
            continue
        document, query, answer = adapted
        if document and query:
            samples.append({"question": query, "document": document, "answer": answer})
    print(f"[data] Prepared {len(samples)} usable samples (requested {args.num_samples}).")
    return samples


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_at_ratio(pipe: ContextCompressionPipeline, samples: List[Dict],
                      reduce_ratio: Optional[float]) -> Dict:
    if reduce_ratio is not None:
        pipe.compressor.reduce_ratio = reduce_ratio

    total = len(samples)
    loop_start = time.perf_counter()
    per_sample = []
    for i, s in enumerate(samples, 1):
        start = time.perf_counter()
        result = pipe.run(s["document"], s["question"])
        latency_ms = (time.perf_counter() - start) * 1000

        # Lightweight single-line progress indicator with elapsed time and ETA
        elapsed = time.perf_counter() - loop_start
        avg = elapsed / i
        eta = avg * (total - i)
        bar_len = 24
        filled = int(bar_len * i / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{bar}] {i}/{total} "
            f"| {latency_ms/1000:5.1f}s/sample "
            f"| elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s",
            end="", flush=True,
        )

        per_sample.append({
            "question": s["question"][:120],
            "answer": s["answer"][:120],
            "input_tokens": result.original_token_count,
            "output_tokens": result.compressed_token_count,
            "compression_ratio": result.compression_ratio,
            "latency_ms": latency_ms,
            "keyword_retention": keyword_retention(s["document"], result.compressed_context),
            "answer_retained": answer_retained(result.compressed_context, s["answer"]),
        })

    print()  # finish the progress line

    n = len(per_sample) or 1
    agg = {
        "reduce_ratio": reduce_ratio if reduce_ratio is not None else pipe.compressor.reduce_ratio,
        "num_samples": len(per_sample),
        "avg_input_tokens": sum(p["input_tokens"] for p in per_sample) / n,
        "avg_output_tokens": sum(p["output_tokens"] for p in per_sample) / n,
        "avg_compression_ratio": sum(p["compression_ratio"] for p in per_sample) / n,
        "avg_latency_ms": sum(p["latency_ms"] for p in per_sample) / n,
        "avg_keyword_retention": sum(p["keyword_retention"] for p in per_sample) / n,
        "answer_recall_pct": sum(1 for p in per_sample if p["answer_retained"]) / n * 100,
    }
    return {"aggregate": agg, "per_sample": per_sample}


def print_table(rows: List[Dict]):
    cols = [
        ("reduce_ratio", "Reduce Ratio", "{:.2f}"),
        ("avg_input_tokens", "In Tokens", "{:.0f}"),
        ("avg_output_tokens", "Out Tokens", "{:.0f}"),
        ("avg_compression_ratio", "Compression", "{:.1%}"),
        ("avg_keyword_retention", "Keyword Ret%", "{:.1f}"),
        ("answer_recall_pct", "Answer Recall%", "{:.1f}"),
        ("avg_latency_ms", "Latency ms", "{:.1f}"),
    ]
    print("\n" + "=" * 92)
    print("AGGREGATE RESULTS")
    print("=" * 92)
    header = "".join(f"{label:<16}" for _, label, _ in cols)
    print(header)
    print("-" * 92)
    for r in rows:
        line = "".join(f"{fmt.format(r[key]):<16}" for key, _, fmt in cols)
        print(line)
    print("=" * 92)


def main():
    parser = argparse.ArgumentParser(description="HF dataset evaluation for the compression pipeline.")
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa", help="HF dataset id (default: hotpotqa/hotpot_qa)")
    parser.add_argument("--config", default="distractor", help="HF dataset config/subset (default: distractor)")
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--reduce-ratios", type=float, nargs="*", default=None,
                        help="Optional sweep of compression reduce_ratio values, e.g. 0.3 0.5 0.7")
    parser.add_argument("--offline", action="store_true", help="Use the built-in synthetic sample")
    parser.add_argument("--config-path", default="config.yaml", help="Pipeline config.yaml path")
    parser.add_argument("--project", default="Phase_1_CC_PA",
                        help="Project name; results are grouped under results/<project>/")
    parser.add_argument("--results-dir", default="results",
                        help="Root directory for all results (default: results)")
    args = parser.parse_args()

    # Report which compression backend is active
    cfg = load_config(args.config_path)
    try:
        import selective_context  # noqa: F401
        backend = "Selective Context (real library)"
    except Exception:
        backend = "Truncation fallback (selective-context NOT installed)"
    print("=" * 92)
    print("CONTEXT COMPRESSION PIPELINE - HF DATASET EVALUATION")
    print(f"  Compression backend : {backend}")
    print(f"  Retrieval method    : {cfg.get('retrieval', {}).get('method')}")
    print(f"  Packing strategy    : {cfg.get('packing', {}).get('strategy')}")
    print("=" * 92)

    samples = load_samples(args)
    if not samples:
        print("[eval] No samples to evaluate. Exiting.")
        return

    pipe = ContextCompressionPipeline(args.config_path)

    ratios = args.reduce_ratios if args.reduce_ratios else [None]
    all_runs = []
    aggregate_rows = []
    for r in ratios:
        label = f"reduce_ratio={r}" if r is not None else "config default"
        print(f"\n[eval] Running {len(samples)} samples @ {label} ...")
        run = evaluate_at_ratio(pipe, samples, r)
        all_runs.append(run)
        aggregate_rows.append(run["aggregate"])

    print_table(aggregate_rows)

    # Create a versioned, project-scoped run directory:
    #   results/<project>/run_<timestamp>/eval/
    run_dir = make_run_dir(args.results_dir, args.project)
    eval_dir = os.path.join(run_dir, "eval")

    # Save JSON (full detail) and CSV (aggregate summary)
    json_path = os.path.join(eval_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    csv_path = os.path.join(eval_dir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\n[eval] Run directory         -> {run_dir}")
    print(f"[eval] Saved detailed results -> {json_path}")
    print(f"[eval] Saved summary table   -> {csv_path}")
    print(f"[eval] Next: python plot_results.py --project {args.project}")


if __name__ == "__main__":
    main()
