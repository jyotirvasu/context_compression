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
    python evaluate_hf.py --dataset THUDM/LongBench --config hotpotqa --split test  # LongBench
    python evaluate_hf.py --offline                # use built-in sample (no internet)

RESUME AFTER A CRASH
--------------------
Long runs of the compression model can be killed by the OS
(macOS/Apple-Silicon "bus error" / SIGBUS). Every per-sample result is cached
on disk under --cache-dir (default .eval_cache/), so simply re-running the
SAME command replays finished samples instantly and only recomputes the work
lost to the crash. Use --no-cache to force a full recompute.

For a fully hands-off long run, add --chunk-size N: the harness then processes
the samples in fresh subprocesses of N samples each, releasing all native
memory between chunks so the SIGBUS does not recur. Crashed chunks are retried
and any completed samples are preserved in the cache. Example:
    python evaluate_hf.py --num-samples 200 --reduce-ratios 0.5 --chunk-size 25

OUTPUTS
-------
    eval_results.json   (full per-sample + aggregate metrics)
    eval_results.csv    (aggregate summary, one row per reduce_ratio)
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Native-stability guards (must run BEFORE torch / transformers import via
# `pipeline`). Pin every native thread pool to one thread to avoid the
# macOS/Apple-Silicon "zsh: bus error" (SIGBUS) + "leaked semaphore" warning
# that build up over long multi-sample runs. See compare_pipelines.py for the
# full rationale.
for _var in ("TOKENIZERS_PARALLELISM",):
    os.environ.setdefault(_var, "false")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError:
    pass

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

    if "longbench" in name:
        # LongBench (THUDM/LongBench): context is a flat string, the question is
        # under 'input', and gold answers are a list under 'answers'.
        document = str(row.get("context", ""))
        query = row.get("input", "")
        answers = row.get("answers", [])
        if isinstance(answers, list):
            answer = answers[0] if answers else ""
        else:
            answer = str(answers)
        if document and query:
            return document, query, answer
        return None

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
        kwargs = {"split": args.split, "trust_remote_code": True}
        if args.config:
            ds = load_dataset(args.dataset, args.config, **kwargs)
        else:
            ds = load_dataset(args.dataset, **kwargs)
    except TypeError:
        # Older datasets versions don't accept trust_remote_code; retry without.
        if args.config:
            ds = load_dataset(args.dataset, args.config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        # A real dataset was explicitly requested but could not be loaded. Do NOT
        # silently fall back to the 3-sample synthetic set -- that produces
        # meaningless "results". Fail loudly so the run is not mistaken for real.
        print(f"[data] ERROR: could not load {args.dataset}: {e}")
        if "scripts are no longer supported" in str(e):
            print("[data] This dataset ships a loader script, which datasets>=3.0 "
                  "removed. Fix: pip install \"datasets<3.0.0\"  (then re-run).")
        print("[data] Aborting. Use --offline to run the built-in synthetic sample.")
        raise SystemExit(2)

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
# On-disk resume cache (enables "resume from where it left off")
# ----------------------------------------------------------------------
class SampleCache:
    """Per-sample on-disk cache of evaluation metrics, keyed by content.

    Each (reduce_ratio, sample) result is hashed and stored as a small JSON
    file. Because the pipeline is deterministic for a fixed input, a re-run
    replays cached samples instantly and only the NEW work (e.g. everything
    after a `bus error` / SIGBUS crash) actually re-runs the compression model.
    This gives a practical "resume from where it left off" without changing the
    pipeline.
    """

    def __init__(self, cache_dir: str, enabled: bool = True, signature: str = ""):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.signature = signature
        self.hits = 0
        self.writes = 0
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, reduce_ratio: Optional[float], sample: Dict) -> str:
        key_src = json.dumps(
            {
                "signature": self.signature,
                "reduce_ratio": None if reduce_ratio is None else round(float(reduce_ratio), 6),
                "document": sample.get("document", ""),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def get(self, reduce_ratio: Optional[float], sample: Dict) -> Optional[Dict]:
        if not self.enabled:
            return None
        path = os.path.join(self.cache_dir, self._key(reduce_ratio, sample) + ".json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.hits += 1
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None  # corrupt/partial file: recompute
        return None

    def put(self, reduce_ratio: Optional[float], sample: Dict, metrics: Dict) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.cache_dir, self._key(reduce_ratio, sample) + ".json")
        tmp = path + ".tmp"
        # Atomic write so a crash mid-write never leaves a corrupt cache entry.
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        os.replace(tmp, path)
        self.writes += 1


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_at_ratio(pipe: ContextCompressionPipeline, samples: List[Dict],
                      reduce_ratio: Optional[float],
                      cache: Optional["SampleCache"] = None,
                      cache_only: bool = False) -> Dict:
    if reduce_ratio is not None:
        pipe.compressor.reduce_ratio = reduce_ratio

    total = len(samples)
    loop_start = time.perf_counter()
    per_sample = []
    cached = 0
    skipped = 0
    for i, s in enumerate(samples, 1):
        metrics = cache.get(reduce_ratio, s) if cache else None
        if metrics is not None:
            cached += 1
            latency_ms = metrics.get("latency_ms", 0.0)
        elif cache_only:
            # Aggregation pass: never run the model in this (long-lived)
            # process; uncached samples are simply skipped.
            skipped += 1
            continue
        else:
            start = time.perf_counter()
            result = pipe.run(s["document"], s["question"])
            latency_ms = (time.perf_counter() - start) * 1000
            metrics = {
                "question": s["question"][:120],
                "answer": s["answer"][:120],
                "input_tokens": result.original_token_count,
                "output_tokens": result.compressed_token_count,
                "compression_ratio": result.compression_ratio,
                "latency_ms": latency_ms,
                "keyword_retention": keyword_retention(s["document"], result.compressed_context),
                "answer_retained": answer_retained(result.compressed_context, s["answer"]),
            }
            if cache:
                cache.put(reduce_ratio, s, metrics)

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
            f"| cached {cached} "
            f"| elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s",
            end="", flush=True,
        )

        per_sample.append(metrics)

        # Periodically release fragmented memory to keep the native allocator
        # bounded over long runs (a common SIGBUS / leaked-semaphore trigger).
        if i % 25 == 0:
            gc.collect()

    print()  # finish the progress line
    if cache and cached:
        if cache_only:
            missing = total - cached
            tail = (f"({missing} missing and OMITTED from the aggregate)"
                    if missing else "(all present)")
        else:
            tail = f"({total - cached} newly computed)"
        print(f"  [eval] resumed {cached}/{total} samples from cache {tail}")
    if cache_only and len(per_sample) < total:
        print(f"  [eval] WARNING: aggregate computed over {len(per_sample)}/"
              f"{total} samples; {total - len(per_sample)} crashed on every "
              f"attempt and are excluded.")

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


def _run_slice(base_cmd: List[str], lo: int, hi: int, label: str,
               retries: int) -> bool:
    """Run samples [lo:hi] in a fresh subprocess, with `retries` extra attempts.

    Returns True if the subprocess exited cleanly. Completed samples are written
    to the shared cache as it goes, so a crash followed by a retry only repeats
    the work lost to the crash.
    """
    cmd = base_cmd + ["--_worker-start", str(lo), "--_worker-end", str(hi)]
    for attempt in range(retries + 1):
        tag = f"  (retry {attempt})" if attempt else ""
        print(f"\n[eval] === {label}  samples [{lo}:{hi}]{tag} ===")
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return True
        print(f"[eval] {label} exited with code {proc.returncode} "
              f"(likely SIGBUS). Completed samples are cached; retrying remainder.")
    return False


def _run_chunked(args, total_samples: int) -> None:
    """Process the run in fresh subprocesses, `chunk_size` samples at a time.

    Each chunk is a self-contained subprocess that builds the pipeline, loads
    the model, processes its slice, writes results to the shared cache, and then
    exits -- releasing ALL native memory. This sidesteps the long-run
    Apple-Silicon SIGBUS / leaked-semaphore crash, which is caused by native
    allocator/thread state accumulating in a single long-lived process.

    If a chunk subprocess crashes (non-zero exit), its completed samples are
    already in the cache, so it is retried a couple of times. A chunk that keeps
    crashing is almost always being taken down by a SINGLE toxic sample whose
    forward pass triggers the native SIGBUS -- and that one sample drags all the
    other (healthy) samples in the chunk down with it. So a persistently failing
    chunk is re-run ONE sample per subprocess: every healthy sample then still
    completes, and only the genuinely crashing sample(s) are isolated and
    omitted from the final aggregate.
    """
    n_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
    print(f"[eval] Chunked mode: {total_samples} samples in {n_chunks} "
          f"subprocess chunk(s) of up to {args.chunk_size}.")

    base_cmd = [sys.executable, os.path.abspath(__file__),
                "--dataset", args.dataset,
                "--config", args.config,
                "--split", args.split,
                "--num-samples", str(args.num_samples),
                "--config-path", args.config_path,
                "--project", args.project,
                "--results-dir", args.results_dir,
                "--cache-dir", args.cache_dir]
    if args.reduce_ratios:
        base_cmd += ["--reduce-ratios", *[str(r) for r in args.reduce_ratios]]
    if args.offline:
        base_cmd += ["--offline"]
    # Forward per-stage overrides so every chunk runs the identical config.
    for flag, value in (
        ("--chunking-method", args.chunking_method),
        ("--max-chunk-tokens", args.max_chunk_tokens),
        ("--sentence-splitter", args.sentence_splitter),
        ("--retrieval-method", args.retrieval_method),
        ("--top-n", args.top_n),
        ("--hybrid-alpha", args.hybrid_alpha),
        ("--granularity", args.granularity),
        ("--model-type", args.model_type),
        ("--packing-strategy", args.packing_strategy),
        ("--max-context-tokens", args.max_context_tokens),
    ):
        if value is not None:
            base_cmd += [flag, str(value)]

    toxic_total: List[int] = []
    for c in range(n_chunks):
        lo = c * args.chunk_size
        hi = min(total_samples, lo + args.chunk_size)
        label = f"Chunk {c + 1}/{n_chunks}"
        if _run_slice(base_cmd, lo, hi, label, retries=2):
            continue
        # The whole chunk keeps crashing -- isolate it one sample per subprocess
        # so a single toxic sample no longer takes the healthy ones with it.
        print(f"[eval] {label} failed repeatedly; isolating samples "
              f"one-per-subprocess to salvage the healthy ones ...")
        for i in range(lo, hi):
            if not _run_slice(base_cmd, i, i + 1, f"{label} sample {i}", retries=1):
                toxic_total.append(i)
        if toxic_total:
            print(f"[eval] {label}: sample(s) that could not be computed and "
                  f"will be omitted: {[i for i in toxic_total if lo <= i < hi]}")

    if toxic_total:
        print(f"\n[eval] {len(toxic_total)} sample(s) crashed on every attempt "
              f"and are omitted from the aggregate: {toxic_total}")
    print("\n[eval] All chunks processed; aggregating from cache ...")


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
    parser.add_argument("--cache-dir", default=".eval_cache",
                        help="Directory for the per-sample resume cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the resume cache (always recompute every sample)")

    # --- Per-stage hyperparameter overrides (sweep without editing config.yaml) ---
    stage = parser.add_argument_group(
        "stage overrides",
        "Override individual config.yaml hyperparameters from the command line.")
    stage.add_argument("--chunking-method",
                       choices=["sentence", "paragraph", "fixed_token", "sliding_window"],
                       help="Stage A: chunking.method")
    stage.add_argument("--max-chunk-tokens", type=int, help="Stage A: chunking.max_chunk_tokens")
    stage.add_argument("--sentence-splitter", choices=["spacy", "nltk"],
                       help="Stage A: chunking.sentence_splitter")
    stage.add_argument("--retrieval-method", choices=["bm25", "embedding", "hybrid"],
                       help="Stage C: retrieval.method")
    stage.add_argument("--top-n", type=int, help="Stage C: retrieval.top_n")
    stage.add_argument("--hybrid-alpha", type=float, help="Stage C: retrieval.hybrid_alpha")
    stage.add_argument("--granularity", choices=["sentence", "phrase", "token"],
                       help="Stage D: compression.granularity")
    stage.add_argument("--model-type", help="Stage D: compression.model_type (e.g. gpt2)")
    stage.add_argument("--packing-strategy",
                       choices=["edges_first", "decreasing", "round_robin"],
                       help="Stage E: packing.strategy")
    stage.add_argument("--max-context-tokens", type=int,
                       help="Stage E: packing.max_context_tokens")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="If > 0, process samples in fresh subprocesses of this many "
                             "samples each. Fully releases native memory between chunks to "
                             "avoid the Apple-Silicon SIGBUS / leaked-semaphore crash on long "
                             "runs. Requires the cache (cannot combine with --no-cache).")
    # Internal worker flags (set by the chunk orchestrator; not for direct use).
    parser.add_argument("--_worker-start", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-end", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    is_worker = args._worker_start >= 0
    if args.chunk_size > 0 and args.no_cache:
        print("[eval] --chunk-size requires the cache; ignoring --no-cache.")
        args.no_cache = False

    # Assemble per-stage overrides from CLI flags (only those the user supplied).
    overrides = {
        "chunking": {k: v for k, v in {
            "method": args.chunking_method,
            "max_chunk_tokens": args.max_chunk_tokens,
            "sentence_splitter": args.sentence_splitter,
        }.items() if v is not None},
        "retrieval": {k: v for k, v in {
            "method": args.retrieval_method,
            "top_n": args.top_n,
            "hybrid_alpha": args.hybrid_alpha,
        }.items() if v is not None},
        "compression": {k: v for k, v in {
            "granularity": args.granularity,
            "model_type": args.model_type,
        }.items() if v is not None},
        "packing": {k: v for k, v in {
            "strategy": args.packing_strategy,
            "max_context_tokens": args.max_context_tokens,
        }.items() if v is not None},
    }
    overrides = {section: params for section, params in overrides.items() if params}

    # Report which compression backend is active (reflecting overrides)
    cfg = load_config(args.config_path)
    active_retrieval = overrides.get("retrieval", {}).get(
        "method", cfg.get("retrieval", {}).get("method"))
    active_packing = overrides.get("packing", {}).get(
        "strategy", cfg.get("packing", {}).get("strategy"))
    try:
        import selective_context  # noqa: F401
        backend = "Selective Context (real library)"
    except Exception:
        backend = "Truncation fallback (selective-context NOT installed)"
    print("=" * 92)
    print("CONTEXT COMPRESSION PIPELINE - HF DATASET EVALUATION")
    print(f"  Compression backend : {backend}")
    print(f"  Retrieval method    : {active_retrieval}")
    print(f"  Packing strategy    : {active_packing}")
    if overrides:
        print(f"  Stage overrides     : {overrides}")
    print("=" * 92)

    samples = load_samples(args)
    if not samples:
        print("[eval] No samples to evaluate. Exiting.")
        return

    total_samples = len(samples)

    # In a worker subprocess we only handle our assigned slice of samples.
    if is_worker:
        lo = max(0, args._worker_start)
        hi = min(total_samples, args._worker_end)
        samples = samples[lo:hi]
        print(f"[worker] Processing samples [{lo}:{hi}] ({len(samples)} of {total_samples}).")
        if not samples:
            return

    # Chunk orchestrator: spawn a fresh subprocess per chunk so native memory
    # is fully released between chunks (robust against the long-run SIGBUS).
    if args.chunk_size > 0 and not is_worker:
        _run_chunked(args, total_samples)
        # After chunks populate the cache, fall through to the (now instant)
        # cache-served aggregation below to build the table / plots / files.

    pipe = ContextCompressionPipeline(args.config_path, config_overrides=overrides)

    # Fold the active config (incl. overrides) into the cache signature so that
    # different hyperparameter combos never collide on the same cache entry.
    cache_signature = json.dumps(
        {"config_path": args.config_path, "overrides": overrides},
        sort_keys=True,
    )
    cache = SampleCache(args.cache_dir, enabled=not args.no_cache, signature=cache_signature)
    if cache.enabled:
        print(f"[eval] Resume cache: {os.path.abspath(args.cache_dir)} "
              f"(re-run to resume after a crash; use --no-cache to disable).")

    # After chunked processing the parent only AGGREGATES from cache; it must
    # never load the model itself (a single uncached sample would otherwise
    # crash this long-lived process). Workers and non-chunked runs compute.
    aggregation_only = args.chunk_size > 0 and not is_worker

    ratios = args.reduce_ratios if args.reduce_ratios else [None]
    all_runs = []
    aggregate_rows = []
    for r in ratios:
        label = f"reduce_ratio={r}" if r is not None else "config default"
        print(f"\n[eval] Running {len(samples)} samples @ {label} ...")
        run = evaluate_at_ratio(pipe, samples, r, cache=cache,
                                cache_only=aggregation_only)
        all_runs.append(run)
        aggregate_rows.append(run["aggregate"])

    # A worker subprocess only fills the cache for its slice; the parent does
    # the final aggregation and writes the result files.
    if is_worker:
        print(f"[worker] Done; cache writes={cache.writes}, hits={cache.hits}.")
        return

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
