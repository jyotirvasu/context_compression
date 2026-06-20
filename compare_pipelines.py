"""
Unified Pipeline Comparison: Selective-Context+PA  vs  LLMLingua
================================================================

Runs BOTH compression pipelines on the SAME dataset samples and reports a
side-by-side comparison, so the two research approaches can be validated
on identical inputs with identical metrics.

The same dataset row feeds both pipelines through two different "lenses":

    HotpotQA row -> question, context (list of paragraphs), answer

    * Selective-Context + Position-Aware  (Phase_1_CC_PA)
        document = "\\n\\n".join(paragraphs)          # flattened blob
        ContextCompressionPipeline.run(document, question)

    * LLMLingua                            (Phase_1_LLMLingua)
        LLMLinguaPipeline.compress(
            contexts=paragraphs,                       # kept as a list
            instruction="Answer the question using the context.",
            question=question,
        )

FAIR-BUDGET ALIGNMENT
---------------------
The two pipelines parametrise compression differently:
    * CC_PA  uses reduce_ratio  = fraction of content REMOVED
    * LLMLingua uses rate       = fraction of content KEPT
This harness drives both from a single "keep ratio" k (fraction to keep):
    CC_PA.reduce_ratio = 1 - k        LLMLingua.rate = k

METRICS (computed identically for both)
---------------------------------------
    - compression achieved (%) : 1 - out_tokens / in_tokens (tiktoken)
    - answer recall (%)        : gold answer present in compressed output
    - keyword retention (%)    : lexical overlap with the source
    - latency (ms)

USAGE  (on a machine with the models available)
-----------------------------------------------
    pip install datasets matplotlib
    python compare_pipelines.py                         # default keep ratios
    python compare_pipelines.py --keep-ratios 0.3 0.5 0.7
    python compare_pipelines.py --num-samples 50
    python compare_pipelines.py --offline               # built-in sample, no internet

RESUME AFTER A CRASH
--------------------
Long runs of the LLMLingua perplexity model can be killed by the OS
(macOS/Apple-Silicon "bus error" / SIGBUS). Every per-sample result is cached
on disk under --cache-dir (default .compare_cache/), so simply re-running the
SAME command replays finished samples instantly and only recomputes the work
lost to the crash. Use --no-cache to force a full recompute.

OUTPUTS  (versioned, like evaluate_hf.py)
-----------------------------------------
    results/comparison/run_<ts>/eval/comparison_results.csv
    results/comparison/run_<ts>/eval/comparison_results.json
    results/comparison/run_<ts>/plot/compression_comparison.png
    results/comparison/run_<ts>/plot/answer_recall_comparison.png
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

# Native-stability guards (must run BEFORE torch / transformers import).
# Prevents the macOS/Apple-Silicon "zsh: bus error" (SIGBUS) and the companion
# "leaked semaphore" warning. Both stem from native thread-pool / allocator
# contention (OpenMP, BLAS, HuggingFace tokenizers) building up over a long
# multi-sample run; the leaked semaphore is just the resource-tracker reporting
# the unclean shutdown after the SIGBUS. Pinning every native pool to a single
# thread removes the contention that triggers the crash.
for _var in (
    "TOKENIZERS_PARALLELISM",  # disable the HF tokenizers Rust thread pool
):
    os.environ.setdefault(_var, "false")
for _var in (
    "OMP_NUM_THREADS",       # OpenMP
    "MKL_NUM_THREADS",       # Intel MKL
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "NUMEXPR_NUM_THREADS",   # numexpr
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate / vecLib
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    import torch

    torch.set_num_threads(1)
    try:
        # Inter-op parallelism must also be pinned; can only be set once, before
        # any parallel work has started, so guard against a late RuntimeError.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError:
    pass

from utils.helpers import count_tokens, load_config

# Reuse helpers/metrics from the single-pipeline evaluator
from evaluate_hf import (
    OFFLINE_SAMPLE,
    answer_retained,
    keyword_retention,
    make_run_dir,
)

DEFAULT_INSTRUCTION = "Answer the question using the context."


# ----------------------------------------------------------------------
# Data loading: return paragraphs (list) so BOTH lenses can be applied
# ----------------------------------------------------------------------
def _hotpot_paragraphs(context) -> List[str]:
    """HotpotQA context = {'title': [...], 'sentences': [[...], ...]} -> list of paras."""
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    paras = []
    for i, title in enumerate(titles):
        sents = sentences[i] if i < len(sentences) else []
        paras.append(f"{title}. " + " ".join(s.strip() for s in sents))
    return [p for p in paras if p.strip()]


def _paragraphs_from_text(text: str) -> List[str]:
    """Best-effort split of a flat string into multiple contexts."""
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) > 1:
            return parts
    # fall back to sentence-ish splitting
    sents = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return sents if len(sents) > 1 else [text]


def load_samples(args) -> List[Dict]:
    """Return list of {question, paragraphs, document, answer}."""
    if args.offline:
        print("[data] Using built-in OFFLINE sample (synthetic, 3 examples).")
        out = []
        for s in OFFLINE_SAMPLE:
            paras = _paragraphs_from_text(s["document"])
            out.append({
                "question": s["question"],
                "paragraphs": paras,
                "document": "\n\n".join(paras),
                "answer": s["answer"],
            })
        return out

    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] 'datasets' not installed (pip install datasets). Using OFFLINE sample.")
        args.offline = True
        return load_samples(args)

    print(f"[data] Loading {args.dataset} ({args.config}/{args.split}) from HuggingFace ...")
    try:
        ds = load_dataset(args.dataset, args.config, split=args.split) if args.config \
            else load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"[data] Failed to load dataset: {e}\n[data] Using OFFLINE sample.")
        args.offline = True
        return load_samples(args)

    n = min(args.num_samples, len(ds))
    samples = []
    for i in range(n):
        row = ds[i]
        ctx = row.get("context")
        if isinstance(ctx, dict):
            paras = _hotpot_paragraphs(ctx)
        else:
            paras = _paragraphs_from_text(str(ctx))
        question = row.get("question", "")
        answer = row.get("answer", "")
        if isinstance(answer, dict):
            answer = (answer.get("value") or "")
        if paras and question:
            samples.append({
                "question": question,
                "paragraphs": paras,
                "document": "\n\n".join(paras),
                "answer": str(answer),
            })
    print(f"[data] Prepared {len(samples)} usable samples (requested {args.num_samples}).")
    return samples


# ----------------------------------------------------------------------
# On-disk resume cache (enables "resume from where it left off")
# ----------------------------------------------------------------------
class SampleCache:
    """Per-sample on-disk cache of compression metrics, keyed by content.

    Each (method, keep_ratio, sample) result is hashed and stored as a small
    JSON file. Because both compression pipelines are deterministic for a fixed
    input, a re-run replays cached samples instantly and only the NEW work
    (e.g. everything after a `bus error` / SIGBUS crash) actually re-runs the
    GPT-2 perplexity model. This gives a practical "resume from where it left
    off" without changing either pipeline.
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.hits = 0
        self.writes = 0
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, method: str, keep: float, sample: Dict) -> str:
        key_src = json.dumps(
            {
                "method": method,
                "keep": round(float(keep), 6),
                "document": sample.get("document", ""),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def get(self, method: str, keep: float, sample: Dict) -> Optional[Dict]:
        if not self.enabled:
            return None
        path = os.path.join(self.cache_dir, self._key(method, keep, sample) + ".json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.hits += 1
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None  # corrupt/partial file: recompute
        return None

    def put(self, method: str, keep: float, sample: Dict, metrics: Dict) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.cache_dir, self._key(method, keep, sample) + ".json")
        tmp = path + ".tmp"
        # Atomic write so a crash mid-write never leaves a corrupt cache entry.
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        os.replace(tmp, path)
        self.writes += 1


# ----------------------------------------------------------------------
# Pipeline runners (each returns a per-sample metric dict or an error)
# ----------------------------------------------------------------------
def run_cc_pa(pipe, sample: Dict, keep: float) -> Dict:
    pipe.compressor.reduce_ratio = 1.0 - keep
    document = sample["document"]
    start = time.perf_counter()
    result = pipe.run(document, sample["question"])
    latency_ms = (time.perf_counter() - start) * 1000
    compressed = result.compressed_context
    in_tok = count_tokens(document)
    out_tok = count_tokens(compressed)
    return _metrics("cc_pa", keep, document, compressed, in_tok, out_tok,
                    sample["answer"], latency_ms)


def run_llmlingua(pipe, sample: Dict, keep: float) -> Dict:
    document = sample["document"]
    start = time.perf_counter()
    result = pipe.compress(
        contexts=sample["paragraphs"],
        instruction=DEFAULT_INSTRUCTION,
        question=sample["question"],
        rate=keep,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    compressed = result.compressed_prompt
    in_tok = count_tokens(document)
    out_tok = count_tokens(compressed)
    return _metrics("llmlingua", keep, document, compressed, in_tok, out_tok,
                    sample["answer"], latency_ms)


def _metrics(method, keep, document, compressed, in_tok, out_tok, answer, latency_ms) -> Dict:
    comp_ratio = (1.0 - out_tok / in_tok) if in_tok else 0.0
    return {
        "method": method,
        "keep_ratio": keep,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "compression_ratio": comp_ratio,
        "keyword_retention": keyword_retention(document, compressed),
        "answer_retained": answer_retained(compressed, answer),
        "latency_ms": latency_ms,
    }


def evaluate_method(method: str, pipe, samples: List[Dict], keep: float,
                    cache: Optional["SampleCache"] = None) -> Dict:
    runner = run_cc_pa if method == "cc_pa" else run_llmlingua
    total = len(samples)
    loop_start = time.perf_counter()
    per_sample = []
    errors = 0
    cached = 0
    for i, s in enumerate(samples, 1):
        try:
            metrics = cache.get(method, keep, s) if cache else None
            if metrics is not None:
                cached += 1
            else:
                metrics = runner(pipe, s, keep)
                if cache:
                    cache.put(method, keep, s, metrics)
            per_sample.append(metrics)
        except Exception as e:
            errors += 1
            if errors <= 2:
                print(f"\n  [{method}] sample {i} failed: {e}")
            continue
        # Periodically release intermediate tensors / fragmented memory to keep
        # the native allocator from growing without bound over a long run (a
        # common trigger for the Apple-Silicon SIGBUS / leaked-semaphore crash).
        if i % 25 == 0:
            gc.collect()
        elapsed = time.perf_counter() - loop_start
        eta = (elapsed / i) * (total - i)
        bar = "#" * int(24 * i / total) + "-" * (24 - int(24 * i / total))
        print(f"\r  {method:<10} keep={keep:.2f} [{bar}] {i}/{total} "
              f"| cached {cached} | elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s",
              end="", flush=True)
    print()
    if cache and cached:
        print(f"  [{method}] resumed {cached}/{total} samples from cache "
              f"({total - cached} newly computed).")

    n = len(per_sample) or 1
    agg = {
        "method": method,
        "keep_ratio": keep,
        "num_samples": len(per_sample),
        "num_errors": errors,
        "avg_input_tokens": sum(p["input_tokens"] for p in per_sample) / n,
        "avg_output_tokens": sum(p["output_tokens"] for p in per_sample) / n,
        "avg_compression_ratio": sum(p["compression_ratio"] for p in per_sample) / n,
        "avg_keyword_retention": sum(p["keyword_retention"] for p in per_sample) / n,
        "answer_recall_pct": sum(1 for p in per_sample if p["answer_retained"]) / n * 100,
        "avg_latency_ms": sum(p["latency_ms"] for p in per_sample) / n,
    }
    return {"aggregate": agg, "per_sample": per_sample}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def print_table(rows: List[Dict]):
    cols = [
        ("method", "Method", "{}"),
        ("keep_ratio", "Keep", "{:.2f}"),
        ("avg_input_tokens", "In Tok", "{:.0f}"),
        ("avg_output_tokens", "Out Tok", "{:.0f}"),
        ("avg_compression_ratio", "Compress", "{:.1%}"),
        ("avg_keyword_retention", "Keyword%", "{:.1f}"),
        ("answer_recall_pct", "AnsRecall%", "{:.1f}"),
        ("avg_latency_ms", "Latency ms", "{:.1f}"),
    ]
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON  (Selective-Context+PA  vs  LLMLingua)")
    print("=" * 100)
    print("".join(f"{label:<13}" for _, label, _ in cols))
    print("-" * 100)
    for r in rows:
        print("".join(f"{fmt.format(r[key]):<13}" for key, _, fmt in cols))
    print("=" * 100)


def make_plots(plot_dir: str, rows: List[Dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed (pip install matplotlib) - skipping plots.")
        return

    def series(method, key):
        pts = sorted([(r["keep_ratio"], r[key]) for r in rows if r["method"] == method])
        return [p[0] for p in pts], [p[1] for p in pts]

    styles = {"cc_pa": ("o-", "#1f4e79", "Selective-Context + PA"),
              "llmlingua": ("s--", "#c0392b", "LLMLingua")}

    # Plot 1: compression achieved
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, (mk, color, label) in styles.items():
        x, y = series(m, "avg_compression_ratio")
        if x:
            ax.plot(x, [v * 100 for v in y], mk, color=color, linewidth=2,
                    markersize=8, label=label)
    ax.set_xlabel("Keep ratio (fraction of tokens retained)")
    ax.set_ylabel("Compression achieved (%)")
    ax.set_title("Compression Achieved: CC+PA vs LLMLingua")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(plot_dir, "compression_comparison.png")
    fig.savefig(out1, dpi=150)
    print(f"[plot] Saved -> {out1}")

    # Plot 2: answer recall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, (mk, color, label) in styles.items():
        x, y = series(m, "answer_recall_pct")
        if x:
            ax.plot(x, y, mk, color=color, linewidth=2, markersize=8, label=label)
    ax.set_xlabel("Keep ratio (fraction of tokens retained)")
    ax.set_ylabel("Answer recall (%)")
    ax.set_title("Answer Recall: CC+PA vs LLMLingua")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out2 = os.path.join(plot_dir, "answer_recall_comparison.png")
    fig.savefig(out2, dpi=150)
    print(f"[plot] Saved -> {out2}")


# ----------------------------------------------------------------------
# Pipeline construction
# ----------------------------------------------------------------------
def build_cc_pa(config_path: str):
    from pipeline import ContextCompressionPipeline
    return ContextCompressionPipeline(config_path)


def build_llmlingua(cc_config: dict):
    from Phase_1_LLMLingua.pipeline import LLMLinguaPipeline
    comp = cc_config.get("compression", {})
    llm_config = {
        "model_name": comp.get("model_type", "gpt2"),
        "device": "cpu",
        "rate": 0.5,
        "context_filter": {"enabled": True, "rank_method": "longllmlingua"},
        "sentence_filter": {"enabled": True},
        "token_compressor": {"enabled": True, "rate": 0.5},
        "recovery": {"concate_question": True, "add_instruction": False},
    }
    return LLMLinguaPipeline(llm_config)


def main():
    parser = argparse.ArgumentParser(description="Compare CC+PA and LLMLingua on the same dataset.")
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa", help="HF dataset id")
    parser.add_argument("--config", default="distractor", help="HF dataset config/subset")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--keep-ratios", type=float, nargs="*", default=[0.3, 0.5, 0.7],
                        help="Fractions of tokens to KEEP (drives both pipelines fairly)")
    parser.add_argument("--methods", nargs="*", default=["cc_pa", "llmlingua"],
                        choices=["cc_pa", "llmlingua"], help="Which pipelines to run")
    parser.add_argument("--offline", action="store_true", help="Use built-in synthetic sample")
    parser.add_argument("--config-path", default="config.yaml", help="CC_PA config.yaml path")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--project", default="comparison", help="Results group name")
    parser.add_argument("--cache-dir", default=".compare_cache",
                        help="Directory for the per-sample resume cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the resume cache (always recompute every sample)")
    args = parser.parse_args()

    print("=" * 100)
    print("UNIFIED PIPELINE COMPARISON")
    print(f"  Methods     : {', '.join(args.methods)}")
    print(f"  Keep ratios : {args.keep_ratios}")
    print("=" * 100)

    samples = load_samples(args)
    if not samples:
        print("[compare] No samples. Exiting.")
        return

    cc_config = load_config(args.config_path)

    # Build the requested pipelines (tolerate build failure per method)
    pipes = {}
    if "cc_pa" in args.methods:
        try:
            pipes["cc_pa"] = build_cc_pa(args.config_path)
        except Exception as e:
            print(f"[compare] Could not build CC+PA pipeline: {e}")
    if "llmlingua" in args.methods:
        try:
            pipes["llmlingua"] = build_llmlingua(cc_config)
        except Exception as e:
            print(f"[compare] Could not build LLMLingua pipeline: {e}")

    if not pipes:
        print("[compare] No pipelines available. Exiting.")
        return

    cache = SampleCache(args.cache_dir, enabled=not args.no_cache)
    if cache.enabled:
        print(f"[compare] Resume cache: {os.path.abspath(args.cache_dir)} "
              f"(re-run to resume after a crash; use --no-cache to disable).")

    all_runs = []
    aggregate_rows = []
    for keep in args.keep_ratios:
        for method, pipe in pipes.items():
            print(f"\n[compare] {method} @ keep={keep} on {len(samples)} samples ...")
            run = evaluate_method(method, pipe, samples, keep, cache=cache)
            all_runs.append(run)
            aggregate_rows.append(run["aggregate"])

    print_table(aggregate_rows)

    # Versioned, project-scoped output
    run_dir = make_run_dir(args.results_dir, args.project)
    eval_dir = os.path.join(run_dir, "eval")
    plot_dir = os.path.join(run_dir, "plot")

    json_path = os.path.join(eval_dir, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    csv_path = os.path.join(eval_dir, "comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    make_plots(plot_dir, aggregate_rows)

    print(f"\n[compare] Run directory      -> {run_dir}")
    print(f"[compare] Saved comparison    -> {json_path}")
    print(f"[compare] Saved summary table -> {csv_path}")


if __name__ == "__main__":
    main()
