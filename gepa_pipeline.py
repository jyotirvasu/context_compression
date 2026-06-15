"""
GEPA Evaluation Pipeline on HotpotQA
====================================

Runs the GEPA prompt optimizer (Phase_2_GEPA) on the SAME source dataset used
by the compression pipelines (HotpotQA), so all three research approaches can
eventually be compared from one dataset.

WHY A SEPARATE PIPELINE?
------------------------
GEPA is a *different category* of method from Selective-Context+PA and
LLMLingua:

    * CC+PA / LLMLingua  -> compress the CONTEXT      (metric: compression % vs answer recall)
    * GEPA               -> optimise the PROMPT        (metric: task accuracy, e.g. EM / F1)

GEPA therefore consumes the dataset differently:

    HotpotQA row -> input  = "Context:\\n<paragraphs>\\n\\nQuestion: <q>"
                    expected = <gold answer>

and it needs a TRAIN/VAL split because the labels are the optimization signal
(they drive reflection, mutation and Pareto selection), not just a metric.

WHAT IT DOES
------------
    1. Loads HotpotQA (reusing the loader/adapters from evaluate_hf.py).
    2. Converts rows into GEPA's {"input", "expected"} format.
    3. Splits them into train (feedback) and val (Pareto) sets.
    4. Defines a seed prompt, an LLM execute function and an F1/EM score function.
    5. Runs GEPAEngine.optimize(...) and reports seed -> best improvement.
    6. Saves results into a versioned folder (like evaluate_hf.py):
         results/Phase_2_GEPA/run_<ts>/eval/gepa_results.json
         results/Phase_2_GEPA/run_<ts>/eval/gepa_results.csv

MODES
-----
    --mock      : no API keys / no internet needed. Uses GEPA's built-in mock
                  evaluator (length-based scoring). Validates the PLUMBING only,
                  NOT real HotpotQA accuracy.
    (default)   : real run. Calls an LLM via litellm/openai to actually answer
                  HotpotQA, so the scores reflect true task accuracy.

USAGE
-----
    # Plumbing smoke-test (no LLM, no internet)
    python gepa_pipeline.py --mock --offline --num-samples 10

    # Real run on HotpotQA (needs `pip install datasets litellm` + an API key)
    export OPENAI_API_KEY=sk-...
    python gepa_pipeline.py --num-samples 50 --model gpt-4o-mini

    # Tune budget / split
    python gepa_pipeline.py --train-size 20 --val-size 20 \\
        --max-iterations 15 --max-metric-calls 120
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Reuse the dataset loader, adapters and run-folder helper from evaluate_hf.py
from evaluate_hf import OFFLINE_SAMPLE, adapt_row, make_run_dir

from Phase_2_GEPA import GEPAEngine, GEPAConfig
from Phase_2_GEPA.adapter import DefaultAdapter
from Phase_2_GEPA.reflector import Reflector
from Phase_2_GEPA.merge import MergeProposer


DEFAULT_SEED_PROMPT = {
    "system_prompt": (
        "You are a precise question-answering assistant. Read the provided "
        "context and answer the question. Respond with the shortest exact "
        "answer span (a word, name, number, or short phrase) and nothing else."
    ),
}


# ----------------------------------------------------------------------
# Metrics: HotpotQA-style normalized Exact Match and token-level F1
# ----------------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD/HotpotQA style)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # If either is empty, F1 is 1 only when both are empty
        return float(pred_tokens == gold_tokens)
    common: Dict[str, int] = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, gold: str) -> float:
    """Combined score used as GEPA's metric: rewards exact match, credits partial overlap."""
    if not gold:
        return 0.0
    em = exact_match(prediction, gold)
    if em == 1.0:
        return 1.0
    return token_f1(prediction, gold)


# ----------------------------------------------------------------------
# Data: HotpotQA row -> GEPA {"input", "expected"} instance
# ----------------------------------------------------------------------
def _row_to_instance(document: str, question: str, answer: str) -> Dict[str, str]:
    return {
        "input": f"Context:\n{document}\n\nQuestion: {question}",
        "expected": answer,
        "question": question,
    }


def load_gepa_samples(args) -> List[Dict[str, str]]:
    """Load HotpotQA and convert into GEPA instances, or use the offline sample."""
    if args.offline:
        print("[data] Using built-in OFFLINE sample (synthetic, 3 examples).")
        return [
            _row_to_instance(s["document"], s["question"], s["answer"])
            for s in OFFLINE_SAMPLE
        ]

    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] 'datasets' not installed. Run: pip install datasets")
        print("[data] Falling back to built-in OFFLINE sample.")
        return [
            _row_to_instance(s["document"], s["question"], s["answer"])
            for s in OFFLINE_SAMPLE
        ]

    print(f"[data] Loading {args.dataset} ({args.config}/{args.split}) from HuggingFace ...")
    try:
        if args.config:
            ds = load_dataset(args.dataset, args.config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"[data] Failed to load dataset: {e}")
        print("[data] Falling back to built-in OFFLINE sample.")
        return [
            _row_to_instance(s["document"], s["question"], s["answer"])
            for s in OFFLINE_SAMPLE
        ]

    n = min(args.num_samples, len(ds))
    samples: List[Dict[str, str]] = []
    for i in range(n):
        adapted = adapt_row(args.dataset, ds[i])
        if adapted is None:
            continue
        document, question, answer = adapted
        if document and question and answer:
            samples.append(_row_to_instance(document, question, answer))
    print(f"[data] Prepared {len(samples)} usable instances (requested {args.num_samples}).")
    return samples


def split_train_val(samples: List[Dict[str, str]], train_size: int,
                    val_size: int) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic train/val split (no shuffle so runs are reproducible)."""
    train = samples[:train_size]
    val = samples[train_size:train_size + val_size]
    if not val:  # tiny datasets: reuse train as val so the loop still runs
        val = train
    return train, val


# ----------------------------------------------------------------------
# LLM backend + on-disk response cache (enables resume-by-replay)
# ----------------------------------------------------------------------
class _CachedResponse:
    """Minimal response shim exposing .choices[0].message.content (for cache hits)."""

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _CachedResponse._Msg(content)

    def __init__(self, content):
        self.choices = [_CachedResponse._Choice(content)]


def resolve_backend():
    """Return (raw_completion, backend_name). Prefers litellm, falls back to openai."""
    try:
        import litellm
        return litellm.completion, "litellm"
    except ImportError:
        pass
    try:
        from openai import OpenAI
        client = OpenAI()

        def completion(model, messages, temperature=0.0, **kwargs):
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
        return completion, "openai"
    except ImportError:
        pass
    raise RuntimeError(
        "No LLM backend found. Install one of:\n"
        "    pip install litellm        (multi-provider)\n"
        "    pip install openai         (OpenAI only)\n"
        "...or run with --mock for a no-LLM plumbing test."
    )


def make_cached_completion(raw_completion, cache_dir: str, enabled: bool = True):
    """Wrap a completion callable with an on-disk cache.

    The cache key is a hash of (model, messages, kwargs). Because GEPA is
    deterministic for a fixed seed, a re-run replays cached calls instantly and
    only the NEW work (e.g. after a crash) actually hits the LLM. This gives a
    practical 'resume from where it left off' without changing the engine.
    """
    if enabled:
        os.makedirs(cache_dir, exist_ok=True)

    def cached(model, messages, **kwargs):
        if not enabled:
            return raw_completion(model=model, messages=messages, **kwargs)
        key_src = json.dumps(
            {"model": model, "messages": messages, "kwargs": kwargs},
            sort_keys=True, default=str,
        )
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return _CachedResponse(json.load(f)["content"])
        resp = raw_completion(model=model, messages=messages, **kwargs)
        content = resp.choices[0].message.content
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f)
        return resp

    return cached


# ----------------------------------------------------------------------
# LLM execution function (real mode)
# ----------------------------------------------------------------------
def build_execute_fn(completion, model: str, max_input_chars: int = 6000,
                     num_ctx: int = 8192, max_retries: int = 1,
                     total_budget: int = 0):
    """Return execute_fn(item, candidate) -> (output, trace) backed by a real LLM.

    `completion` is a callable(model, messages, **kwargs) -> response object
    (typically the cached wrapper from make_cached_completion).

    Robustness:
      * Long contexts are truncated to `max_input_chars` so that local models
        (e.g. Ollama llama3.1) do not crash with "unexpected EOF" / OOM.
      * For Ollama models a `num_ctx` option is passed to widen the context window.
      * Each call is retried up to `max_retries` times; if it still fails the
        error is swallowed and an empty answer (score 0) is returned, so a single
        bad rollout never aborts the whole optimization run.

    Progress:
      * Every LLM call prints a single-line progress indicator showing the
        rollout count, elapsed time, avg latency, and ETA against `total_budget`
        (the metric-call budget). `total_budget=0` disables the ETA estimate.
    """
    is_ollama = model.startswith("ollama/") or model.startswith("ollama_chat/")

    # Mutable progress state shared across calls
    progress = {"calls": 0, "errors": 0, "start": None}

    def _show_progress(failed: bool = False):
        progress["calls"] += 1
        if failed:
            progress["errors"] += 1
        if progress["start"] is None:
            progress["start"] = time.perf_counter()
        done = progress["calls"]
        elapsed = time.perf_counter() - progress["start"]
        avg = elapsed / done if done else 0.0
        if total_budget > 0:
            remaining = max(0, total_budget - done)
            eta = avg * remaining
            bar_len = 24
            filled = min(bar_len, int(bar_len * done / total_budget))
            bar = "#" * filled + "-" * (bar_len - filled)
            tail = f"[{bar}] {done}/{total_budget} (budget)"
        else:
            eta = 0.0
            tail = f"call {done}"
        err = f" | err {progress['errors']}" if progress["errors"] else ""
        print(
            f"\r  [llm] {tail} | {avg:4.1f}s/call "
            f"| elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s{err}",
            end="", flush=True,
        )

    def _truncate(text: str) -> str:
        """Keep the question (tail) and as much context as fits in the budget."""
        if len(text) <= max_input_chars:
            return text
        # The input is "Context:\n...\n\nQuestion: ...". Preserve the question.
        marker = "\n\nQuestion:"
        idx = text.rfind(marker)
        if idx == -1:
            return text[:max_input_chars]
        question_part = text[idx:]
        context_budget = max(0, max_input_chars - len(question_part))
        return text[:context_budget].rstrip() + "\n...[context truncated]..." + question_part

    def execute_fn(item: Dict[str, str], candidate: Dict[str, str]) -> Tuple[str, str]:
        system_prompt = candidate.get("system_prompt", "")
        output_format = candidate.get("output_format", "")
        system_content = (system_prompt + "\n" + output_format).strip()
        user_content = _truncate(item["input"])

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        kwargs: Dict[str, Any] = {"temperature": 0.0}
        if is_ollama:
            # Widen Ollama's context window to reduce "unexpected EOF" crashes.
            kwargs["num_ctx"] = num_ctx

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                response = completion(model=model, messages=messages, **kwargs)
                output = response.choices[0].message.content.strip()
                _show_progress()
                trace = (
                    f"=== Execution Trace ===\n"
                    f"System prompt: {system_content[:200]}\n"
                    f"Question: {item.get('question', '')[:200]}\n"
                    f"Model output: {output[:200]}\n"
                    f"Gold answer: {item.get('expected', '')[:120]}\n"
                )
                return output, trace
            except Exception as e:  # noqa: BLE001 - keep the run alive on any LLM failure
                last_err = e

        # All attempts failed: return an empty answer (scores 0) + diagnostic trace.
        _show_progress(failed=True)
        print(f"\n  [llm] call failed ({type(last_err).__name__}): "
              f"{str(last_err)[:160]} -- scoring 0 and continuing.")
        trace = (
            f"=== Execution Trace (FAILED) ===\n"
            f"System prompt: {system_content[:200]}\n"
            f"Question: {item.get('question', '')[:200]}\n"
            f"Error: {str(last_err)[:300]}\n"
            f"Gold answer: {item.get('expected', '')[:120]}\n"
        )
        return "", trace

    return execute_fn


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run GEPA prompt optimization on HotpotQA")
    # Data
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa",
                        help="HF dataset id (default: hotpotqa/hotpot_qa)")
    parser.add_argument("--config", default="distractor",
                        help="Dataset config (default: distractor)")
    parser.add_argument("--split", default="validation",
                        help="Dataset split (default: validation)")
    parser.add_argument("--num-samples", type=int, default=40,
                        help="Total HotpotQA rows to load (split into train/val)")
    parser.add_argument("--train-size", type=int, default=20,
                        help="Number of instances for the train (feedback) set")
    parser.add_argument("--val-size", type=int, default=20,
                        help="Number of instances for the val (Pareto) set")
    parser.add_argument("--offline", action="store_true",
                        help="Use the built-in synthetic sample (no internet)")
    # GEPA budget
    parser.add_argument("--max-iterations", type=int, default=15,
                        help="Max GEPA optimization iterations")
    parser.add_argument("--max-metric-calls", type=int, default=120,
                        help="Max metric (evaluation) calls budget")
    parser.add_argument("--minibatch-size", type=int, default=5,
                        help="Minibatch size for reflective mutation")
    parser.add_argument("--no-merge", action="store_true",
                        help="Disable the system-aware merge step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # LLM
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM used for answering + reflection (real mode)")
    parser.add_argument("--mock", action="store_true",
                        help="No-LLM plumbing test (GEPA built-in mock evaluator)")
    parser.add_argument("--max-input-chars", type=int, default=6000,
                        help="Truncate each context+question to this many chars "
                             "(prevents local-model 'unexpected EOF' / OOM crashes)")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="Context window passed to Ollama models (default: 8192)")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Retries per failed LLM call before scoring it 0 (default: 1)")
    parser.add_argument("--cache-dir", default=".llm_cache",
                        help="Directory for the on-disk LLM response cache (default: .llm_cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the LLM response cache (disables resume-by-replay)")
    # Output
    parser.add_argument("--results-dir", default="results",
                        help="Root directory for results (default: results)")
    parser.add_argument("--project", default="Phase_2_GEPA",
                        help="Project name under results/ (default: Phase_2_GEPA)")
    args = parser.parse_args()

    print("=" * 92)
    print("GEPA PROMPT OPTIMIZATION - HotpotQA")
    print(f"  Mode        : {'MOCK (no LLM, plumbing only)' if args.mock else 'REAL LLM (' + args.model + ')'}")
    print(f"  Dataset     : {'OFFLINE sample' if args.offline else args.dataset}")
    print(f"  Budget      : {args.max_iterations} iters / {args.max_metric_calls} metric calls")
    print("=" * 92)

    # --- Load + split data ---
    samples = load_gepa_samples(args)
    if not samples:
        print("[error] No samples available; aborting.")
        return
    train_data, val_data = split_train_val(samples, args.train_size, args.val_size)
    print(f"[data] Train: {len(train_data)} | Val: {len(val_data)}")

    # --- Configure GEPA ---
    config = GEPAConfig(
        max_iterations=args.max_iterations,
        max_metric_calls=args.max_metric_calls,
        minibatch_size=args.minibatch_size,
        use_merge=not args.no_merge,
        mock_mode=args.mock,
        seed=args.seed,
        verbose=True,
    )

    # --- Build adapter + reflector ---
    if args.mock:
        adapter = DefaultAdapter(mock_mode=True)
        reflector = Reflector(mock_mode=True)
        merge_proposer = None  # engine builds a mock merge proposer
    else:
        raw_completion, backend = resolve_backend()
        print(f"[llm] Using backend: {backend}  (model={args.model})")
        if not args.no_cache:
            print(f"[llm] Response cache: {args.cache_dir}  "
                  f"(re-run the SAME command to resume from where it left off)")
        completion = make_cached_completion(
            raw_completion, args.cache_dir, enabled=not args.no_cache
        )

        execute_fn = build_execute_fn(
            completion,
            args.model,
            max_input_chars=args.max_input_chars,
            num_ctx=args.num_ctx,
            max_retries=args.max_retries,
            total_budget=args.max_metric_calls,
        )
        adapter = DefaultAdapter(execute_fn=execute_fn, score_fn=score_answer)

        # Reflection AND merge reuse the same cached LLM (so re-runs replay fast).
        is_ollama = args.model.startswith("ollama/") or args.model.startswith("ollama_chat/")

        def reflection_lm(messages):
            extra = {"num_ctx": args.num_ctx} if is_ollama else {}
            return completion(model=args.model, messages=messages,
                              temperature=0.7, **extra)

        reflector = Reflector(lm=reflection_lm, mock_mode=False)
        # IMPORTANT: give the merge proposer an LLM too, else it crashes when
        # the system-aware merge step fires (TypeError: 'NoneType' not callable).
        merge_proposer = MergeProposer(
            lm=reflection_lm,
            max_merge_attempts=config.max_merge_attempts,
            mock_mode=False,
        ) if config.use_merge else None

    engine = GEPAEngine(config, adapter=adapter, reflector=reflector,
                        merge_proposer=merge_proposer)

    # --- Run optimization ---
    start = time.perf_counter()
    result = engine.optimize(
        seed_candidate=DEFAULT_SEED_PROMPT,
        train_data=train_data,
        val_data=val_data,
    )
    elapsed = time.perf_counter() - start

    seed_score = result.state.candidates[0].average_score
    improvement = result.best_score - seed_score

    # --- Report ---
    print("\n" + "=" * 92)
    print("GEPA RESULTS - HotpotQA")
    print("=" * 92)
    print(f"  Seed prompt score   : {seed_score:.4f}")
    print(f"  Best prompt score   : {result.best_score:.4f}")
    print(f"  Improvement         : {improvement:+.4f}")
    print(f"  Iterations          : {result.total_iterations}")
    print(f"  Metric calls        : {result.total_metric_calls}")
    print(f"  Candidates explored : {len(result.state.candidates)}")
    print(f"  Pareto front size   : {len(result.pareto_front)}")
    print(f"  Wall time           : {elapsed:.1f}s")
    print("=" * 92)

    print("\n--- Best Prompt ---")
    for component, text in result.best_candidate.items():
        print(f"\n[{component}]:\n{text}")

    # --- Save versioned results ---
    run_dir = make_run_dir(args.results_dir, args.project)
    eval_dir = os.path.join(run_dir, "eval")

    summary = {
        "mode": "mock" if args.mock else "real",
        "model": None if args.mock else args.model,
        "dataset": "offline" if args.offline else args.dataset,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "seed_score": seed_score,
        "best_score": result.best_score,
        "improvement": improvement,
        "total_iterations": result.total_iterations,
        "total_metric_calls": result.total_metric_calls,
        "candidates_explored": len(result.state.candidates),
        "pareto_front_size": len(result.pareto_front),
        "wall_time_s": elapsed,
        "seed_prompt": DEFAULT_SEED_PROMPT,
        "best_prompt": result.best_candidate,
        "history": result.history,
    }

    json_path = os.path.join(eval_dir, "gepa_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(eval_dir, "gepa_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "score"])
        writer.writerow(["seed", f"{seed_score:.6f}"])
        writer.writerow(["best", f"{result.best_score:.6f}"])

    print(f"\n[gepa] Run directory   -> {run_dir}")
    print(f"[gepa] Saved summary    -> {json_path}")
    print(f"[gepa] Saved CSV        -> {csv_path}")


if __name__ == "__main__":
    main()
