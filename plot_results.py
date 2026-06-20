"""
Plot Evaluation Results
=======================

Single plotting entry point for BOTH result types:

  * Single-pipeline eval   (evaluate_hf.py -> eval_results.csv)
        1. compression_achieved.png   - Compression (%) vs reduce_ratio
        2. information_preserved.png   - Answer recall & keyword retention

  * Pipeline comparison    (compare_pipelines.py -> comparison_results.csv)
        1. compression_comparison.png    - CC+PA vs LLMLingua compression
        2. answer_recall_comparison.png  - CC+PA vs LLMLingua answer recall

The script auto-detects which CSV is present (or which columns it has) and
renders the matching charts into the run's plot/ folder.

By default it auto-discovers the LATEST run for a project under
results/<project>/ (via results/<project>/latest.txt).

USAGE
-----
    pip install matplotlib
    python plot_results.py                                  # latest Phase_1_CC_PA run
    python plot_results.py --project Phase_1_CC_PA
    python plot_results.py --project comparison             # plot a comparison run
    python plot_results.py --run run_20260613_101500        # a specific run
    python plot_results.py --input path/to/results.csv      # explicit CSV

Generate the CSV first with:
    python evaluate_hf.py --reduce-ratios 0.3 0.5 0.7
    python compare_pipelines.py --keep-ratios 0.3 0.5 0.7
"""

import argparse
import csv
import os


def resolve_run_dir(results_dir: str, project: str, run: str = None) -> str:
    """Return the run directory to use for a project.

    If `run` is given, use it. Otherwise read results/<project>/latest.txt,
    falling back to the most recently modified run_* folder.
    """
    project_dir = os.path.join(results_dir, project)
    if run:
        return os.path.join(project_dir, run)

    latest_file = os.path.join(project_dir, "latest.txt")
    if os.path.isfile(latest_file):
        with open(latest_file, "r", encoding="utf-8") as f:
            name = f.read().strip()
        candidate = os.path.join(project_dir, name)
        if os.path.isdir(candidate):
            return candidate

    # Fallback: newest run_* directory by modification time
    if os.path.isdir(project_dir):
        runs = [d for d in os.listdir(project_dir)
                if d.startswith("run_") and os.path.isdir(os.path.join(project_dir, d))]
        if runs:
            runs.sort(key=lambda d: os.path.getmtime(os.path.join(project_dir, d)))
            return os.path.join(project_dir, runs[-1])
    return None


def load_csv(path):
    """Load a CSV; numeric fields are floats, non-numeric (e.g. method) stay str."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def find_csv(run_dir: str):
    """Locate the results CSV in a run folder; returns (csv_path, kind)."""
    eval_dir = os.path.join(run_dir, "eval")
    comparison = os.path.join(eval_dir, "comparison_results.csv")
    single = os.path.join(eval_dir, "eval_results.csv")
    if os.path.isfile(comparison):
        return comparison, "comparison"
    if os.path.isfile(single):
        return single, "single"
    return None, None


def detect_kind(rows):
    """Infer result kind from the columns present."""
    if rows and "method" in rows[0] and "keep_ratio" in rows[0]:
        return "comparison"
    return "single"


# ----------------------------------------------------------------------
# Plot renderers
# ----------------------------------------------------------------------
def plot_single(plt, rows, plot_dir):
    rows = sorted(rows, key=lambda r: r["reduce_ratio"])
    reduce_ratios = [r["reduce_ratio"] for r in rows]
    compression = [r["avg_compression_ratio"] * 100 for r in rows]
    answer_recall = [r["answer_recall_pct"] for r in rows]
    keyword_ret = [r["avg_keyword_retention"] for r in rows]
    note = "  (single point — run a --reduce-ratios sweep for a curve)" if len(rows) == 1 else ""

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(reduce_ratios, compression, "o-", color="#1f4e79", linewidth=2, markersize=8)
    ax1.set_xlabel("Configured reduce_ratio")
    ax1.set_ylabel("Compression achieved (%)")
    ax1.set_title("Compression Achieved vs reduce_ratio" + note)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(reduce_ratios, compression):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    fig1.tight_layout()
    out1 = os.path.join(plot_dir, "compression_achieved.png")
    fig1.savefig(out1, dpi=150)
    print(f"Saved plot -> {out1}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(reduce_ratios, answer_recall, "s-", color="#c0392b",
             linewidth=2, markersize=8, label="Answer recall %")
    ax2.plot(reduce_ratios, keyword_ret, "^--", color="#27ae60",
             linewidth=2, markersize=8, label="Keyword retention %")
    ax2.set_xlabel("Configured reduce_ratio")
    ax2.set_ylabel("Information preserved (%)")
    ax2.set_title("Information Preserved vs reduce_ratio" + note)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    out2 = os.path.join(plot_dir, "information_preserved.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved plot -> {out2}")


def plot_comparison(plt, rows, plot_dir):
    styles = {"cc_pa": ("o-", "#1f4e79", "SC_PA"),
              "llmlingua": ("s--", "#c0392b", "PB")}

    def series(method, key):
        pts = sorted([(r["keep_ratio"], r[key]) for r in rows if r.get("method") == method])
        return [p[0] for p in pts], [p[1] for p in pts]

    # Plot 1: compression achieved
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, (mk, color, label) in styles.items():
        x, y = series(m, "avg_compression_ratio")
        if x:
            ax.plot(x, [v * 100 for v in y], mk, color=color, linewidth=2,
                    markersize=8, label=label)
    ax.set_xlabel("Keep ratio (fraction of tokens retained)")
    ax.set_ylabel("Compression achieved (%)")
    ax.set_title("Compression Achieved: SC_PA vs PB")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(plot_dir, "compression_comparison.png")
    fig.savefig(out1, dpi=150)
    print(f"Saved plot -> {out1}")

    # Plot 2: answer recall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, (mk, color, label) in styles.items():
        x, y = series(m, "answer_recall_pct")
        if x:
            ax.plot(x, y, mk, color=color, linewidth=2, markersize=8, label=label)
    ax.set_xlabel("Keep ratio (fraction of tokens retained)")
    ax.set_ylabel("Answer recall (%)")
    ax.set_title("Answer Recall: SC_PA vs PB")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out2 = os.path.join(plot_dir, "answer_recall_comparison.png")
    fig.savefig(out2, dpi=150)
    print(f"Saved plot -> {out2}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation or comparison results into a run folder")
    parser.add_argument("--project", default="Phase_1_CC_PA",
                        help="Project name under results/ (default: Phase_1_CC_PA; use 'comparison' for compare runs)")
    parser.add_argument("--results-dir", default="results",
                        help="Root directory for all results (default: results)")
    parser.add_argument("--run", default=None,
                        help="Specific run folder name (default: latest run for the project)")
    parser.add_argument("--input", default=None,
                        help="Explicit CSV path (overrides project/run auto-discovery)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    # Resolve the input CSV and the output plot directory
    if args.input:
        csv_path = args.input
        plot_dir = os.path.dirname(os.path.abspath(csv_path)) or "."
    else:
        run_dir = resolve_run_dir(args.results_dir, args.project, args.run)
        if not run_dir or not os.path.isdir(run_dir):
            print(f"No run found under {os.path.join(args.results_dir, args.project)}. "
                  f"Run evaluate_hf.py or compare_pipelines.py first.")
            return
        csv_path, _ = find_csv(run_dir)
        if not csv_path:
            print(f"No results CSV found under {os.path.join(run_dir, 'eval')}.")
            return
        plot_dir = os.path.join(run_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    rows = load_csv(csv_path)
    if not rows:
        print(f"No data found in {csv_path}")
        return

    kind = detect_kind(rows)
    print(f"Detected result type: {kind}  ({os.path.basename(csv_path)})")
    if kind == "comparison":
        plot_comparison(plt, rows, plot_dir)
    else:
        plot_single(plt, rows, plot_dir)


if __name__ == "__main__":
    main()
