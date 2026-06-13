"""
Plot Evaluation Results
=======================

Reads the aggregate summary produced by evaluate_hf.py and renders TWO
separate PNG charts into the same versioned run folder:

    1. compression_achieved.png   - Compression achieved (%) vs reduce_ratio
    2. information_preserved.png   - Answer recall (%) & keyword retention (%)
                                     vs reduce_ratio

By default it auto-discovers the LATEST run for a project under
results/<project>/ (via results/<project>/latest.txt) and reads
    results/<project>/<run>/eval/eval_results.csv
then writes the plots to
    results/<project>/<run>/plot/

USAGE
-----
    pip install matplotlib
    python plot_results.py                                  # latest Phase_1_CC_PA run
    python plot_results.py --project Phase_1_CC_PA
    python plot_results.py --run run_20260613_101500        # a specific run
    python plot_results.py --input path/to/eval_results.csv # explicit CSV

The plots need a sweep, so generate the CSV first with:
    python evaluate_hf.py --reduce-ratios 0.3 0.5 0.7
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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["reduce_ratio"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results into a versioned run folder")
    parser.add_argument("--project", default="Phase_1_CC_PA",
                        help="Project name under results/ (default: Phase_1_CC_PA)")
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
                  f"Run evaluate_hf.py first.")
            return
        csv_path = os.path.join(run_dir, "eval", "eval_results.csv")
        plot_dir = os.path.join(run_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    rows = load_csv(csv_path)
    if not rows:
        print(f"No data found in {csv_path}")
        return

    reduce_ratios = [r["reduce_ratio"] for r in rows]
    compression = [r["avg_compression_ratio"] * 100 for r in rows]   # -> %
    answer_recall = [r["answer_recall_pct"] for r in rows]
    keyword_ret = [r["avg_keyword_retention"] for r in rows]

    note = "  (single point — run a --reduce-ratios sweep for a curve)" if len(rows) == 1 else ""

    # ------------------------------------------------------------------
    # Plot 1: Compression achieved vs reduce_ratio
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(reduce_ratios, compression, "o-", color="#1f4e79",
             linewidth=2, markersize=8)
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

    # ------------------------------------------------------------------
    # Plot 2: Information preserved vs reduce_ratio
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
