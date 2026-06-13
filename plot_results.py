"""
Plot Evaluation Results
=======================

Reads the aggregate summary produced by evaluate_hf.py (eval_results.csv)
and renders TWO separate PNG charts suitable for a report:

    1. compression_achieved.png   - Compression achieved (%) vs reduce_ratio
    2. information_preserved.png   - Answer recall (%) & keyword retention (%)
                                     vs reduce_ratio

USAGE
-----
    pip install matplotlib
    python plot_results.py                          # reads eval_results.csv
    python plot_results.py --input eval_results.csv --output-prefix myrun

The plots need a sweep, so generate the CSV first with:
    python evaluate_hf.py --reduce-ratios 0.3 0.5 0.7
"""

import argparse
import csv


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    rows.sort(key=lambda r: r["reduce_ratio"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results from eval_results.csv")
    parser.add_argument("--input", default="eval_results.csv", help="CSV produced by evaluate_hf.py")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for output PNG names")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    rows = load_csv(args.input)
    if not rows:
        print(f"No data found in {args.input}")
        return

    reduce_ratios = [r["reduce_ratio"] for r in rows]
    compression = [r["avg_compression_ratio"] * 100 for r in rows]   # -> %
    answer_recall = [r["answer_recall_pct"] for r in rows]
    keyword_ret = [r["avg_keyword_retention"] for r in rows]

    note = "  (single point — run a --reduce-ratios sweep for a curve)" if len(rows) == 1 else ""
    prefix = args.output_prefix

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
    out1 = f"{prefix}compression_achieved.png"
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
    out2 = f"{prefix}information_preserved.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved plot -> {out2}")


if __name__ == "__main__":
    main()
