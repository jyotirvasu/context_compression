"""
Plot Evaluation Results
=======================

Reads the aggregate summary produced by evaluate_hf.py (eval_results.csv)
and renders the key compression-vs-answer-recall trade-off curve as a PNG
suitable for inclusion in a report.

USAGE
-----
    pip install matplotlib
    python plot_results.py                          # reads eval_results.csv
    python plot_results.py --input eval_results.csv --output compression_vs_recall.png

The most informative plot needs a sweep, so generate the CSV first with:
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
    parser = argparse.ArgumentParser(description="Plot compression-vs-recall curve from eval_results.csv")
    parser.add_argument("--input", default="eval_results.csv", help="CSV produced by evaluate_hf.py")
    parser.add_argument("--output", default="compression_vs_recall.png", help="Output PNG path")
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

    single_point = len(rows) == 1

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left axis: compression achieved
    color_comp = "#1f4e79"
    ax1.set_xlabel("Configured reduce_ratio")
    ax1.set_ylabel("Compression achieved (%)", color=color_comp)
    ax1.plot(reduce_ratios, compression, "o-", color=color_comp,
             linewidth=2, markersize=7, label="Compression %")
    ax1.tick_params(axis="y", labelcolor=color_comp)
    ax1.set_ylim(0, 100)

    # Right axis: information preserved (answer recall + keyword retention)
    ax2 = ax1.twinx()
    color_rec = "#c0392b"
    color_kw = "#27ae60"
    ax2.set_ylabel("Information preserved (%)")
    ax2.plot(reduce_ratios, answer_recall, "s--", color=color_rec,
             linewidth=2, markersize=7, label="Answer recall %")
    ax2.plot(reduce_ratios, keyword_ret, "^:", color=color_kw,
             linewidth=2, markersize=7, label="Keyword retention %")
    ax2.tick_params(axis="y")
    ax2.set_ylim(0, 105)

    title = "Compression vs. Information Preservation"
    if single_point:
        title += "  (single point — run a --reduce-ratios sweep for a curve)"
    plt.title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot -> {args.output}")


if __name__ == "__main__":
    main()
