"""One-off helper: merge several single-keep-ratio comparison runs into one
combined run (CSV + JSON), then render the final chart + print the final table.

Usage:
    python _merge_runs.py run_A run_B run_C
"""
import csv
import json
import os
import sys

RESULTS = os.path.join("results", "comparison")

# Reuse the project's own table printer + plotter for identical formatting.
from compare_pipelines import print_table  # noqa: E402
from plot_results import plot_comparison  # noqa: E402


def main():
    run_names = sys.argv[1:]
    if not run_names:
        print("Provide run folder names to merge.")
        return

    all_runs = []          # full {aggregate, per_sample} objects -> merged JSON
    aggregate_rows = []    # aggregate dicts -> merged CSV + table
    for name in run_names:
        path = os.path.join(RESULTS, name, "eval", "comparison_results.json")
        if not os.path.isfile(path):
            print(f"[merge] skip (not found): {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for run in data.get("runs", []):
            all_runs.append(run)
            aggregate_rows.append(run["aggregate"])
        print(f"[merge] loaded {len(data.get('runs', []))} method-runs from {name}")

    if not aggregate_rows:
        print("[merge] nothing to merge.")
        return

    # Sort by keep ratio then method so the table/curve read left-to-right.
    order = {"cc_pa": 0, "llmlingua": 1}
    aggregate_rows.sort(key=lambda r: (r["keep_ratio"], order.get(r["method"], 9)))
    all_runs.sort(key=lambda r: (r["aggregate"]["keep_ratio"],
                                 order.get(r["aggregate"]["method"], 9)))

    # New combined run directory
    out_run = "run_20260620_combined_200"
    run_dir = os.path.join(RESULTS, out_run)
    eval_dir = os.path.join(run_dir, "eval")
    plot_dir = os.path.join(run_dir, "plot")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    json_path = os.path.join(eval_dir, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs}, f, indent=2)

    csv_path = os.path.join(eval_dir, "comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    # Point the project at the combined run for auto-discovery.
    with open(os.path.join(RESULTS, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(out_run)

    # Charts (same renderer compare_pipelines uses)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_comparison(plt, aggregate_rows, plot_dir)
    except ImportError:
        print("[merge] matplotlib not installed - skipping charts.")

    # Final table
    print_table(aggregate_rows)
    print(f"\n[merge] Combined run -> {run_dir}")
    print(f"[merge] CSV  -> {csv_path}")
    print(f"[merge] JSON -> {json_path}")


if __name__ == "__main__":
    main()
