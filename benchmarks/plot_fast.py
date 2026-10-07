"""Plot total 2D inference time and agreement with standard Cellpose-SAM masks."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_benchmarks import export, style_axis, hardware_label, INK, REFERENCE, OPTIMIZED, IMAGE_NAMES


def plot(report, directory, name):
    rows = report["results"]
    colors = [REFERENCE, OPTIMIZED, "#2563eb", "#c2410c"]
    variants = ["reference", "exact", "standard_batch32", "fast"]
    labels = ["Upstream (batch 8)", "Exact optimization (batch 8)",
              "Standard bfloat16 (batch 32)", "Fast FP16 (batch 32)"]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(7.8, len(rows) * 1.15 + 2.2)),
                             gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
    fig.subplots_adjust(left=.24, right=.92, top=.80, bottom=.23, wspace=.35)
    fig.text(.035, .96, "Cellpose-SAM: 2D segmentation time and mask agreement",
             fontsize=20, weight="bold", color=INK)
    fig.text(.035, .915, "Full inference includes preprocessing, network, dynamics, flow quality control and mask cleanup",
             fontsize=11, color=REFERENCE)
    handles = [plt.Line2D([0], [0], color=c, linewidth=8) for c in colors]
    fig.legend(handles, labels, bbox_to_anchor=(.235, .87), loc="upper left",
               frameon=False, ncol=2, fontsize=10)
    longest = max(max(row["variants"][v]["seconds"]) * 1000 for row in rows for v in variants)
    axes[0].set_xlim(0, longest * 1.65)
    table = []
    for i, row in enumerate(rows):
        baseline = np.median(row["variants"]["reference"]["seconds"]) * 1000
        for v, color, offset in zip(variants, colors, [-.30, -.10, .10, .30]):
            result = row["variants"][v]
            times = np.asarray(result["seconds"]) * 1000
            median = np.median(times)
            axes[0].barh(i + offset, median, height=.17, color=color,
                         xerr=[[median - times.min()], [times.max() - median]],
                         error_kw={"elinewidth": 1, "capsize": 2})
            suffix = ""
            if v != "reference":
                saved = baseline - median
                suffix = f" | {baseline / median:.2f}x | {abs(saved):.0f} ms {'saved' if saved > 0 else 'longer'}"
            axes[0].text(max(median, times.max()) + longest * .015, i + offset,
                         f"{median:.0f} ms{suffix}", fontsize=9, va="center", color=INK)
            if v != "reference":
                agree = result["agreement"]
                riou = np.mean([a["reference_object_mean_iou"] for a in agree])
                piou = np.mean([a["prediction_object_mean_iou"] for a in agree])
                value = min(riou, piou)
                misses = max(a["unmatched_reference_cells"] for a in agree)
                extra = max(a["unmatched_prediction_cells"] for a in agree)
                if v == "fast":
                    axes[1].scatter(value, i, color=color, s=45)
                    axes[1].text(1.025, i, f"IoU: {value:.4f}\nMissing: {misses}\nExtra: {extra}",
                                 va="center", fontsize=9, color=INK,
                                 transform=axes[1].get_yaxis_transform())
                table.append([row["model"], row["image"], row["diameter"], v, baseline,
                              median, baseline - median, baseline / median, riou, piou, misses, extra])
    ylabels = []
    for row in rows:
        shape = row["shape"]
        diameter = "Native scale (no resizing)" if row["diameter"] is None else f"Cell diameter: {row['diameter']:g} pixels"
        ylabels.append(f"{row['model']}: {IMAGE_NAMES.get(row['image'], row['image'])}\n"
                       f"2D image: {shape[0]} x {shape[1]} pixels\n{diameter}")
    axes[0].set_yticks(range(len(rows)), ylabels, fontsize=10)
    axes[0].set_ylim(len(rows) - .45, -.55)
    axes[0].set_xlabel("Total inference time (milliseconds)", fontsize=12)
    minimum = min(min(np.mean([a[k] for a in row["variants"]["fast"]["agreement"]])
                      for k in ["reference_object_mean_iou", "prediction_object_mean_iou"])
                  for row in rows)
    axes[1].set_xlim(max(0, minimum - .02), 1.003)
    axes[1].set_xlabel("Fast mode: object IoU agreement\nwith standard masks (higher is closer)", fontsize=11)
    axes[1].axvline(1, color=REFERENCE, linestyle="--", linewidth=1)
    for ax in axes:
        style_axis(ax)
    fig.text(.035, .115, "Object agreement: smaller of reference/prediction mean IoU after one-to-one matching at IoU >= 0.5; unmatched cells count as zero.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .080, "Agreement measures similarity to standard inference, not ground-truth accuracy. Exact mode matches mask bytes; fast mode can change boundaries and counts.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .045, f"Medians of {report['parameters']['repeats']} synchronized repeats; whiskers: min-max. Standard/exact batch: 8; fast batch: 32. {hardware_label(report)}.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .010, "Default bfloat16 standard network versus FP16 fast network. Model loading, image decoding and initial FP16 cache creation excluded; shared GPU.",
             fontsize=9, color=REFERENCE)
    export(fig, directory, name)
    with (directory / f"{name}_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["checkpoint", "image", "diameter", "mode", "reference_ms", "mode_ms",
                         "saved_ms", "speedup", "reference_object_mean_iou",
                         "prediction_object_mean_iou", "missing_cells_max", "extra_cells_max"])
        writer.writerows(table)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name", default="sam_fast_inference")
    args = p.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    plot(json.loads(args.report.read_text()), args.output_dir, args.name)
