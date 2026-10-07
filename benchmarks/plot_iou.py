"""Render standalone Cellpose-SAM fast-mode mask agreement plots from raw results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_benchmarks import export, style_axis, hardware_label, IMAGE_NAMES, INK, REFERENCE


def plot(report, directory, name):
    rows = report["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, .95 * len(rows) + 2.2)), sharey=True)
    fig.subplots_adjust(left=.30, right=.96, top=.79, bottom=.22, wspace=.30)
    fig.text(.035, .96, "Cellpose-SAM: fast-mode mask IoU agreement on 2D images",
             fontsize=18, weight="bold", color=INK)
    fig.text(.035, .90, "FP16 with GPU preprocessing and tiling compared with unmodified standard bfloat16 inference",
             fontsize=10, color=REFERENCE)
    labels, scores = [], []
    for i, row in enumerate(rows):
        agreement = row["variants"]["fast"]["agreement"]
        objects = np.minimum([a["reference_object_mean_iou"] for a in agreement],
                             [a["prediction_object_mean_iou"] for a in agreement])
        foreground = np.asarray([a["foreground_iou"] for a in agreement])
        missed = max(a["unmatched_reference_cells"] for a in agreement)
        extra = max(a["unmatched_prediction_cells"] for a in agreement)
        scale = "Native scale" if row["diameter"] is None else f"Cell diameter: {row['diameter']:g} pixels"
        labels.append(f"{row['model']}: {IMAGE_NAMES.get(row['image'], row['image'])}\n"
                      f"{scale}\nMissing cells: {missed}; extra cells: {extra}")
        for ax, values, color in zip(axes, [objects, foreground], ["#c2410c", "#2563eb"]):
            center = np.mean(values)
            ax.errorbar(center, i, xerr=[[max(0., center - values.min())],
                                       [max(0., values.max() - center)]],
                        fmt="o", color=color, markersize=7, capsize=3)
            ax.annotate(f"{center:.4f}", (center, i), xytext=(-10, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=10)
            scores.extend(values)
    minimum = min(scores)
    for ax in axes:
        ax.set_xlim(max(0, minimum - .013), 1.001)
        ax.axvline(1, color=REFERENCE, linestyle="--", linewidth=1)
        style_axis(ax)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=10)
    axes[0].set_ylim(len(rows) - .5, -.5)
    axes[0].set_title("Object agreement (unmatched cells count as zero)", fontsize=11, pad=12)
    axes[1].set_title("Foreground agreement (all cell pixels)", fontsize=11, pad=12)
    for ax in axes:
        ax.set_xlabel("IoU agreement with standard masks (higher is closer)", fontsize=10)
    fig.text(.035, .13, "Object score: smaller of reference/prediction mean IoU after one-to-one matching at IoU >= 0.5. Missing/extra counts: repeat maxima.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .085, "Agreement measures similarity to standard inference, not accuracy against manual annotations. Fast-mode mask bytes differ in every configuration.",
             fontsize=9, color=REFERENCE)
    fig.text(.035, .04, f"Markers: mean agreement across {report['parameters']['repeats']} repeats; whiskers: observed min-max. {hardware_label(report)}. Flow QC and iterations retained.",
             fontsize=9, color=REFERENCE)
    export(fig, directory, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="sam_2d_iou_public")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    plot(json.loads(args.report.read_text()), args.output_dir, args.name)
