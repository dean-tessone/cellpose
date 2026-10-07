"""Measure optional fast inference against unmodified Cellpose main.

IoU measures agreement with standard inference, not ground-truth accuracy.
Model loading, decoding and lazy FP16 cache creation are excluded from timings.
"""
import argparse
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import torch

from benchmark_sam import (ROOT, BASE_REVISION, git, reference_functions,
                           implementation, timed, digest, file_digest, arrays_exact)
from cellpose import models, dynamics, io, metrics


def agreement(reference, prediction):
    """One-to-one object matching; unmatched objects contribute zero IoU."""
    nr, npred = int(reference.max()), int(prediction.max())
    union = np.count_nonzero((reference > 0) | (prediction > 0))
    foreground = (np.count_nonzero((reference > 0) & (prediction > 0)) / union
                  if union else 1.)
    if nr and npred:
        ious, ids = metrics.mask_ious(reference, prediction)
        matched = (ids > 0) & (ious >= .5)
        total = float(ious[matched].sum())
        count = int(matched.sum())
    else:
        total, count = 0., 0
    return {"bitwise_identical": bool(arrays_exact(reference, prediction)),
            "foreground_iou": float(foreground),
            "reference_object_mean_iou": total / nr if nr else float(npred == 0),
            "prediction_object_mean_iou": total / npred if npred else float(nr == 0),
            "matched_object_mean_iou": total / count if count else float(nr == npred == 0),
            "matched_objects_at_iou_0_5": count,
            "reference_cells": nr, "prediction_cells": npred,
            "unmatched_reference_cells": nr - count,
            "unmatched_prediction_cells": npred - count}


def reference_methods(revision):
    source = git("show", f"{revision}:cellpose/models.py").decode()
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "CellposeModel")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in {"eval", "_run_net"}]
    namespace = dict(vars(models))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<reference models>", "exec"), namespace)
    return {n.name: namespace[n.name] for n in nodes}


@contextmanager
def model_methods(methods):
    saved = {name: getattr(models.CellposeModel, name) for name in methods}
    try:
        for name, method in methods.items():
            setattr(models.CellposeModel, name, method)
        yield
    finally:
        for name, method in saved.items():
            setattr(models.CellposeModel, name, method)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["cpsam_v2", "cpsam"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diameters", nargs="+", default=["native"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if min(args.repeats, args.warmup) < 1:
        p.error("repeats and warmup must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0")
    ref_functions = reference_functions(BASE_REVISION)
    ref_methods = reference_methods(BASE_REVISION)
    variants = {"reference": {"batch_size": 8},
                "exact": {"batch_size": 8, "fast": False},
                "standard_batch32": {"batch_size": 32, "fast": False},
                "fast_batch8": {"batch_size": 8, "fast": True},
                "fast": {"fast": True}}
    report = {"reference_revision": BASE_REVISION,
              "candidate_revision": git("rev-parse", "HEAD").decode().strip(),
              "source_sha256": {name: file_digest(ROOT / "cellpose" / name)
                                for name in ["fast.py", "models.py", "dynamics.py"]},
              "benchmark_source_sha256": file_digest(__file__),
              "python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device), "threads": args.threads,
              "parameters": {"do_3D": False, "augment": False, "flow_threshold": .4,
                             "niter": "upstream default (diameter dependent)",
                             "min_size": 15, "tile_overlap": .1, "resample": True,
                             "warmup": args.warmup, "repeats": args.repeats},
              "variants": variants, "models": {}, "results": []}
    for model_name in args.models:
        model = models.CellposeModel(gpu=True, device=device, pretrained_model=model_name)
        report["models"][model_name] = {"weights_sha256": file_digest(model.pretrained_model),
                                      "standard_dtype": str(model.net.dtype), "fast_dtype": "torch.float16"}
        for diameter_text in args.diameters:
            diameter = None if diameter_text == "native" else float(diameter_text)
            for path in args.images:
                image = io.imread(str(path))
                def evaluate(label):
                    if label == "reference":
                        with implementation(ref_functions), model_methods(ref_methods):
                            return model.eval(image, diameter=diameter, **variants[label])
                    return model.eval(image, diameter=diameter, **variants[label])
                for _ in range(args.warmup):
                    for label in variants:
                        outputs = evaluate(label)
                        if label == "reference":
                            baseline = outputs
                row = {"model": model_name, "image": path.name, "shape": list(image.shape),
                       "input_sha256": digest(image), "diameter": diameter,
                       "variants": {name: {"seconds": [], "agreement": [], "peak_allocated_bytes": 0}
                                    for name in variants}}
                for repeat in range(args.repeats):
                    # Rotate order so fast mode is neither always first nor always last.
                    labels = list(variants)
                    labels = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
                    for label in labels:
                        outputs, seconds, memory = timed(lambda: evaluate(label), device)
                        result = row["variants"][label]
                        result["seconds"].append(seconds)
                        result["peak_allocated_bytes"] = max(result["peak_allocated_bytes"], memory)
                        result["agreement"].append(agreement(baseline[0], outputs[0]))
                        if label in {"reference", "exact"}:
                            if not (arrays_exact(baseline[0], outputs[0]) and
                                    all(arrays_exact(a, b) for a, b in zip(baseline[1], outputs[1])) and
                                    arrays_exact(baseline[2], outputs[2])):
                                raise AssertionError("Standard output changed, including after fast calls")
                report["results"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                ref_time = statistics.median(row["variants"]["reference"]["seconds"])
                fast = row["variants"]["fast"]
                print(f"{model_name} {path.name} diameter={diameter}: "
                      f"fast {ref_time / statistics.median(fast['seconds']):.2f}x; "
                      f"object IoU {fast['agreement'][0]['reference_object_mean_iou']:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
