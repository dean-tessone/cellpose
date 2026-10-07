"""Compare SAM inference and cached-flow mask construction to unmodified main.

Run from a checkout with SAM dependencies and both pretrained checkpoints.
Image decoding and model loading are excluded from the inference timings.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cellpose import dynamics, io, models, transforms

BASE_REVISION = "a54cb48849b7e225a81e8e43dcb042d42427f543"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def reference_functions(revision):
    source = git("show", f"{revision}:cellpose/dynamics.py").decode()
    names = {"steps_interp", "get_masks_torch"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError("Reference must expose steps_interp and get_masks_torch")
    namespace = dict(vars(dynamics))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<git reference>", "exec"), namespace)
    return {name: namespace[name] for name in names}


@contextmanager
def implementation(functions):
    saved = {name: getattr(dynamics, name) for name in functions}
    try:
        for name, fn in functions.items():
            setattr(dynamics, name, fn)
        yield
    finally:
        for name, fn in saved.items():
            setattr(dynamics, name, fn)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def arrays_exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def comparison(ref, pred):
    same_shape = ref.shape == pred.shape
    return {"exact": bool(arrays_exact(ref, pred)),
            "reference_dtype": str(ref.dtype), "candidate_dtype": str(pred.dtype),
            "changed_pixels": int(np.count_nonzero(ref != pred)) if same_shape else None,
            "foreground_changed_pixels": int(np.count_nonzero((ref > 0) != (pred > 0)))
            if same_shape else None,
            "reference_sha256": digest(ref), "candidate_sha256": digest(pred),
            "reference_cells": int(ref.max()), "candidate_cells": int(pred.max())}


def timed(fn, device):
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, elapsed, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["cpsam_v2", "cpsam"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default=BASE_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diameters", nargs="+", default=["native"],
                        help="'native' keeps upstream diameter=None; or specify pixels")
    parser.add_argument("--float32", action="store_true", help="Override default bfloat16 network weights")
    parser.add_argument("--do-3d", action="store_true")
    parser.add_argument("--z-axis", type=int, help="Default is 0 for 3D volumes")
    parser.add_argument("--channel-axis", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must both be positive")
    diameters = [None if d == "native" else float(d) for d in args.diameters]
    z_axis = 0 if args.do_3d and args.z_axis is None else args.z_axis
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    reference = reference_functions(args.reference)
    candidate = {name: getattr(dynamics, name) for name in reference}
    images = [(p.name, io.imread(str(p))) for p in args.images]
    if any(image is None for _, image in images):
        raise ValueError("Could not decode an input image")
    report = {"reference_revision": git("rev-parse", args.reference).decode().strip(),
        "candidate_revision": git("rev-parse", "HEAD").decode().strip(),
        "candidate_diff_sha256": hashlib.sha256(git("diff", "--", "cellpose/dynamics.py")).hexdigest(),
        "dynamics_source_sha256": file_digest(ROOT / "cellpose/dynamics.py"),
        "benchmark_source_sha256": file_digest(__file__),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "segment_anything": version("segment-anything"),
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threads": args.threads, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader"], text=True).strip() if device.type == "cuda" else None,
        "parameters": {"batch_size": args.batch_size, "channel_axis": args.channel_axis,
        "z_axis": z_axis,
        "flow_threshold": .4, "min_size": 15, "augment": False,
        "tile_overlap": .1, "resample": True, "do_3D": args.do_3d,
        "use_bfloat16": not args.float32, "warmup": args.warmup,
        "repeats": args.repeats}, "models": {}, "results": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for model_name in args.models:
        model = models.CellposeModel(gpu=device.type == "cuda", pretrained_model=model_name,
                                     device=device, use_bfloat16=not args.float32)
        report["models"][model_name] = {"model_sha256": file_digest(model.pretrained_model),
                                      "network_dtype": str(model.net.dtype)}
        for diameter in diameters:
            for name, image in images:
                def evaluate(functions):
                    with implementation(functions):
                        return model.eval(image, diameter=diameter,
                            batch_size=args.batch_size, do_3D=args.do_3d,
                            channel_axis=args.channel_axis, z_axis=z_axis,
                            augment=False, flow_threshold=.4, min_size=15, tile_overlap=.1)
                for _ in range(args.warmup):
                    baseline = evaluate(reference)
                    evaluate(candidate)
                converted = transforms.convert_image(image, do_3D=args.do_3d,
                    channel_axis=args.channel_axis, z_axis=z_axis)
                shape = (1, *converted.shape) if converted.ndim < 4 else converted.shape
                flow, probability = baseline[1][1:]
                if not args.do_3d:
                    flow, probability = flow[:, None], probability[None]
                iterations = 200 if diameter is None or diameter <= 0 else int(200 / (30. / diameter))

                def masks_only(functions):
                    with implementation(functions):
                        return model._compute_masks(shape, flow, probability,
                            flow_threshold=.4, min_size=15, niter=iterations, do_3D=args.do_3d)

                row = {"model": model_name, "image": name, "shape": list(image.shape),
                       "input_sha256": digest(image), "diameter": diameter,
                       "reference_stable": True, "masks_exact_all_repeats": True,
                       "flows_exact_all_repeats": True, "styles_exact_all_repeats": True,
                       "mask_stage_exact_all_repeats": True}
                for stage, fn in [("eval", evaluate), ("masks", masks_only)]:
                    if stage == "masks":
                        for _ in range(args.warmup):
                            fn(reference)
                            fn(candidate)
                    row[stage] = {"reference_seconds": [], "candidate_seconds": []}
                    for repeat in range(args.repeats):
                        order = [("reference", reference), ("candidate", candidate)]
                        if repeat % 2:
                            order.reverse()
                        for label, functions in order:
                            output, elapsed, memory = timed(lambda: fn(functions), device)
                            row[stage][f"{label}_seconds"].append(elapsed)
                            row[stage][f"{label}_peak_allocated_bytes"] = max(
                                row[stage].get(f"{label}_peak_allocated_bytes", 0), memory or 0)
                            mask = output[0] if stage == "eval" else output
                            cmp = comparison(baseline[0], mask)
                            if label == "reference":
                                row["reference_stable"] &= cmp["exact"]
                            elif stage == "eval":
                                row["masks_exact_all_repeats"] &= cmp["exact"]
                                row["mask_comparison"] = cmp
                                row["flows_exact_all_repeats"] &= all(
                                    arrays_exact(a, b) for a, b in zip(baseline[1], output[1]))
                                row["styles_exact_all_repeats"] &= arrays_exact(baseline[2], output[2])
                            else:
                                row["mask_stage_exact_all_repeats"] &= cmp["exact"]
                    row[stage]["speedup"] = statistics.median(row[stage]["reference_seconds"]) / statistics.median(row[stage]["candidate_seconds"])
                report["results"].append(row)
                save()
                print(f"{model_name} {name} diameter={diameter}: eval {row['eval']['speedup']:.2f}x; masks {row['masks']['speedup']:.2f}x; exact={row['masks_exact_all_repeats']}", flush=True)
                if not all(row[k] for k in ("reference_stable", "masks_exact_all_repeats",
                    "flows_exact_all_repeats", "styles_exact_all_repeats", "mask_stage_exact_all_repeats")):
                    report["all_exact"] = False
                    save()
                    raise SystemExit("Exact equivalence check failed; see report")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["all_exact"] = True
    save()


if __name__ == "__main__":
    main()
