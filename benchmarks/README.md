This contribution targets current Cellpose **main**, based on
`a54cb48849b7e225a81e8e43dcb042d42427f543`. It optimizes shared mask dynamics used
by Cellpose-SAM and the other current model backbones.

The runtime patch changes two operations in `cellpose/dynamics.py`:

* Update all flow coordinates together after `grid_sample`, keeping the FP32
  addition and clamp as separate operations. Per-coordinate arithmetic remains
  the same; 2D uses two arithmetic calls instead of four per step, and 3D uses
  two instead of six.
* Grow binary seed masks with native `max_pool2d`/`max_pool3d` instead of a
  sequence of axis-wise tensor operations. Since these masks contain only 0 and
  1, native pooling preserves the values, including boundary maxima and ties.

The histogram pooling, seed ordering, seed gathering, maximum scatter, mask
quality control, hole filling, label renumbering, normalization, resizing,
network weights and network precision retain upstream behavior. Current main
already includes vectorized seed gathering and maximum scatter. This change
reduces additional coordinate and pooling overhead with no new runtime dependency
or inference option.

Validation passed for both `cpsam` and main's default `cpsam_v2`, using upstream's
default bfloat16 network weights. It also passed for FP32, diameter 15 rescaling,
and a 3D volume. Across the runs listed below, all **93 timed full inference calls**
returned mask, flow and style arrays identical in shape, dtype and bytes to a
stable unmodified reference. All **93 timed cached-flow mask calls** also returned
identical mask bytes. The test run passed five tests (four new regression tests
and the existing upstream dynamics test), including 120 subtests. The randomized
CPU/CUDA differential audit passed 320 2D/3D mask comparisons.

MPS, other CUDA architectures, historical PyTorch versions, DINO end-to-end
inference and the full upstream GUI/training suite were not exercised. DINO uses
the same modified dynamics functions, but no DINO speed result is claimed.

Reproduce the checks in an environment with current Cellpose's dependencies:

```bash
python -m pytest tests/test_exact_dynamics.py tests/test_dynamics.py -q
python benchmarks/audit_exact_masks.py --output /tmp/sam-mask-audit.json
```

The tests cover exact coordinate bytes for 2D and 3D on CPU and CUDA, empty/single/
multiple point sets, 0/1/200 steps, zero flow and large displacements. Seed tests
cover image boundaries, overlapping equal-height seeds, the seed count threshold
and oversized mask removal. The audit uses randomized seeds with ties and
overlapping windows, and four maximum mask size fractions.

The public inputs are Cellpose's own test fixtures, from the source used by
[upstream conftest.py](https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/conftest.py).
They provide a small reproducible performance evaluation, not a full held-out
dataset benchmark or a ground-truth segmentation accuracy score. The equality
reference is inference from unmodified main on the same model and inputs.

```bash
curl -L --fail https://osf.io/download/s52q3/ -o /tmp/cellpose-data.zip
unzip /tmp/cellpose-data.zip -d /tmp/cellpose-fixtures
python benchmarks/benchmark_sam.py \
  --images /tmp/cellpose-fixtures/data/2D/gray_2D.png \
           /tmp/cellpose-fixtures/data/2D/rgb_2D.png \
           /tmp/cellpose-fixtures/data/2D/rgb_2D_tif.tif \
  --output /tmp/sam-public.json
```

Weights are cached in `CELLPOSE_LOCAL_MODELS_PATH` or the usual `.cellpose/models`
directory. For the exact checkpoints used here, obtain `cpsam` and `cpsam_v2` from
[this pinned model revision](https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837).
Raw reports include model and image SHA256 hashes. Weights and private images
are not included in this branch.

Additional configurations:

```bash
# Both checkpoints with twice the network input resolution, 100 dynamics steps
python benchmarks/benchmark_sam.py --images /path/to/gray_2D.png \
  --diameters 15 --output /tmp/sam-scaled.json

# FP32 network weights with the upstream TF32 setting preserved
python benchmarks/benchmark_sam.py --images /path/to/rgb_2D.png \
  --models cpsam_v2 --float32 --output /tmp/sam-float32.json

# Public 75 x 75 x 75 grayscale volume, depth on axis 0
python benchmarks/benchmark_sam.py --images /path/to/gray_3D.tif \
  --models cpsam_v2 --do-3d --z-axis 0 --repeats 3 --warmup 1 \
  --output /tmp/sam-3d.json
```

The reference functions are loaded from the pinned main revision in this git
checkout. Both implementations share the same model object and input, and the
only substituted functions are `steps_interp` and `get_masks_torch`. The full
inference timer includes the entire `CellposeModel.eval` call, including image
conversion, normalization, tiling, network inference, flow QC and mask cleanup.
It excludes image decoding and model loading. The mask timer uses the same
cached network flows through `_compute_masks`, including flow QC and cleanup.
It excludes network inference. Every timed result is checked after the timer.

Default settings are two warmups per implementation and image, five synchronized
repeats with alternating reference/candidate order, tile batch size 8, native
diameter (`None`), tile overlap 0.1, resampling enabled, flow threshold 0.4, and
minimum mask size 15. The 3D case uses one warmup and three repeats. A noisy supplemental microscopy
case was additionally measured with 15 repeats; both the initial and repeated
measurements are reported. No cases failing a speed threshold were discarded.

The machine was an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition with
PyTorch 2.9.1+cu128, Python 3.12.13 and eight CPU threads. The network uses default
bfloat16 unless explicitly marked FP32; dynamics use FP32 as upstream does. Other
GPU jobs were active, and full-inference differences of a few percent should not
be treated as a reliable overall speedup on this shared machine. Raw timing
samples and environment flags are included in `results/` for the public inputs.

On the default public 2D fixtures, the measured mask-stage improvement was
**1.13–1.28x**. The 3D case's mask stage improved **1.55x** (39.3 to 25.3 ms), while
its full inference time stayed around 3.18 seconds. Larger supplemental microscopy images showed
small overall timing differences. This supports a mask dynamics optimization;
it does not demonstrate a large SAM network or full-inference speedup.

PR figures (public test fixtures):

![SAM total inference latency and speedup](figures/sam_inference.png)

The left panel shows median full-inference latency and the change in milliseconds.
The right panel shows the ratio of median reference time to median optimized time.

![SAM mask construction and total inference effects](figures/sam_mask_effect.png)

The second figure compares full inference with mask construction from cached
network flows, including the public 3D volume. Whiskers are observed repeat ranges,
not confidence intervals. Shared GPU load makes small full-inference differences
uncertain; the mask-stage speedup is shown separately.

[Inference SVG](figures/sam_inference.svg) · [Inference PDF](figures/sam_inference.pdf) ·
[Mask-stage SVG](figures/sam_mask_effect.svg) · [Mask-stage PDF](figures/sam_mask_effect.pdf)

Regenerate these figures with NumPy and Matplotlib installed:

```bash
python benchmarks/plot_benchmarks.py --kind sam \
  --report benchmarks/results/public.json \
  --volume-report benchmarks/results/public-3d.json \
  --output-dir benchmarks/figures
```

Median latencies in milliseconds. All candidate masks, flows and styles matched
the reference bytes in every timed repeat.

| Input/configuration | Model | Eval reference | Eval optimized | Mask reference | Mask optimized | Mask speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gray_2D.png | cpsam_v2 | 308.7 | 297.9 | 66.2 | 58.7 | 1.13x |
| rgb_2D.png | cpsam_v2 | 153.7 | 153.9 | 57.9 | 51.2 | 1.13x |
| rgb_2D_tif.tif | cpsam_v2 | 161.9 | 154.2 | 58.0 | 50.1 | 1.16x |
| gray_2D.png | cpsam | 306.6 | 314.8 | 72.3 | 61.4 | 1.18x |
| rgb_2D.png | cpsam | 168.2 | 163.4 | 67.5 | 52.7 | 1.28x |
| rgb_2D_tif.tif | cpsam | 159.7 | 153.0 | 54.8 | 45.3 | 1.21x |
| Microscopy A | cpsam_v2 | 828.8 | 819.9 | 148.8 | 139.8 | 1.06x |
| Microscopy B | cpsam_v2 | 814.0 | 803.0 | 146.7 | 142.9 | 1.03x |
| Microscopy C | cpsam_v2 | 766.4 | 776.3 | 111.2 | 103.1 | 1.08x |
| Microscopy A | cpsam | 771.2 | 783.7 | 145.0 | 141.1 | 1.03x |
| Microscopy B | cpsam | 806.9 | 810.0 | 152.7 | 146.3 | 1.04x |
| Microscopy C | cpsam | 765.9 | 756.6 | 107.0 | 114.6 | 0.93x |
| gray_2D.png (diameter 15) | cpsam_v2 | 853.6 | 859.5 | 71.8 | 67.2 | 1.07x |
| gray_2D.png (diameter 15) | cpsam | 878.2 | 889.2 | 69.0 | 63.3 | 1.09x |
| rgb_2D.png (FP32) | cpsam_v2 | 205.9 | 202.4 | 58.4 | 49.5 | 1.18x |
| gray_3D.tif (3D) | cpsam_v2 | 3178.6 | 3174.9 | 39.3 | 25.3 | 1.55x |
| Microscopy C (15 repeats) | cpsam | 646.9 | 645.1 | 105.7 | 97.5 | 1.08x |

Supplemental inputs were three 1004 x 1362 uint16 RGB microscopy images.
These supplemental image files are not distributed. The initial Microscopy C
`cpsam` mask-stage result was noisy and slower; the longer repeat measured a
1.08x speedup. This is reported alongside the initial result.
