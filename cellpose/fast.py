"""Opt-in CUDA inference with FP16 and GPU image/tile processing.

This path is approximate. Network precision, percentile arithmetic, bilinear
resizing and tile accumulation can change mask boundaries and cell counts.
Mask dynamics, flow quality control and cleanup remain in the standard pipeline.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import transforms


def supports_normalization(params):
    """GPU normalization handles per-image percentiles, including custom percentiles."""
    return (params.get("normalize", True) and params.get("lowhigh") is None
            and not params.get("invert", False)
            and not params.get("sharpen_radius", 0)
            and not params.get("smooth_radius", 0)
            and not params.get("tile_norm_blocksize", 0))


def normalize_gpu(images, percentile=None):
    """Per-image, per-channel percentile normalization of BCHW float32 tensors."""
    lower, upper = (1., 99.) if percentile is None else percentile
    if not 0 <= lower < upper <= 100:
        raise ValueError("Invalid percentile range, should be between 0 and 100")
    height, width = images.shape[-2:]
    sy, sx = (max(1, height // 224), max(1, width // 224)) \
        if height * width > 224**3 else (1, 1)
    sampled = images[..., ::sy, ::sx].flatten(2)
    q = torch.tensor([lower / 100., upper / 100.], device=images.device,
                     dtype=torch.float32)
    if sampled.numel() > 2**24:
        # quantile limits the total input size, including batch/channel axes.
        flat = sampled.reshape(-1, sampled.shape[-1])
        bounds = torch.stack([torch.quantile(channel, q) for channel in flat], dim=1)
        bounds = bounds.reshape(2, *sampled.shape[:-1])
    else:
        bounds = torch.quantile(sampled, q, dim=-1)
    lo, hi = bounds[0][..., None, None], bounds[1][..., None, None]
    span = hi - lo
    return torch.where(span > 1e-3, (images - lo) / span.clamp_min(1e-3),
                       torch.zeros_like(images))


def _upload(images, device, cache):
    shape = (len(images), images.shape[-1], *images.shape[1:3])
    size = int(np.prod(shape))
    host = cache.get("host")
    if host is None or host.numel() < size:
        host = torch.empty(size, dtype=torch.float32, pin_memory=True)
        cache["host"] = host
    host = host[:size].view(shape)
    np.copyto(host.numpy(), np.moveaxis(images, -1, 1), casting="unsafe")
    return host.to(device, non_blocking=True)


def _tile_positions(height, width, ly, lx, bsize, overlap):
    overlap = min(.5, max(.05, overlap))
    ny = 1 if height <= bsize else int(np.ceil((1 + 2 * overlap) * height / bsize))
    nx = 1 if width <= bsize else int(np.ceil((1 + 2 * overlap) * width / bsize))
    ys = np.linspace(0, height - ly, ny).astype(int)
    xs = np.linspace(0, width - lx, nx).astype(int)
    return np.repeat(ys, nx), np.tile(xs, ny)


def _fp16_parameters(net, cache):
    parameters = dict(net.named_parameters())
    signature = tuple((name, p.data_ptr(), p._version, p.dtype, p.device)
                      for name, p in parameters.items())
    if cache.get("parameter_signature") != signature:
        cache["fp16_parameters"] = {
            name: p.detach().to(torch.float16) if p.is_floating_point() else p.detach()
            for name, p in parameters.items()}
        cache["parameter_signature"] = signature
    return cache["fp16_parameters"]


@torch.inference_mode()
def run_net_fast(net, images, batch_size=32, tile_overlap=.1, bsize=256,
                 rescale=1., resample=True, normalize_params=None, cache=None,
                 min_tile_size=True, normalize_style=False):
    """Run unaugmented 2D tiles, preserving input channels and output shapes.

    Cache buffers and FP16 parameters belong to one model's sequential calls.
    The original parameters are restored by functional_call after each forward;
    the FP16 cache is refreshed when parameter versions or storage change. Each chunk
    downloads its result before the pinned input buffer is reused. FP16 overflow
    raises an error instead of producing masks from nonfinite network outputs.
    """
    device = net.device
    if device.type != "cuda":
        raise ValueError("run_net_fast requires a CUDA device")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache = {} if cache is None else cache
    fp16_parameters = _fp16_parameters(net, cache)
    functional_call = getattr(getattr(torch, "func", None), "functional_call", None)
    if functional_call is None:
        from torch.nn.utils.stateless import functional_call
    count, height, width, _ = images.shape
    hr, wr = int(height * rescale), int(width * rescale)
    if min(hr, wr) < 1:
        raise ValueError("diameter/rescale produces an empty image")
    padding = transforms.get_pad_yx(hr, wr,
        min_size=(bsize, bsize) if min_tile_size else None)
    py1, py2, px1, px2 = padding
    hp, wp = hr + py1 + py2, wr + px1 + px2
    ly, lx = min(bsize, hp), min(bsize, wp)
    ys, xs = _tile_positions(hp, wp, ly, lx, bsize, tile_overlap)
    tile_count = len(ys)
    images_per_chunk = max(1, batch_size // tile_count)
    index_y = torch.as_tensor(ys, device=device)
    index_x = torch.as_tensor(xs, device=device)
    taper_key = ("taper", device, ly, lx)
    if taper_key not in cache:
        cache[taper_key] = torch.as_tensor(transforms._taper_mask(ly, lx),
                                          device=device, dtype=torch.float32)
    taper = cache[taper_key]
    net.eval()
    results, styles = [], []
    for start in range(0, count, images_per_chunk):
        inputs = _upload(images[start:start + images_per_chunk], device, cache)
        if normalize_params is not None:
            inputs = normalize_gpu(inputs, normalize_params.get("percentile"))
        if (hr, wr) != (height, width):
            inputs = F.interpolate(inputs, size=(hr, wr), mode="bilinear",
                                   align_corners=False)
        inputs = F.pad(inputs, (px1, px2, py1, py2), mode="constant")
        # All candidate windows are views; advanced indexing materializes only
        # the selected upstream tile coordinates, preserving channels/order.
        windows = inputs.unfold(2, ly, 1).unfold(3, lx, 1)
        tiles = windows[:, :, index_y, index_x].permute(0, 2, 1, 3, 4)
        batch = len(inputs)
        tiles = tiles.reshape(batch * tile_count, inputs.shape[1], ly, lx)
        predictions, tile_styles = [], []
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for offset in range(0, len(tiles), batch_size):
                prediction, style = functional_call(net, fp16_parameters,
                    (tiles[offset:offset + batch_size].to(torch.float16),))[:2]
                predictions.append(prediction.float())
                if normalize_style:
                    tile_styles.append(style.float())
        predictions = torch.cat(predictions).reshape(batch, tile_count, -1, ly, lx)
        output = torch.zeros((batch, predictions.shape[2], hp, wp), device=device)
        weights = torch.zeros((1, 1, hp, wp), device=device)
        # Ordered accumulation avoids atomic floating point scatter and keeps
        # the same tile layout/taper; FP32 weighting remains approximate.
        for tile, (y, x) in enumerate(zip(ys, xs)):
            output[..., y:y + ly, x:x + lx].add_(predictions[:, tile] * taper)
            weights[..., y:y + ly, x:x + lx].add_(taper)
        output.div_(weights)
        output = output[..., py1:py1 + hr, px1:px1 + wr]
        if resample and (hr, wr) != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear",
                                   align_corners=False)
        result = output.permute(0, 2, 3, 1).cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("FP16 fast inference produced nonfinite outputs; "
                                     "rerun with fast=False")
        results.append(result)
        if normalize_style:
            style = torch.cat(tile_styles).reshape(batch, tile_count, -1).sum(dim=1)
            style = style / style.norm(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            style = torch.zeros((batch, 256), device=device)
        styles.append(style.cpu().numpy())
    return np.concatenate(results), np.concatenate(styles)
