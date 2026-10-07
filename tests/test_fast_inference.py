"""Optional fast inference: numerical contracts, routing, and weight isolation."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cellpose import fast, models, transforms, cli


class ToyNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False).to(device)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(3, device=device).reshape(3, 3, 1, 1))
        self.device = torch.device(device)

    def forward(self, x):
        return self.conv(x), torch.zeros((len(x), 256), device=x.device)


class FastInferenceTests(unittest.TestCase):
    def test_normalization(self):
        rng = np.random.default_rng(10)
        image = rng.uniform(0, 1, (2, 35, 47, 3)).astype(np.float32)
        image[..., 1] = 12
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for percentiles in [(1, 99), (5, 90)]:
                tensor = torch.from_numpy(image.transpose(0, 3, 1, 2)).to(device)
                actual = fast.normalize_gpu(tensor, percentiles).cpu().numpy().transpose(0, 2, 3, 1)
                expected = transforms.normalize_img(image.copy(), percentile=percentiles,
                                                     norm3D=False)
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for bad in [(99, 1), (-1, 99), (1, 101)]:
            with self.assertRaises(ValueError):
                fast.normalize_gpu(torch.ones(1, 3, 5, 5), bad)

    def test_advanced_normalization_routing(self):
        self.assertTrue(fast.supports_normalization({"percentile": [5, 95]}))
        for key, value in [("normalize", False), ("lowhigh", [0, 10]),
                           ("invert", True), ("sharpen_radius", 1),
                           ("smooth_radius", 1), ("tile_norm_blocksize", 32)]:
            self.assertFalse(fast.supports_normalization({key: value}))

    def test_cli(self):
        parser = cli.get_arg_parser()
        self.assertFalse(parser.parse_args([]).fast)
        self.assertIsNone(parser.parse_args([]).batch_size)
        self.assertTrue(parser.parse_args(["--fast"]).fast)
        self.assertEqual(parser.parse_args(["--fast", "--batch_size", "4"]).batch_size, 4)
        self.assertIn("not bitwise identical", " ".join(parser.format_help().split()))

    def test_eval_routing_and_mask_parameters(self):
        model = models.CellposeModel.__new__(models.CellposeModel)
        model.device = torch.device("cuda")
        image = np.ones((24, 25, 3), np.float32)

        def network(x, **kwargs):
            return np.zeros((2, *x.shape[:3]), np.float32), np.zeros(x.shape[:3], np.float32), np.zeros(256)

        def masks(shape, *args, **kwargs):
            return np.zeros(shape[:-1], np.uint16)

        for kwargs, expected_fast, batch, steps in [
                ({}, False, 8, 200), ({"fast": True}, True, 32, 200),
                ({"fast": True, "batch_size": 3, "diameter": 15}, True, 3, 100),
                ({"fast": True, "niter": 57, "flow_threshold": .25}, True, 32, 57),
                ({"fast": True, "augment": True}, False, 8, 200),
                ({"fast": True, "stitch_threshold": .1, "z_axis": 0, "channel_axis": 3}, False, 8, 200),
                ({"fast": True, "do_3D": True, "z_axis": 0, "channel_axis": 3}, False, 8, 200)]:
            with patch.object(model, "_run_net", side_effect=network, create=True) as run, \
                 patch.object(model, "_compute_masks", side_effect=masks, create=True) as compute:
                model.eval(np.stack([image, image]) if "z_axis" in kwargs else image, **kwargs)
                self.assertEqual(run.call_args.kwargs["fast"], expected_fast)
                self.assertEqual(run.call_args.kwargs["batch_size"], batch)
                self.assertEqual(compute.call_args.kwargs["niter"], steps)
                self.assertEqual(compute.call_args.kwargs["flow_threshold"], kwargs.get("flow_threshold", .4))
        model.device = torch.device("cpu")
        with patch.object(model, "_run_net", side_effect=network, create=True) as run:
            model.eval(image, fast=True, compute_masks=False)
            self.assertFalse(run.call_args.kwargs["fast"])
            self.assertEqual(run.call_args.kwargs["batch_size"], 8)
        model.device = torch.device("cuda")
        with patch.object(model, "_run_net", side_effect=network, create=True) as run:
            model.eval([image, image], fast=True, compute_masks=False)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.kwargs["fast"] and call.kwargs["batch_size"] == 32
                                for call in run.call_args_list))
            model.eval(image, fast=True, normalize={"lowhigh": [0, 10]}, compute_masks=False)
            self.assertIsNone(run.call_args.kwargs["normalize_params"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fast backend")
    def test_tiles_resampling_cache_and_original_weights(self):
        net = ToyNet("cuda")
        parameter = net.conv.weight
        original = parameter.detach().clone()
        image = np.random.default_rng(12).uniform(0, 1, (2, 53, 71, 3)).astype(np.float32)
        untouched = image.copy()
        cache = {}
        for scale, resample in [(1., True), (1.5, True), (.75, False)]:
            actual, style = fast.run_net_fast(net, image, batch_size=3, bsize=32,
                rescale=scale, resample=resample, cache=cache)
            expected = torch.from_numpy(image.transpose(0, 3, 1, 2)).cuda()
            if scale != 1:
                expected = torch.nn.functional.interpolate(expected,
                    size=(int(53 * scale), int(71 * scale)), mode="bilinear", align_corners=False)
            expected = expected.half().float()
            if scale != 1 and resample:
                expected = torch.nn.functional.interpolate(expected, size=(53, 71),
                                                            mode="bilinear", align_corners=False)
            np.testing.assert_allclose(actual, expected.cpu().numpy().transpose(0, 2, 3, 1),
                                       atol=6e-4, rtol=1e-3)
            self.assertEqual(style.shape, (2, 256))
            self.assertFalse(style.any())
            self.assertIs(net.conv.weight, parameter)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.equal(parameter, original))
        np.testing.assert_array_equal(image, untouched)
        old_cache = cache["fp16_parameters"]
        with torch.no_grad():
            net.conv.weight.mul_(2)
        fast.run_net_fast(net, image[:1], batch_size=3, bsize=32, cache=cache)
        self.assertIsNot(cache["fp16_parameters"], old_cache)
        self.assertTrue(torch.equal(cache["fp16_parameters"]["conv.weight"], original.half() * 2))
        with patch.object(net, "forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                fast.run_net_fast(net, image[:1], bsize=32, cache=cache)
        self.assertIs(net.conv.weight, parameter)
        with torch.no_grad():
            parameter.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            fast.run_net_fast(net, image[:1], bsize=32, cache=cache)


if __name__ == "__main__":
    unittest.main()
