"""Check benchmark agreement metrics penalize missed and additional cells."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmark_fast import agreement


class AgreementTests(unittest.TestCase):
    def test_label_permutation_and_unmatched_cells(self):
        reference = np.zeros((16, 16), np.uint16)
        reference[1:5, 1:5] = 1
        reference[8:12, 8:12] = 2
        permutation = reference.copy()
        permutation[reference == 1], permutation[reference == 2] = 2, 1
        result = agreement(reference, permutation)
        self.assertFalse(result["bitwise_identical"])
        self.assertEqual(result["reference_object_mean_iou"], 1)
        self.assertEqual(result["prediction_object_mean_iou"], 1)
        missing = reference.copy()
        missing[missing == 2] = 0
        result = agreement(reference, missing)
        self.assertEqual(result["reference_object_mean_iou"], .5)
        self.assertEqual(result["prediction_object_mean_iou"], 1)
        self.assertEqual(result["unmatched_reference_cells"], 1)
        result = agreement(missing, reference)
        self.assertEqual(result["prediction_object_mean_iou"], .5)
        self.assertEqual(result["unmatched_prediction_cells"], 1)

    def test_empty_and_disjoint_masks(self):
        empty = np.zeros((16, 16), np.uint16)
        self.assertEqual(agreement(empty, empty)["foreground_iou"], 1)
        mask = empty.copy()
        mask[1:5, 1:5] = 1
        for ref, pred in [(mask, empty), (empty, mask), (mask, np.flip(mask).copy())]:
            result = agreement(ref, pred)
            self.assertEqual(result["foreground_iou"], 0)
            self.assertEqual(result["reference_object_mean_iou"], 0)
            self.assertEqual(result["prediction_object_mean_iou"], 0)


if __name__ == "__main__":
    unittest.main()
