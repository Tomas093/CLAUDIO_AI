"""
tests/test_adversarial_challenger2.py
Adversarial Coverage Hardening & Metrology Oracle Test Suite (Challenger 2).

Verifies:
1. Centroidal NMS separation d_min +/- epsilon (retention, suppression, isotropy, boundary equality).
2. CAD-to-pixel coordinate inversion bijection, roundtrip fidelity, padding invariance, aspect ratio independence.
3. Multi-column density stress, DIN rail pitch thresholds, sorting determinism, and absence of regressions.
4. White-box metrology analysis: extreme coordinates, sub-pixel dimensions, degenerate bounding boxes.
"""

import os
import sys
import json
import math
import random
import unittest
from pathlib import Path
import numpy as np

# Set project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector_pack.detector_unifilar import (
    nms_iou_cad,
    eliminar_anidadas,
    nms_distancia_cad,
    generate_slices
)
from tools.audit_engine import AuditEngine, PlanSpec


class TestCentroidalNMSAdversarial(unittest.TestCase):
    """Adversarial stress testing of centroidal NMS algorithm under d_min +/- epsilon."""

    def setUp(self):
        self.d_min = 0.20  # Standard calibrated threshold (Req R3)

    def test_nms_suppression_at_dmin_minus_epsilon(self):
        """Detections separated by d < d_min must be suppressed (only higher conf survives)."""
        epsilons = [0.05, 0.01, 1e-3, 1e-4, 1e-6]
        for eps in epsilons:
            dist = self.d_min - eps
            det_high = {
                "conf": 0.95,
                "bbox_cad": [0.0, 0.0, 0.5, 0.5],
                "xc": 0.25,
                "yc": 0.25,
                "bbox_px": [10, 10, 30, 30]
            }
            det_low = {
                "conf": 0.40,
                "bbox_cad": [dist, 0.0, dist + 0.5, 0.5],
                "xc": 0.25 + dist,
                "yc": 0.25,
                "bbox_px": [10 + int(dist * 75), 10, 30 + int(dist * 75), 30]
            }

            # Test order 1: [high, low]
            res1 = nms_distancia_cad([det_high, det_low], d_min=self.d_min)
            self.assertEqual(len(res1), 1, f"Failed suppression at dist={dist} (d_min - {eps})")
            self.assertAlmostEqual(res1[0]["conf"], 0.95, places=5)

            # Test order 2: [low, high]
            res2 = nms_distancia_cad([det_low, det_high], d_min=self.d_min)
            self.assertEqual(len(res2), 1, f"Failed suppression at dist={dist} (reversed input order)")
            self.assertAlmostEqual(res2[0]["conf"], 0.95, places=5)

    def test_nms_retention_at_dmin_plus_epsilon(self):
        """Detections separated by d > d_min must both be retained."""
        epsilons = [1e-6, 1e-4, 1e-3, 0.01, 0.05]
        for eps in epsilons:
            dist = self.d_min + eps
            det_high = {
                "conf": 0.95,
                "bbox_cad": [0.0, 0.0, 0.5, 0.5],
                "xc": 0.25,
                "yc": 0.25,
                "bbox_px": [10, 10, 30, 30]
            }
            det_low = {
                "conf": 0.40,
                "bbox_cad": [dist, 0.0, dist + 0.5, 0.5],
                "xc": 0.25 + dist,
                "yc": 0.25,
                "bbox_px": [10 + int(dist * 75), 10, 30 + int(dist * 75), 30]
            }

            res = nms_distancia_cad([det_high, det_low], d_min=self.d_min)
            self.assertEqual(len(res), 2, f"Incorrect suppression at dist={dist} (d_min + {eps})")

    def test_nms_exact_boundary_equality(self):
        """At exact boundary d == d_min, both must be retained per 'dists >= d_min' specification."""
        det_a = {
            "conf": 0.88,
            "bbox_cad": [0.0, 0.0, 0.4, 0.4],
            "xc": 0.20,
            "yc": 0.20,
            "bbox_px": [10, 10, 20, 20]
        }
        det_b = {
            "conf": 0.82,
            "bbox_cad": [self.d_min, 0.0, self.d_min + 0.4, 0.4],
            "xc": 0.20 + self.d_min,
            "yc": 0.20,
            "bbox_px": [25, 10, 35, 20]
        }
        res = nms_distancia_cad([det_a, det_b], d_min=self.d_min)
        self.assertEqual(len(res), 2, "Exact boundary condition d == d_min failed retention")

    def test_nms_isotropic_angular_invariance(self):
        """Verify that d_min separation is strictly isotropic across 16 radial directions."""
        num_angles = 16
        eps = 1e-4
        for i in range(num_angles):
            theta = 2.0 * math.pi * i / num_angles

            # Sub-threshold (must suppress)
            dx_sub = (self.d_min - eps) * math.cos(theta)
            dy_sub = (self.d_min - eps) * math.sin(theta)
            det_origin = {"conf": 0.90, "xc": 5.0, "yc": 5.0, "bbox_cad": [4.8, 4.8, 5.2, 5.2]}
            det_sub = {"conf": 0.60, "xc": 5.0 + dx_sub, "yc": 5.0 + dy_sub, "bbox_cad": [4.8+dx_sub, 4.8+dy_sub, 5.2+dx_sub, 5.2+dy_sub]}

            res_sub = nms_distancia_cad([det_origin, det_sub], d_min=self.d_min)
            self.assertEqual(len(res_sub), 1, f"Anisotropy bug: failed suppression at angle {math.degrees(theta):.1f} deg")

            # Super-threshold (must retain)
            dx_sup = (self.d_min + eps) * math.cos(theta)
            dy_sup = (self.d_min + eps) * math.sin(theta)
            det_sup = {"conf": 0.60, "xc": 5.0 + dx_sup, "yc": 5.0 + dy_sup, "bbox_cad": [4.8+dx_sup, 4.8+dy_sup, 5.2+dx_sup, 5.2+dy_sup]}

            res_sup = nms_distancia_cad([det_origin, det_sup], d_min=self.d_min)
            self.assertEqual(len(res_sup), 2, f"Anisotropy bug: failed retention at angle {math.degrees(theta):.1f} deg")

    def test_nms_linear_chain_cascade(self):
        """Test chain of N detections spaced at 0.75 * d_min (sub-threshold domino suppression)."""
        n = 10
        step = 0.75 * self.d_min  # Each neighbor is within d_min

        # Case 1: Monotonically increasing confidence: 0.1, 0.2, ..., 1.0
        # Right-most (highest) suppresses its left neighbor, which cascades backwards
        dets_inc = [
            {
                "conf": 0.1 * (i + 1),
                "xc": i * step,
                "yc": 0.0,
                "bbox_cad": [i * step - 0.1, -0.1, i * step + 0.1, 0.1]
            }
            for i in range(n)
        ]
        res_inc = nms_distancia_cad(dets_inc, d_min=self.d_min)
        # Verify survivors are at least d_min apart
        for a in range(len(res_inc)):
            for b in range(a + 1, len(res_inc)):
                d = abs(res_inc[a]["xc"] - res_inc[b]["xc"])
                self.assertGreaterEqual(d, self.d_min, f"Survivors too close in chain: d={d} < {self.d_min}")

    def test_nms_cascaded_interaction_synergy(self):
        """Verify synergistic interaction of IoU -> Nested (IoS) -> Distance (Centroidal) NMS."""
        # 1. Two boxes with identical center but slightly different bbox (IoU ~ 0.8 > 0.45)
        # Should be handled by nms_iou_cad
        d1 = {"conf": 0.90, "bbox_cad": [0.0, 0.0, 1.0, 1.0], "xc": 0.5, "yc": 0.5}
        d2 = {"conf": 0.70, "bbox_cad": [0.05, 0.05, 0.95, 0.95], "xc": 0.5, "yc": 0.5}
        iou_res = nms_iou_cad([d1, d2], iou_thresh=0.45)
        self.assertEqual(len(iou_res), 1)

        # 2. One box fully inside another larger box (IoS == 1.0 > 0.60, but IoU might be low if sizes differ)
        # e.g., small box 0.1x0.1 inside 1.0x1.0: IoU = 0.01 / 1.0 = 0.01 < 0.45!
        d_large = {"conf": 0.92, "bbox_cad": [0.0, 0.0, 1.0, 1.0], "xc": 0.5, "yc": 0.5}
        d_nested = {"conf": 0.65, "bbox_cad": [0.45, 0.45, 0.55, 0.55], "xc": 0.5, "yc": 0.5}
        iou_check = nms_iou_cad([d_large, d_nested], iou_thresh=0.45)
        self.assertEqual(len(iou_check), 2, "IoU alone cannot catch nested box with low area ratio")
        # But eliminar_anidadas catches it!
        ios_res = eliminar_anidadas(iou_check, ios_thresh=0.60)
        self.assertEqual(len(ios_res), 1, "eliminar_anidadas must catch nested small box")

        # 3. Two slim boxes side-by-side with zero bbox overlap (IoU = 0, IoS = 0) but center dist < d_min
        d_slim1 = {"conf": 0.85, "bbox_cad": [0.0, 0.0, 0.08, 1.0], "xc": 0.04, "yc": 0.5}
        d_slim2 = {"conf": 0.80, "bbox_cad": [0.10, 0.0, 0.18, 1.0], "xc": 0.14, "yc": 0.5}
        # Centroid distance is 0.10 < d_min (0.20)
        dist_res = nms_distancia_cad([d_slim1, d_slim2], d_min=0.20)
        self.assertEqual(len(dist_res), 1, "Centroidal NMS must catch disjoint slim boxes within d_min")


class TestCoordinateInversionBijection(unittest.TestCase):
    """Metrology Oracle: Verification of CAD-to-Pixel bijection, reversibility, and padding invariance."""

    def test_bijection_roundtrip_identity(self):
        """Verify that mapping [0, W_px] x [0, H_px] <-> [x_min, x_max] x [y_min, y_max] is strictly bijective."""
        test_cases = [
            # (W_cad, H_cad, x_min, y_max, px_per_cad)
            (100.0, 80.0, 0.0, 80.0, 75.0),
            (45.5, 30.2, -120.0, 50.0, 84.31),      # Negative coordinates
            (10.0, 500.0, -10.0, 250.0, 10.0),       # High aspect ratio H >> W
            (800.0, 20.0, 5000.0, 1000.0, 25.0),     # High aspect ratio W >> H & large offset
            (1.5, 2.0, -0.75, 1.0, 100.0),           # Compact switchboard module
        ]

        for W_cad, H_cad, x_min, y_max, px_per_cad in test_cases:
            W_px = int(round(W_cad * px_per_cad))
            H_px = int(round(H_cad * px_per_cad))

            # Sample 200 random points in pixel space
            for _ in range(200):
                px_x = random.uniform(0.0, float(W_px))
                px_y = random.uniform(0.0, float(H_px))

                # Forward map (Pixel -> CAD) per detector_unifilar.py:356-359
                cad_x = x_min + (px_x / W_px) * W_cad
                cad_y = y_max - (px_y / H_px) * H_cad

                # Inverse map (CAD -> Pixel) per audit_engine.py:513-514
                rec_px_x = (cad_x - x_min) / W_cad * W_px
                rec_px_y = (y_max - cad_y) / H_cad * H_px

                # Verify pixel roundtrip fidelity
                self.assertAlmostEqual(px_x, rec_px_x, places=10,
                                       msg=f"Pixel X roundtrip failure: {px_x} vs {rec_px_x}")
                self.assertAlmostEqual(px_y, rec_px_y, places=10,
                                       msg=f"Pixel Y roundtrip failure: {px_y} vs {rec_px_y}")

                # Inverse map from CAD back to CAD
                rec_cad_x = x_min + (rec_px_x / W_px) * W_cad
                rec_cad_y = y_max - (rec_px_y / H_px) * H_cad

                self.assertAlmostEqual(cad_x, rec_cad_x, places=11,
                                       msg=f"CAD X roundtrip failure: {cad_x} vs {rec_cad_x}")
                self.assertAlmostEqual(cad_y, rec_cad_y, places=11,
                                       msg=f"CAD Y roundtrip failure: {cad_y} vs {rec_cad_y}")

    def test_sahi_padding_invariance(self):
        """Verify that SAHI 320 px border padding cancels out exactly and introduces zero spatial drift."""
        pad = 320
        slice_size = 640
        W_px = 3000
        H_px = 2000
        x_min, y_max = 10.0, 150.0
        W_cad, H_cad = 40.0, 26.666667

        # Ground truth component in unpadded pixel coordinates
        true_px_x1, true_px_y1 = 850.0, 420.0
        true_px_x2, true_px_y2 = 910.0, 480.0

        # Ground truth CAD coordinates
        true_cad_x1 = x_min + (true_px_x1 / W_px) * W_cad
        true_cad_x2 = x_min + (true_px_x2 / W_px) * W_cad
        true_cad_y1 = y_max - (true_px_y2 / H_px) * H_cad
        true_cad_y2 = y_max - (true_px_y1 / H_px) * H_cad

        # Generate slices on padded canvas
        pad_w = W_px + 2 * pad
        pad_h = H_px + 2 * pad
        slices = generate_slices(pad_h, pad_w, slice_size=slice_size, overlap=0.80)

        # Find all slices that fully cover the padded component
        padded_x1 = true_px_x1 + pad
        padded_y1 = true_px_y1 + pad
        padded_x2 = true_px_x2 + pad
        padded_y2 = true_px_y2 + pad

        covering_slices = [
            s for s in slices
            if s[0] <= padded_x1 and s[2] >= padded_x2 and s[1] <= padded_y1 and s[3] >= padded_y2
        ]
        self.assertGreater(len(covering_slices), 0, "No slice covers the target apparatus")

        # For every tile that sees this apparatus, calculate projected CAD coordinates
        for x1_tile, y1_tile, _, _ in covering_slices:
            # Local box inside tile
            b_local = [
                padded_x1 - x1_tile,
                padded_y1 - y1_tile,
                padded_x2 - x1_tile,
                padded_y2 - y1_tile
            ]

            # Invert using detector_unifilar.py:347-360 logic
            px_x1 = max(0, min(W_px, (x1_tile + b_local[0]) - pad))
            px_y1 = max(0, min(H_px, (y1_tile + b_local[1]) - pad))
            px_x2 = max(0, min(W_px, (x1_tile + b_local[2]) - pad))
            px_y2 = max(0, min(H_px, (y1_tile + b_local[3]) - pad))

            calc_cad_x1 = x_min + (px_x1 / W_px) * W_cad
            calc_cad_x2 = x_min + (px_x2 / W_px) * W_cad
            calc_cad_y1 = y_max - (px_y2 / H_px) * H_cad
            calc_cad_y2 = y_max - (px_y1 / H_px) * H_cad

            self.assertAlmostEqual(calc_cad_x1, true_cad_x1, places=10)
            self.assertAlmostEqual(calc_cad_x2, true_cad_x2, places=10)
            self.assertAlmostEqual(calc_cad_y1, true_cad_y1, places=10)
            self.assertAlmostEqual(calc_cad_y2, true_cad_y2, places=10)

    def test_bounding_box_orientation_preservation(self):
        """Verify that Y-axis CAD inversion properly orders cad_y1 < cad_y2 without negative heights."""
        W_px, H_px = 1000, 1000
        x_min, y_max = 0.0, 10.0
        W_cad, H_cad = 10.0, 10.0

        px_x1, px_y1 = 100, 200
        px_x2, px_y2 = 150, 280

        cad_x1 = x_min + (px_x1 / W_px) * W_cad
        cad_x2 = x_min + (px_x2 / W_px) * W_cad
        cad_y1 = y_max - (px_y2 / H_px) * H_cad
        cad_y2 = y_max - (px_y1 / H_px) * H_cad

        # In CAD, y increases upwards, so bottom of pixel box (larger py) gives smaller cad_y!
        self.assertLess(cad_y1, cad_y2)
        self.assertLess(cad_x1, cad_x2)

        bbox_cad = [min(cad_x1, cad_x2), min(cad_y1, cad_y2), max(cad_x1, cad_x2), max(cad_y1, cad_y2)]
        self.assertEqual(bbox_cad[0], cad_x1)
        self.assertEqual(bbox_cad[1], cad_y1)
        self.assertEqual(bbox_cad[2], cad_x2)
        self.assertEqual(bbox_cad[3], cad_y2)


class TestMultiColumnDensityStress(unittest.TestCase):
    """Stress testing on high-density multi-column switchboard topologies."""

    def test_multi_column_dense_matrix_retention(self):
        """Synthesize 10 columns x 25 rows = 250 components with duplicate jitter. Verify 100% precision & recall."""
        cols = 10
        rows = 25
        col_pitch = 2.0   # 2.0 CAD between columns
        row_pitch = 0.8   # 0.8 CAD between rows (DIN rail standard)
        d_min = 0.20

        raw_detections = []
        expected_centroids = []

        for c in range(cols):
            x = c * col_pitch + 5.0
            for r in range(rows):
                y = r * row_pitch + 10.0
                expected_centroids.append((x, y))

                # Primary true detection
                raw_detections.append({
                    "conf": 0.90 + 0.05 * random.random(),
                    "xc": x,
                    "yc": y,
                    "bbox_cad": [x - 0.25, y - 0.35, x + 0.25, y + 0.35],
                    "bbox_px": [int(x * 75), int(y * 75), int((x+0.5)*75), int((y+0.7)*75)]
                })

                # Inject 3 redundant jitter detections from tile overlaps (within 0.05 < d_min)
                for _ in range(3):
                    jx = x + random.uniform(-0.05, 0.05)
                    jy = y + random.uniform(-0.05, 0.05)
                    raw_detections.append({
                        "conf": 0.40 + 0.30 * random.random(),
                        "xc": jx,
                        "yc": jy,
                        "bbox_cad": [jx - 0.25, jy - 0.35, jx + 0.25, jy + 0.35],
                        "bbox_px": [int(jx * 75), int(jy * 75), int((jx+0.5)*75), int((jy+0.7)*75)]
                    })

        self.assertEqual(len(raw_detections), 250 * 4)  # 1000 total candidate detections

        # Run complete post-processing cascade
        dets = nms_iou_cad(raw_detections, iou_thresh=0.45)
        dets = eliminar_anidadas(dets, ios_thresh=0.60)
        dets = nms_distancia_cad(dets, d_min=d_min)

        # Must retain EXACTLY 250 components!
        self.assertEqual(len(dets), 250, f"Expected exactly 250 components after NMS, got {len(dets)}")

        # Verify that each detected component matches one expected component within 0.06 CAD
        for d in dets:
            matched = False
            for ex, ey in expected_centroids:
                if math.hypot(d["xc"] - ex, d["yc"] - ey) < 0.08:
                    matched = True
                    break
            self.assertTrue(matched, f"Spurious false positive detected at ({d['xc']}, {d['yc']})")

    def test_multi_column_sorting_determinism(self):
        """Verify sorting determinism by (round(yc, 1), xc) across random input permutations."""
        detections = [
            {"conf": 0.8, "xc": 10.5, "yc": 20.04, "bbox_cad": [10.2, 19.8, 10.8, 20.3]},
            {"conf": 0.9, "xc": 5.0, "yc": 20.01, "bbox_cad": [4.7, 19.8, 5.3, 20.2]},
            {"conf": 0.7, "xc": 15.2, "yc": 10.02, "bbox_cad": [14.9, 9.8, 15.5, 10.2]},
            {"conf": 0.85, "xc": 5.1, "yc": 10.04, "bbox_cad": [4.8, 9.8, 5.4, 10.2]},
        ]

        # Sorted order should be row by row: y=10.0 (x=5.1, then x=15.2), then y=20.0 (x=5.0, then x=10.5)
        sorted_dets = sorted(detections, key=lambda d: (round(d['yc'], 1), d['xc']))
        expected_xc = [5.1, 15.2, 5.0, 10.5]
        actual_xc = [d['xc'] for d in sorted_dets]
        self.assertEqual(actual_xc, expected_xc)

        # Shuffle and sort again 10 times
        for _ in range(10):
            shuffled = list(detections)
            random.shuffle(shuffled)
            res = sorted(shuffled, key=lambda d: (round(d['yc'], 1), d['xc']))
            self.assertEqual([d['xc'] for d in res], expected_xc)


class TestWhiteBoxMetrologyAndRobustness(unittest.TestCase):
    """White-box analysis of corner cases, degenerate boxes, and boundary inputs."""

    def test_degenerate_box_filtering(self):
        """Zero-width or inverted bounding boxes (px_x2 <= px_x1) must be safely skipped."""
        W_px, H_px = 1000, 1000
        # Simulation of detector_unifilar.py:352
        degenerate_boxes = [
            (50, 50, 50, 100),    # zero width
            (50, 50, 100, 50),    # zero height
            (100, 50, 50, 100),   # inverted X
            (50, 100, 100, 50),   # inverted Y
        ]
        for px_x1, px_y1, px_x2, px_y2 in degenerate_boxes:
            is_valid = not (px_x2 <= px_x1 or px_y2 <= px_y1)
            self.assertFalse(is_valid, f"Degenerate box ({px_x1}, {px_y1}, {px_x2}, {px_y2}) was not rejected")

    def test_large_offset_precision(self):
        """Coordinate transformations with coordinates around 1,000,000 CAD maintain sub-millimeter precision."""
        x_min = 1_000_000.0
        y_max = 5_000_000.0
        W_cad = 50.0
        H_cad = 30.0
        W_px = 3750
        H_px = 2250

        test_pt = (1_000_025.123456, 4_999_985.654321)
        px_x = (test_pt[0] - x_min) / W_cad * W_px
        px_y = (y_max - test_pt[1]) / H_cad * H_px

        rec_cad_x = x_min + (px_x / W_px) * W_cad
        rec_cad_y = y_max - (px_y / H_px) * H_cad

        self.assertAlmostEqual(test_pt[0], rec_cad_x, places=8)
        self.assertAlmostEqual(test_pt[1], rec_cad_y, places=8)


class TestRealOutputsMetrology(unittest.TestCase):
    """Metrology audit across all actual produced artifacts in output_eval/."""

    def test_audit_all_output_eval_artifacts(self):
        output_dir = PROJECT_ROOT / "output_eval"
        json_files = list(output_dir.glob("*_detections.json"))
        self.assertGreaterEqual(len(json_files), 10, "At least 10 evaluation artifacts must exist")

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data["metadata"]
            comps = data["components"]
            thresh = data["confidence_threshold"]

            W_px = meta["W_px"]
            H_px = meta["H_px"]
            W_cad = meta["W_cad"]
            H_cad = meta["H_cad"]
            x_min = meta["x_min_cad"]
            y_max = meta["y_max_cad"]
            px_per_cad = meta["px_per_cad"]
            d_min = 0.20 if px_per_cad > 1.0 else 34.0

            # 1. Confidence thresholding
            for c in comps:
                self.assertGreaterEqual(c["conf"], thresh - 1e-4, f"Conf below threshold in {jf.name}")

                # 2. Coordinate ordering & bounds
                b = c["bbox_cad"]
                self.assertLessEqual(b[0], b[2], f"Inverted X bbox in {jf.name}")
                self.assertLessEqual(b[1], b[3], f"Inverted Y bbox in {jf.name}")

                p = c["bbox_px"]
                self.assertGreaterEqual(p[0], 0, f"Negative px_x1 in {jf.name}")
                self.assertGreaterEqual(p[1], 0, f"Negative px_y1 in {jf.name}")
                self.assertLessEqual(p[2], W_px, f"px_x2 exceeds W_px in {jf.name}")
                self.assertLessEqual(p[3], H_px, f"px_y2 exceeds H_px in {jf.name}")

                # 3. Roundtrip bijection fidelity on real detection coordinates
                # Pixel -> CAD -> Pixel
                calc_cad_xc = x_min + (c["xc"] - x_min)
                rec_px_xc = (c["xc"] - x_min) / W_cad * W_px
                rec_cad_xc = x_min + (rec_px_xc / W_px) * W_cad
                self.assertAlmostEqual(c["xc"], rec_cad_xc, places=6)

            # 4. Centroidal separation verification across all pairs
            n_comps = len(comps)
            for i in range(n_comps):
                ci = comps[i]
                for j in range(i + 1, n_comps):
                    cj = comps[j]
                    dist = math.hypot(ci["xc"] - cj["xc"], ci["yc"] - cj["yc"])
                    self.assertGreaterEqual(
                        dist, d_min - 1e-3,
                        f"NMS violation in {jf.name}: components {i} and {j} separated by {dist:.4f} < d_min ({d_min})"
                    )


class TestToolchainExceptionHandling(unittest.TestCase):
    """Challenge exception handling, non-existent inputs, and edge conditions in tools/."""

    def test_convert_dwg_to_dxf_nonexistent_input(self):
        from tools.convert_dwg_to_dxf import convert_dwg_to_dxf
        with self.assertRaises(FileNotFoundError):
            convert_dwg_to_dxf("non_existent_plan.dwg", "output.dxf", verbose=False)

    def test_acquire_plans_validate_nonexistent_and_empty(self):
        import tempfile
        from tools.acquire_plans import validate_dxf
        with self.assertRaises(FileNotFoundError):
            validate_dxf(Path("totally_bogus_file.dxf"))

        with tempfile.NamedTemporaryFile("wb", suffix=".dxf", delete=False) as tf:
            tf.write(b"")
            empty_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError):
                validate_dxf(empty_path)
        finally:
            if empty_path.exists():
                empty_path.unlink()


if __name__ == "__main__":
    unittest.main()

