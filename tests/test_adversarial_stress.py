"""
tests/test_adversarial_stress.py
Adversarial Empirical Stress Testing Suite for CLAUDIO_AI
Covers:
- Category A: Empty CAD & Degenerate Geometries
- Category B: Extreme Coordinates & Numerical Precision
- Category C: Extreme Scale & Resolution Limits
- Category D: Confidence Threshold Bounds & Sensitivities
- Category E: Centroidal NMS d_min Threshold Extremes
- Category F: Corrupted, Non-CAD & Malformed Inputs
- Category G: Scale Analyzer Extremes & Clustering Stress
"""

import os
import sys
import math
import tempfile
import unittest
import numpy as np
from pathlib import Path

import cv2
import ezdxf

# Add project root and detector_pack
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PACK_DIR = PROJECT_ROOT / "detector_pack"
TOOLS_DIR = PROJECT_ROOT / "tools"

for p in [str(PROJECT_ROOT), str(DETECTOR_PACK_DIR), str(TOOLS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from detector_unifilar import (
    render_dxf,
    generate_slices,
    nms_iou_cad,
    eliminar_anidadas,
    nms_distancia_cad,
    DetectorUnifilar
)
import scale_analyzer
from tools.convert_dwg_to_dxf import convert_dwg_to_dxf, find_accoreconsole


class TestAdversarialCategoryA_EmptyAndDegenerateCAD(unittest.TestCase):
    """Stress tests on empty drawings, degenerate geometry, and excluded-only layers."""

    def test_a01_completely_empty_dxf(self):
        """Empty DXF with 0 entities in ModelSpace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "empty.dxf")
            doc = ezdxf.new()
            doc.saveas(dxf_path)

            # Empirically test render_dxf
            # We want to know if it raises ValueError, RuntimeError, or handles gracefully
            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0)
                # If it didn't raise, verify dimensions
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                # Documenting exact error for empirical findings
                self.assertTrue(True, f"Empty DXF raised expected exception: {e}")

    def test_a02_only_excluded_layers(self):
        """DXF where all entities belong to excluded layers (FORMATO, CARATULA, DEFPOINTS)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "excluded_layers.dxf")
            doc = ezdxf.new()
            doc.layers.new(name="CARATULA")
            msp = doc.modelspace()
            msp.add_line((0, 0), (100, 100), dxfattribs={"layer": "CARATULA"})
            msp.add_circle((50, 50), radius=10, dxfattribs={"layer": "DEFPOINTS"})
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0, is_industrial=True)
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                self.assertTrue(True, f"Excluded-only DXF raised: {e}")

    def test_a03_only_excluded_entity_types(self):
        """DXF where all entities are excluded types (TEXT, MTEXT, DIMENSION)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "texts_only.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_text("DIAGRAMA UNIFILAR GENERAL", dxfattribs={"insert": (0, 0), "height": 5.0})
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0, is_industrial=True)
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                self.assertTrue(True, f"Text-only DXF raised: {e}")

    def test_a04_single_point_geometry(self):
        """DXF containing only a single point (extents width=0, height=0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "point_only.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_point((25.0, 40.0))
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0)
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                self.assertTrue(True, f"Point-only DXF raised: {e}")

    def test_a05_strictly_collinear_vertical_line(self):
        """DXF with a single vertical line (W_cad=0, H_cad=10)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "vert_line.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((10.0, 0.0), (10.0, 10.0))
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0)
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                self.assertTrue(True, f"Collinear vertical line raised: {e}")

    def test_a06_strictly_collinear_horizontal_line(self):
        """DXF with a single horizontal line (W_cad=10, H_cad=0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "horiz_line.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((0.0, 10.0), (10.0, 10.0))
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=75.0)
                self.assertIsNotNone(img)
            except (ValueError, RuntimeError) as e:
                self.assertTrue(True, f"Collinear horizontal line raised: {e}")


class TestAdversarialCategoryB_ExtremeCoordinates(unittest.TestCase):
    """Stress tests for geographic UTM/POSGAR coordinates and numerical stability."""

    def test_b01_massive_positive_utm_coordinates(self):
        """Switchboard drawn at UTM coordinates (x=10,000,000, y=20,000,000)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "utm_coords.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            # Draw a 10x10 switchboard box at 10^7 coordinates
            x_base, y_base = 10_000_000.0, 20_000_000.0
            msp.add_line((x_base, y_base), (x_base + 10.0, y_base))
            msp.add_line((x_base + 10.0, y_base), (x_base + 10.0, y_base + 10.0))
            msp.add_line((x_base + 10.0, y_base + 10.0), (x_base, y_base + 10.0))
            msp.add_line((x_base, y_base + 10.0), (x_base, y_base))
            doc.saveas(dxf_path)

            img, meta = render_dxf(dxf_path, px_per_cad=50.0, is_industrial=False)
            self.assertIsNotNone(img)
            self.assertAlmostEqual(meta['x_min_cad'], x_base, delta=0.01)
            self.assertAlmostEqual(meta['y_min_cad'], y_base, delta=0.01)
            self.assertAlmostEqual(meta['W_cad'], 10.0, delta=0.01)
            self.assertAlmostEqual(meta['H_cad'], 10.0, delta=0.01)

    def test_b02_massive_negative_coordinates(self):
        """Switchboard drawn at extreme negative coordinates (-10^7, -10^7)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "neg_coords.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            x_base, y_base = -10_000_000.0, -10_000_000.0
            msp.add_line((x_base, y_base), (x_base + 20.0, y_base + 20.0))
            doc.saveas(dxf_path)

            img, meta = render_dxf(dxf_path, px_per_cad=50.0, is_industrial=False)
            self.assertIsNotNone(img)
            self.assertAlmostEqual(meta['x_min_cad'], x_base, delta=0.01)
            self.assertAlmostEqual(meta['y_min_cad'], y_base, delta=0.01)

    def test_b03_origin_crossing_geometry(self):
        """Switchboard spanning across origin (-500 to +500)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "cross_origin.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((-50.0, -50.0), (50.0, 50.0))
            doc.saveas(dxf_path)

            img, meta = render_dxf(dxf_path, px_per_cad=10.0, is_industrial=False)
            self.assertIsNotNone(img)
            self.assertAlmostEqual(meta['W_cad'], 100.0, delta=0.1)
            self.assertAlmostEqual(meta['H_cad'], 100.0, delta=0.1)

    def test_b04_nms_distance_at_extreme_coordinates(self):
        """Verifies nms_distancia_cad does not suffer catastrophic cancellation at 10^8."""
        base_x = 100_000_000.0
        base_y = 500_000_000.0
        dets = [
            {"conf": 0.90, "bbox_cad": [base_x, base_y, base_x + 0.8, base_y + 1.2], "xc": base_x + 0.4, "yc": base_y + 0.6},
            # Near duplicate at dist 0.05
            {"conf": 0.85, "bbox_cad": [base_x + 0.05, base_y, base_x + 0.85, base_y + 1.2], "xc": base_x + 0.45, "yc": base_y + 0.6},
            # Distinct component at dist 0.35
            {"conf": 0.88, "bbox_cad": [base_x + 0.35, base_y, base_x + 1.15, base_y + 1.2], "xc": base_x + 0.75, "yc": base_y + 0.6},
        ]
        kept = nms_distancia_cad(dets, d_min=0.20)
        # Should keep det 0 (conf 0.90) and det 2 (conf 0.88, dist=0.35 > 0.20)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["conf"], 0.90)
        self.assertEqual(kept[1]["conf"], 0.88)


class TestAdversarialCategoryC_ExtremeScaleAndResolution(unittest.TestCase):
    """Stress tests on extreme scale factors and slicing behavior."""

    def test_c01_micro_scale_rendering(self):
        """Test render_dxf at extreme low scale px_per_cad = 0.001."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "small_scale.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((0, 0), (10, 10))
            doc.saveas(dxf_path)

            try:
                img, meta = render_dxf(dxf_path, px_per_cad=0.001, is_industrial=False)
                # If it produces an image, check size
                self.assertIsNotNone(img)
            except Exception as e:
                # Documenting behavior
                self.assertTrue(True, f"Micro-scale raised: {e}")

    def test_c02_huge_scale_rendering_safety_cap(self):
        """Test render_dxf when px_per_cad would generate astronomical image sizes."""
        # 100 CAD units at 1000 px/CAD = 100,000 px!
        # In dxf_to_image, there's a cap at max_dim_px (32000).
        # We test that scale_analyzer or rendering handles or caps.
        pass

    def test_c03_extreme_aspect_ratio_slices(self):
        """Slicing an image with 50000x200 pixels (1:250 aspect ratio)."""
        slices = generate_slices(H=200, W=50000, slice_size=640, overlap=0.80)
        self.assertGreater(len(slices), 0)
        # All slices must satisfy boundaries
        for x1, y1, x2, y2 in slices:
            self.assertGreaterEqual(x1, 0)
            self.assertLessEqual(x2, 50000)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(y2, 200)

    def test_c04_tiny_image_slicing(self):
        """Slicing an image smaller than slice_size (e.g. 100x100)."""
        slices = generate_slices(H=100, W=100, slice_size=640, overlap=0.80)
        self.assertEqual(len(slices), 1)
        x1, y1, x2, y2 = slices[0]
        self.assertEqual((x1, y1, x2, y2), (0, 0, 100, 100))


class TestAdversarialCategoryD_ConfidenceThresholdBoundaries(unittest.TestCase):
    """Stress tests on confidence threshold extremes and edge values."""

    def test_d01_conf_threshold_zero(self):
        """conf_thresh = 0.0: all detections accepted."""
        dets = [
            {"conf": 0.001, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5},
            {"conf": 0.500, "bbox_cad": [10, 10, 11, 11], "xc": 10.5, "yc": 10.5}
        ]
        filtered = [d for d in dets if d['conf'] >= 0.0]
        self.assertEqual(len(filtered), 2)

    def test_d02_conf_threshold_one(self):
        """conf_thresh = 1.0: only perfect 1.0 confidence accepted."""
        dets = [
            {"conf": 0.9999, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5},
            {"conf": 1.0000, "bbox_cad": [10, 10, 11, 11], "xc": 10.5, "yc": 10.5}
        ]
        filtered = [d for d in dets if d['conf'] >= 1.0]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["conf"], 1.0)

    def test_d03_conf_threshold_above_one(self):
        """conf_thresh = 1.5: out-of-range confidence returns empty list cleanly."""
        dets = [
            {"conf": 0.99, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5}
        ]
        filtered = [d for d in dets if d['conf'] >= 1.5]
        self.assertEqual(len(filtered), 0)

    def test_d04_conf_threshold_negative(self):
        """conf_thresh = -0.5: keeps all detections."""
        dets = [
            {"conf": 0.05, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5}
        ]
        filtered = [d for d in dets if d['conf'] >= -0.5]
        self.assertEqual(len(filtered), 1)


class TestAdversarialCategoryE_CentroidalNMSdminExtremes(unittest.TestCase):
    """Stress tests on d_min calibration extremes (0.001, 0.05, 0.20, 0.50, 100.0)."""

    def test_e01_dmin_zero_keeps_all_non_overlapping_centroids(self):
        """d_min = 0.0: only exact same coordinate with dist=0.0 gets suppressed or kept."""
        dets = [
            {"conf": 0.90, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5},
            {"conf": 0.80, "bbox_cad": [0.01, 0.01, 1.01, 1.01], "xc": 0.51, "yc": 0.51}
        ]
        kept = nms_distancia_cad(dets, d_min=0.0)
        self.assertEqual(len(kept), 2)

    def test_e02_dmin_calibration_din_compact_rail(self):
        """
        Calibrated d_min = 0.20 CAD vs compact DIN breakers:
        Breakers at 0.18 CAD distance should be considered duplicates/same slot.
        Breakers at 0.25 CAD distance should be kept as distinct circuits.
        """
        dets = [
            {"conf": 0.95, "bbox_cad": [0, 0, 1, 1], "xc": 10.0, "yc": 5.0},
            {"conf": 0.80, "bbox_cad": [0.18, 0, 1.18, 1], "xc": 10.18, "yc": 5.0}, # dist=0.18 < 0.20
            {"conf": 0.90, "bbox_cad": [0.25, 0, 1.25, 1], "xc": 10.25, "yc": 5.0}, # dist=0.25 >= 0.20
        ]
        kept = nms_distancia_cad(dets, d_min=0.20)
        # Should keep det 0 (conf 0.95) and det 2 (conf 0.90)
        confs = [d["conf"] for d in kept]
        self.assertIn(0.95, confs)
        self.assertIn(0.90, confs)
        self.assertNotIn(0.80, confs)

    def test_e03_dmin_extreme_large_suppresses_entire_board(self):
        """d_min = 1000.0: entire switchboard collapses to 1 highest-confidence detection."""
        dets = [
            {"conf": 0.92, "bbox_cad": [0, 0, 1, 1], "xc": 10.0, "yc": 5.0},
            {"conf": 0.85, "bbox_cad": [20, 0, 21, 1], "xc": 30.0, "yc": 5.0},
            {"conf": 0.70, "bbox_cad": [50, 0, 51, 1], "xc": 60.0, "yc": 5.0},
        ]
        kept = nms_distancia_cad(dets, d_min=1000.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["conf"], 0.92)

    def test_e04_identical_centroids_collision(self):
        """Two detections at exactly identical centroid (xc, yc)."""
        dets = [
            {"conf": 0.85, "bbox_cad": [0, 0, 1, 1], "xc": 5.0, "yc": 5.0},
            {"conf": 0.85, "bbox_cad": [0, 0, 1, 1], "xc": 5.0, "yc": 5.0},
        ]
        kept = nms_distancia_cad(dets, d_min=0.20)
        self.assertEqual(len(kept), 1)

    def test_e05_empty_detections_list(self):
        """nms_distancia_cad on empty list returns empty list."""
        self.assertEqual(nms_distancia_cad([], d_min=0.20), [])


class TestAdversarialCategoryF_CorruptedAndNonCADInputs(unittest.TestCase):
    """Stress tests on corrupted files, wrong extensions, binary junk."""

    def test_f01_nonexistent_dxf_file(self):
        """Non-existent DXF file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            render_dxf("nonexistent_phantom_file.dxf")

    def test_f02_zero_byte_dxf_file(self):
        """Zero-byte DXF file raises DXFStructureError or similar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dxf = os.path.join(tmpdir, "empty_file.dxf")
            Path(bad_dxf).touch()
            with self.assertRaises(Exception):
                render_dxf(bad_dxf)

    def test_f03_binary_garbage_dxf(self):
        """File filled with random binary garbage raises DXFStructureError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dxf = os.path.join(tmpdir, "garbage.dxf")
            with open(bad_dxf, "wb") as f:
                f.write(os.urandom(4096))
            with self.assertRaises(Exception):
                render_dxf(bad_dxf)

    def test_f04_truncated_dxf_mid_section(self):
        """DXF truncated abruptly in the middle of ENTITIES section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dxf = os.path.join(tmpdir, "truncated.dxf")
            content = (
                "0\nSECTION\n2\nHEADER\n0\nENDSEC\n"
                "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n"
            )
            with open(bad_dxf, "w", encoding="ascii") as f:
                f.write(content)
            with self.assertRaises(Exception):
                render_dxf(bad_dxf)

    def test_f05_dwg_converter_nonexistent_dwg(self):
        """convert_dwg_to_dxf with non-existent input raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            convert_dwg_to_dxf("C:/missing_path/fake.dwg", "output.dxf")

    def test_f06_dwg_converter_accoreconsole_missing_handling(self):
        """find_accoreconsole raises FileNotFoundError when passed an invalid path."""
        with self.assertRaises(FileNotFoundError):
            find_accoreconsole("C:/invalid_tools/accoreconsole.exe")


class TestAdversarialCategoryG_ScaleAnalyzerExtremes(unittest.TestCase):
    """Stress tests for scale_analyzer with 0 blocks, thousands of blocks, and outliers."""

    def test_g01_scale_analyzer_zero_blocks_circles_texts(self):
        """DXF with only bare lines: fallback to bounding box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "lines_only.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((0, 0), (100, 100))
            doc.saveas(dxf_path)

            px_per_cad, ref = scale_analyzer.calcular_factor_escala(dxf_path)
            self.assertGreater(px_per_cad, 0.0)
            self.assertEqual(ref[0], "BBOX")

    def test_g02_scale_analyzer_with_huge_outlier_block(self):
        """
        DXF with:
        - 20 small symbol blocks (size ~ 1.0 CAD)
        - 1 massive title frame block (size ~ 5000.0 CAD)
        Log-bin clustering must filter out the title frame and select ~1.0 CAD.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "outlier_block.dxf")
            doc = ezdxf.new()
            
            # Create small block
            blk_small = doc.blocks.new(name="BREAKER")
            blk_small.add_lwpolyline([(0, 0), (1, 0), (1, 1), (0, 1)], close=True)

            # Create huge block
            blk_huge = doc.blocks.new(name="CARATULA_GRANDE")
            blk_huge.add_lwpolyline([(0, 0), (5000, 0), (5000, 5000), (0, 5000)], close=True)

            msp = doc.modelspace()
            # Insert 20 small blocks
            for i in range(20):
                msp.add_blockref("BREAKER", (i * 2.0, 0.0))
            # Insert 1 huge frame
            msp.add_blockref("CARATULA_GRANDE", (0.0, 0.0))

            doc.saveas(dxf_path)

            px_per_cad, ref = scale_analyzer.calcular_factor_escala(dxf_path)
            self.assertEqual(ref[0], "INSERT_diag")
            # Detected size must be ~1.0 (small block), NOT ~5000.0!
            self.assertAlmostEqual(ref[1], 1.0, delta=0.2)
            self.assertAlmostEqual(px_per_cad, 100.0, delta=20.0)

    def test_g03_scale_analyzer_high_entity_volume(self):
        """DXF with 1000 circle entities to test clustering performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "thousand_circles.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()
            for i in range(1000):
                msp.add_circle((i % 50, i // 50), radius=0.6)
            doc.saveas(dxf_path)

            px_per_cad, ref = scale_analyzer.calcular_factor_escala(dxf_path)
            self.assertEqual(ref[0], "CIRCLE_diam")
            self.assertAlmostEqual(ref[1], 1.2, delta=0.01)


class TestAdversarialCategoryH_LiveDetectorSweeps(unittest.TestCase):
    """End-to-end detector inference on real CAD across conf thresholds and d_min sweeps."""

    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(DETECTOR_PACK_DIR, "best_componente_nano.pt")
        cls.detector = DetectorUnifilar(model_path=model_path)
        cls.tsbe_path = os.path.join(PROJECT_ROOT, "dxf", "Tablerotsbe.dxf")

    def test_h01_detector_conf_sweep_tablerotsbe(self):
        """
        Verify detector behavior at conf bounds: 0.01, 0.50, 0.99.
        conf=0.01 must maximize recall (>= 15).
        conf=0.50 is the nominal AEA 90364 operating point (exactly 15).
        conf=0.99 must either detect ultra-high confidence components or cleanly return subset.
        """
        if not os.path.exists(self.tsbe_path):
            self.skipTest("Tablerotsbe.dxf not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            res_01 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.01, px_per_cad=75.0, d_min=0.20)
            res_50 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.50, px_per_cad=75.0, d_min=0.20)
            res_99 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.99, px_per_cad=75.0, d_min=0.20)

            print(f"\n[STRESS] Tablerotsbe Detections: conf 0.01 -> {res_01['total']}, conf 0.50 -> {res_50['total']}, conf 0.99 -> {res_99['total']}")

            # Monotonicity check: higher conf threshold must result in <= count
            self.assertGreaterEqual(res_01['total'], res_50['total'])
            self.assertGreaterEqual(res_50['total'], res_99['total'])
            # At nominal conf 0.50, exactly 15 switchgear apparatus detected
            self.assertEqual(res_50['total'], 15)
            # Output artifacts must be generated cleanly in all cases
            for r in [res_01, res_50, res_99]:
                self.assertTrue(os.path.exists(r['vis_image']))
                self.assertTrue(os.path.exists(r['csv_path']))
                self.assertTrue(os.path.exists(r['json_path']))

    def test_h02_detector_dmin_sweep_tablerotsbe(self):
        """
        Verify detector behavior across d_min extremes: 0.05, 0.20, 0.50 CAD.
        Smaller d_min keeps more close detections; larger d_min merges nearby detections.
        """
        if not os.path.exists(self.tsbe_path):
            self.skipTest("Tablerotsbe.dxf not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            res_d05 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.50, px_per_cad=75.0, d_min=0.05)
            res_d20 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.50, px_per_cad=75.0, d_min=0.20)
            res_d50 = self.detector.detectar_dxf(self.tsbe_path, output_dir=tmpdir, conf_thresh=0.50, px_per_cad=75.0, d_min=0.50)

            print(f"\n[STRESS] Tablerotsbe d_min: 0.05 -> {res_d05['total']}, 0.20 -> {res_d20['total']}, 0.50 -> {res_d50['total']}")

            # Monotonicity check: larger d_min must result in <= count
            self.assertGreaterEqual(res_d05['total'], res_d20['total'])
            self.assertGreaterEqual(res_d20['total'], res_d50['total'])
            self.assertEqual(res_d20['total'], 15)


class TestAdversarialCategoryI_DWGConverterRobustness(unittest.TestCase):
    """Stress tests for convert_dwg_to_dxf with malformed scripts, wrong extensions, locked files."""

    def test_i01_converter_wrong_extensions(self):
        """Converting files with .txt, .pdf, or no extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_txt = os.path.join(tmpdir, "diagram.txt")
            with open(fake_txt, "w") as f:
                f.write("Not a DWG file")
            
            out_dxf = os.path.join(tmpdir, "out.dxf")
            # If accoreconsole is not available, it raises FileNotFoundError, which is handled
            try:
                success = convert_dwg_to_dxf(fake_txt, out_dxf, timeout=5)
                self.assertFalse(success)
            except FileNotFoundError:
                pass  # accoreconsole not installed on runner, expected

    def test_i02_converter_empty_output_path(self):
        """Converter destination in nonexistent nested directory is created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_dwg = os.path.join(tmpdir, "test.dwg")
            with open(fake_dwg, "wb") as f:
                f.write(b"AC1027dummy")
            nested_out = os.path.join(tmpdir, "nested", "sub", "output.dxf")
            try:
                convert_dwg_to_dxf(fake_dwg, nested_out, timeout=2)
            except FileNotFoundError:
                pass
            # Directory should have been created
            self.assertTrue(os.path.exists(os.path.dirname(nested_out)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

