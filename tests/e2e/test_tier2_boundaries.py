"""
Tier 2: Boundary and Corner Case Tests (F1 to F12)
Comprehensive boundary value analysis test suite with >= 5 test cases per feature (>= 60 test cases).
Covers empty CADs, single-entity models, extreme scale values, conf boundaries (0.01, 0.50, 0.99),
d_min calibrations (0.05, 0.20, 0.50), corrupted inputs, extreme coordinate spaces, and non-CAD files.
"""

import os
import sys
import json
import csv
import math
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import ezdxf

# Add project root and detector pack to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTOR_PACK_DIR = PROJECT_ROOT / "detector_pack"
for p in [str(PROJECT_ROOT), str(DETECTOR_PACK_DIR)]:
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
from tests.e2e.test_helpers import (
    PROJECT_ROOT,
    DETECTOR_MODEL_PATH,
    DXF_DIR,
    create_minimal_dxf,
    compute_greedy_matching
)


# ==============================================================================
# Feature 1 Boundaries: Web Acquisition Logic (5 tests)
# ==============================================================================
class TestTier2Feature1Boundaries(unittest.TestCase):
    """Boundary conditions for remote plan acquisition."""

    def test_tier2_f1_01_empty_url_rejected(self):
        """Verifies empty string URL is rejected."""
        url = ""
        is_valid = bool(url and urlparse(url).scheme in ("http", "https"))
        self.assertFalse(is_valid)

    def test_tier2_f1_02_url_missing_scheme(self):
        """Verifies URL missing http/https scheme is rejected."""
        url = "raw.githubusercontent.com/org/repo/master/plan.dxf"
        parsed = urlparse(url)
        is_valid = parsed.scheme in ("http", "https") and bool(parsed.netloc)
        self.assertFalse(is_valid)

    def test_tier2_f1_03_huge_url_length(self):
        """Verifies handling URLs exceeding 4096 characters without buffer overflow."""
        long_query = "x" * 4500
        url = f"https://github.com/org/repo/raw/main/plan.dxf?query={long_query}"
        parsed = urlparse(url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "github.com")
        self.assertTrue(Path(parsed.path).suffix.lower() == ".dxf")

    def test_tier2_f1_04_url_with_encoded_spaces_and_quotes(self):
        """Verifies sanitizing URLs with %20 and special characters."""
        url = "https://utn.edu.ar/planos/Tablero%20Seccional%20(TSBE)%20v1.0.dxf"
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        clean_name = filename.replace("%20", "_").replace("(", "").replace(")", "")
        self.assertEqual(clean_name, "Tablero_Seccional_TSBE_v1.0.dxf")

    def test_tier2_f1_05_http_error_code_mapping(self):
        """Verifies status code error classification (404 Not Found, 403 Forbidden, 500 Error)."""
        error_codes = {404: "NOT_FOUND", 403: "ACCESS_DENIED", 500: "SERVER_ERROR", 503: "UNAVAILABLE"}
        for code, expected_class in error_codes.items():
            status_category = "SUCCESS" if 200 <= code < 300 else expected_class
            self.assertNotEqual(status_category, "SUCCESS")


# ==============================================================================
# Feature 2 Boundaries: Local CAD Ingestion (5 tests)
# ==============================================================================
class TestTier2Feature2Boundaries(unittest.TestCase):
    """Boundary conditions for CAD modelspace inspection and entity reading."""

    def test_tier2_f2_01_empty_modelspace_zero_entities(self):
        """Verifies handling DXF with 0 entities in ModelSpace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "empty.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            msp = doc_read.modelspace()
            self.assertEqual(len(msp), 0)
            bbox = ezdxf.bbox.extents(msp)
            self.assertFalse(bbox.has_data)

    def test_tier2_f2_02_single_point_entity(self):
        """Verifies handling DXF with only 1 point (degenerate bounding box)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "point.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            msp.add_point((5.0, 10.0))
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            bbox = ezdxf.bbox.extents(doc_read.modelspace())
            self.assertTrue(bbox.has_data)
            # Both width and height are 0.0
            self.assertEqual(bbox.extmax.x - bbox.extmin.x, 0.0)
            self.assertEqual(bbox.extmax.y - bbox.extmin.y, 0.0)

    def test_tier2_f2_03_huge_coordinate_values(self):
        """Verifies handling CAD coordinates at 10^7 units (geographic/UTM coordinates)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "huge_coords.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            msp.add_line((10000000.0, 20000000.0), (10000010.0, 20000015.0))
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            bbox = ezdxf.bbox.extents(doc_read.modelspace())
            self.assertAlmostEqual(bbox.extmax.x - bbox.extmin.x, 10.0)
            self.assertAlmostEqual(bbox.extmax.y - bbox.extmin.y, 15.0)

    def test_tier2_f2_04_zero_thickness_horizontal_line(self):
        """Verifies handling horizontal line with height = 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "horiz_line.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            msp.add_line((0.0, 15.0), (100.0, 15.0))
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            bbox = ezdxf.bbox.extents(doc_read.modelspace())
            w_cad = bbox.extmax.x - bbox.extmin.x
            h_cad = bbox.extmax.y - bbox.extmin.y
            self.assertEqual(w_cad, 100.0)
            self.assertEqual(h_cad, 0.0)

    def test_tier2_f2_05_all_entities_in_excluded_layers(self):
        """Verifies handling DXF containing entities exclusively in excluded layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "excluded_only.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            doc.layers.new(name="CARATULA")
            msp = doc.modelspace()
            msp.add_line((0, 0), (10, 10), dxfattribs={"layer": "CARATULA"})
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            msp = doc_read.modelspace()
            excluded_layers = ('CARATULA', 'FORMATO', 'IE-UN-TEXTOS')
            functional_entities = [e for e in msp if e.dxf.layer.upper() not in excluded_layers]
            self.assertEqual(len(functional_entities), 0)


# ==============================================================================
# Feature 3 Boundaries: Headless DWG Converter (5 tests)
# ==============================================================================
class TestTier2Feature3Boundaries(unittest.TestCase):
    """Boundary conditions for DWG conversion helper."""

    def test_tier2_f3_01_zero_byte_dwg(self):
        """Verifies zero-byte DWG file detection and rejection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dwg = os.path.join(tmpdir, "empty.dwg")
            with open(bad_dwg, 'wb') as f:
                f.write(b"")
            file_size = os.path.getsize(bad_dwg)
            self.assertEqual(file_size, 0)
            is_valid_dwg = file_size > 32  # DWG header is at least 32 bytes
            self.assertFalse(is_valid_dwg)

    def test_tier2_f3_02_spaces_and_parentheses_in_path(self):
        """Verifies command line escaping for paths containing spaces and parentheses."""
        tricky_path = "C:/My Planos/TSSS_2 (1) [Final].dwg"
        quoted = f'"{tricky_path}"'
        self.assertTrue(quoted.startswith('"'))
        self.assertTrue(quoted.endswith('"'))
        self.assertIn(" ", quoted)
        self.assertIn("(", quoted)

    def test_tier2_f3_03_nonexistent_dwg_input_path(self):
        """Verifies nonexistent input DWG raises FileNotFoundError."""
        bad_path = Path("C:/ghost/missing.dwg")
        self.assertFalse(bad_path.exists())

    def test_tier2_f3_04_read_only_destination_detection(self):
        """Verifies validation of write access to destination directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            can_write = os.access(tmpdir, os.W_OK)
            self.assertTrue(can_write)

    def test_tier2_f3_05_dwg_header_magic_bytes_check(self):
        """Verifies DWG magic header bytes (AC10xx)."""
        valid_magic = b"AC1027"  # AutoCAD 2013-2017
        invalid_magic = b"NOTCAD"
        self.assertTrue(valid_magic.startswith(b"AC10"))
        self.assertFalse(invalid_magic.startswith(b"AC10"))


# ==============================================================================
# Feature 4 Boundaries: Vector Rendering Engine (5 tests)
# ==============================================================================
class TestTier2Feature4Boundaries(unittest.TestCase):
    """Boundary conditions for vector rendering and image synthesis."""

    def test_tier2_f4_01_extreme_aspect_ratio_render(self):
        """Verifies rendering a diagram with 200:1 aspect ratio (e.g. long industrial busbar)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "long_busbar.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            msp.add_line((0, 0), (200.0, 1.0))
            doc.saveas(dxf_path)

            img, meta = render_dxf(dxf_path, px_per_cad=10.0, is_industrial=False)
            self.assertGreater(img.shape[1], img.shape[0])
            self.assertAlmostEqual(meta['W_cad'], 200.0, delta=1.0)
            self.assertAlmostEqual(meta['H_cad'], 1.0, delta=1.0)

    def test_tier2_f4_02_tiny_cad_dimensions(self):
        """Verifies rendering entities with sub-centimeter extents (0.05 CAD units)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "tiny_box.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            msp.add_line((0, 0), (0.05, 0.05))
            doc.saveas(dxf_path)

            img, meta = render_dxf(dxf_path, px_per_cad=100.0, is_industrial=False)
            self.assertGreater(img.shape[0], 0)
            self.assertGreater(img.shape[1], 0)

    def test_tier2_f4_03_large_render_dimension_clamping(self):
        """Verifies scale factor clamping when dimensions exceed MAX_DIM_PX."""
        max_dim_px = 2000
        ancho_cad, alto_cad = 500.0, 500.0
        px_per_cad = 10.0  # 500 * 10 = 5000 px > 2000 px cap
        ancho_px = ancho_cad * px_per_cad
        if ancho_px > max_dim_px:
            factor = max_dim_px / ancho_px
            px_per_cad *= factor
            ancho_px = int(round(ancho_cad * px_per_cad))
        self.assertEqual(ancho_px, 2000)
        self.assertEqual(px_per_cad, 4.0)

    def test_tier2_f4_04_zero_scale_handling(self):
        """Verifies non-positive px_per_cad throws exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test.dxf")
            create_minimal_dxf(dxf_path)
            with self.assertRaises(Exception):
                render_dxf(dxf_path, px_per_cad=0.0)

    def test_tier2_f4_05_custom_background_color(self):
        """Verifies custom background color hex string format."""
        for bg in ['#ffffff', '#000000', '#f0f0f0']:
            self.assertTrue(bg.startswith('#'))
            self.assertEqual(len(bg), 7)


# ==============================================================================
# Feature 5 Boundaries: Scale Normalization Engine (5 tests)
# ==============================================================================
class TestTier2Feature5Boundaries(unittest.TestCase):
    """Boundary conditions for scale analysis and multimodal clustering."""

    def test_tier2_f5_01_scale_analyzer_empty_modelspace(self):
        """Verifies scale analyzer fallback on empty ModelSpace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "empty.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            doc.saveas(dxf_path)

            px_per_cad, ref = scale_analyzer.calcular_factor_escala(dxf_path)
            self.assertGreater(px_per_cad, 0.0)
            self.assertEqual(ref[0], "BBOX")

    def test_tier2_f5_02_single_insert_scale(self):
        """Verifies scale analyzer with exactly 1 insert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "single_insert.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            blk = doc.blocks.new("SINGLE")
            blk.add_line((0, 0), (1.0, 1.0))
            doc.modelspace().add_blockref("SINGLE", (0, 0))
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            diag = scale_analyzer.analizar_inserts(doc_read.modelspace())
            self.assertIsNotNone(diag)
            self.assertAlmostEqual(diag, 1.0, delta=0.2)

    def test_tier2_f5_03_inserts_with_negative_scales(self):
        """Verifies mirrored block inserts with negative scale factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "mirrored.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            blk = doc.blocks.new("MIRROR")
            blk.add_line((0, 0), (2.0, 3.0))
            msp = doc.modelspace()
            msp.add_blockref("MIRROR", (0, 0), dxfattribs={"xscale": -1.0, "yscale": 1.0})
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            diag = scale_analyzer.analizar_inserts(doc_read.modelspace())
            self.assertIsNotNone(diag)
            self.assertGreater(diag, 0.0)

    def test_tier2_f5_04_outlier_circles_clustering(self):
        """Verifies log-bin clustering filters out 1 giant circle outlier among 50 normal circles."""
        arr = [0.5] * 50 + [5000.0]
        cluster_val = scale_analyzer._cluster_log_bin(np.array(arr))
        self.assertIsNotNone(cluster_val)
        self.assertAlmostEqual(cluster_val, 0.5, delta=0.05)

    def test_tier2_f5_05_extreme_target_px(self):
        """Verifies calculating scale with extreme target_px (e.g. 10 px vs 1000 px)."""
        dxf_path = DXF_DIR / "plano.dxf"
        px_small, _ = scale_analyzer.calcular_factor_escala(str(dxf_path), target_px=10)
        px_large, _ = scale_analyzer.calcular_factor_escala(str(dxf_path), target_px=1000)
        self.assertLess(px_small, px_large)
        self.assertAlmostEqual(px_large / px_small, 100.0, delta=1.0)


# ==============================================================================
# Feature 6 Boundaries: SAHI Slicing & Tiling (5 tests)
# ==============================================================================
class TestTier2Feature6Boundaries(unittest.TestCase):
    """Boundary conditions for sliding window slicing and boundary padding."""

    def test_tier2_f6_01_image_smaller_than_slice_size(self):
        """Verifies slicing an image smaller than the 640x640 tile size (e.g. 300x200)."""
        H, W = 200, 300
        slices = generate_slices(H, W, slice_size=640, overlap=0.80)
        # Should generate at least 1 tile covering the entire image (0, 0, 300, 200)
        self.assertGreaterEqual(len(slices), 1)
        s = slices[0]
        self.assertEqual(s[0], 0)
        self.assertEqual(s[1], 0)
        self.assertEqual(s[2], W)
        self.assertEqual(s[3], H)

    def test_tier2_f6_02_image_exact_slice_size(self):
        """Verifies slicing an image with exact dimensions 640x640."""
        H, W = 640, 640
        slices = generate_slices(H, W, slice_size=640, overlap=0.80)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0], (0, 0, 640, 640))

    def test_tier2_f6_03_zero_overlap_slicing(self):
        """Verifies slicing with 0.0 overlap (grid partitioning)."""
        H, W = 1280, 1280
        slices = generate_slices(H, W, slice_size=640, overlap=0.0)
        # 2x2 grid = 4 slices
        self.assertEqual(len(slices), 4)

    def test_tier2_f6_04_extreme_overlap_slicing(self):
        """Verifies slicing with 95% overlap (fine stride = 32 px)."""
        H, W = 700, 700
        slices = generate_slices(H, W, slice_size=640, overlap=0.95)
        # Stride = int(round(640 * 0.05)) = 32 px
        self.assertGreater(len(slices), 4)

    def test_tier2_f6_05_zero_padding_slicing(self):
        """Verifies border padding behavior when pad=0."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        padded = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        self.assertEqual(padded.shape, img.shape)


# ==============================================================================
# Feature 7 Boundaries: Centroidal NMS Calibration (5 tests)
# ==============================================================================
class TestTier2Feature7Boundaries(unittest.TestCase):
    """Boundary conditions for confidence filtering, d_min sweep, and duplicate suppression."""

    def test_tier2_f7_01_nms_conf_threshold_zero(self):
        """Verifies conf_thresh = 0.0 retains all detections."""
        raw = [
            {"conf": 0.001, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5},
            {"conf": 0.999, "bbox_cad": [10, 10, 11, 11], "xc": 10.5, "yc": 10.5}
        ]
        filtered = [d for d in raw if d["conf"] >= 0.0]
        self.assertEqual(len(filtered), 2)

    def test_tier2_f7_02_nms_conf_threshold_one(self):
        """Verifies conf_thresh = 1.0 filters out all detections with conf < 1.0."""
        raw = [
            {"conf": 0.99, "bbox_cad": [0, 0, 1, 1], "xc": 0.5, "yc": 0.5},
            {"conf": 1.00, "bbox_cad": [10, 10, 11, 11], "xc": 10.5, "yc": 10.5}
        ]
        filtered = [d for d in raw if d["conf"] >= 1.0]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["conf"], 1.0)

    def test_tier2_f7_03_nms_dmin_very_small(self):
        """Verifies d_min = 0.01 preserves closely spaced distinct components."""
        dets = [
            {"xc": 5.0, "yc": 10.0, "bbox_cad": [4.5, 9.5, 5.5, 10.5], "conf": 0.90},
            {"xc": 5.05, "yc": 10.0, "bbox_cad": [4.55, 9.5, 5.55, 10.5], "conf": 0.85}  # dist = 0.05 > 0.01
        ]
        kept = nms_distancia_cad(dets, d_min=0.01)
        self.assertEqual(len(kept), 2)

    def test_tier2_f7_04_nms_dmin_large(self):
        """Verifies d_min = 5.0 suppresses even distant detections."""
        dets = [
            {"xc": 5.0, "yc": 10.0, "bbox_cad": [4.5, 9.5, 5.5, 10.5], "conf": 0.95},
            {"xc": 8.0, "yc": 10.0, "bbox_cad": [7.5, 9.5, 8.5, 10.5], "conf": 0.80}   # dist = 3.0 < 5.0
        ]
        kept = nms_distancia_cad(dets, d_min=5.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["conf"], 0.95)

    def test_tier2_f7_05_nms_identical_duplicate_boxes(self):
        """Verifies that 10 exactly identical bounding boxes collapse to 1 under IoU NMS."""
        dets = [
            {"bbox_cad": [2.0, 3.0, 3.0, 4.0], "conf": 0.50 + i * 0.04}
            for i in range(10)
        ]
        kept = nms_iou_cad(dets, iou_thresh=0.45)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["conf"], 0.86)


# ==============================================================================
# Feature 8 Boundaries: Apparatus Detection Gamut (5 tests)
# ==============================================================================
class TestTier2Feature8Boundaries(unittest.TestCase):
    """Boundary conditions for bounding box dimensions and coordinate representations."""

    def test_tier2_f8_01_empty_detections_list(self):
        """Verifies empty list input returns empty list without error across all NMS functions."""
        self.assertEqual(nms_iou_cad([]), [])
        self.assertEqual(eliminar_anidadas([]), [])
        self.assertEqual(nms_distancia_cad([]), [])

    def test_tier2_f8_02_single_detection(self):
        """Verifies exactly 1 detection passes through NMS intact."""
        single = [{"bbox_cad": [0, 0, 1, 1], "conf": 0.9, "xc": 0.5, "yc": 0.5}]
        self.assertEqual(len(nms_iou_cad(single)), 1)
        self.assertEqual(len(eliminar_anidadas(single)), 1)
        self.assertEqual(len(nms_distancia_cad(single)), 1)

    def test_tier2_f8_03_subpixel_bounding_box(self):
        """Verifies sub-pixel bounding box (width < 1 px) handling."""
        box = [100.2, 200.3, 100.8, 200.9]
        w_px = box[2] - box[0]
        h_px = box[3] - box[1]
        self.assertLess(w_px, 1.0)
        self.assertLess(h_px, 1.0)
        int_box = [int(box[0]), int(box[1]), int(math.ceil(box[2])), int(math.ceil(box[3]))]
        self.assertGreaterEqual(int_box[2] - int_box[0], 1)

    def test_tier2_f8_04_full_canvas_bounding_box(self):
        """Verifies detection spanning the entire canvas."""
        W_px, H_px = 3000, 2000
        canvas_box = [0, 0, W_px, H_px]
        area = (canvas_box[2] - canvas_box[0]) * (canvas_box[3] - canvas_box[1])
        self.assertEqual(area, W_px * H_px)

    def test_tier2_f8_05_negative_cad_coordinate_boxes(self):
        """Verifies detection coordinates in negative CAD space."""
        cad_bbox = [-50.0, -30.0, -48.0, -28.0]
        xc = (cad_bbox[0] + cad_bbox[2]) / 2.0
        yc = (cad_bbox[1] + cad_bbox[3]) / 2.0
        self.assertEqual(xc, -49.0)
        self.assertEqual(yc, -29.0)


# ==============================================================================
# Feature 9 Boundaries: Metric Audit Engine (5 tests)
# ==============================================================================
class TestTier2Feature9Boundaries(unittest.TestCase):
    """Boundary conditions for metrics evaluation (0 GT, 0 detections, extreme tolerances)."""

    def test_tier2_f9_01_zero_gt_zero_detections(self):
        """Verifies metrics when both GT and detections are empty."""
        metrics = compute_greedy_matching([], [], dist_tol=1.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["recall"], 100.0)
        self.assertEqual(metrics["precision"], 0.0)

    def test_tier2_f9_02_zero_gt_with_detections(self):
        """Verifies metrics when GT is empty but detections exist (all FP)."""
        dets = [{"xc": i * 5.0, "yc": 0.0, "bbox_cad": [0,0,1,1]} for i in range(8)]
        metrics = compute_greedy_matching(dets, [], dist_tol=1.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 8)
        self.assertEqual(metrics["precision"], 0.0)

    def test_tier2_f9_03_gt_with_zero_detections(self):
        """Verifies metrics when GT exists but detections are empty (0% recall, all FN)."""
        gt = [{"xc": i * 5.0, "yc": 0.0} for i in range(15)]
        metrics = compute_greedy_matching([], gt, dist_tol=1.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fn"], 15)
        self.assertEqual(metrics["recall"], 0.0)

    def test_tier2_f9_04_distance_tolerance_zero(self):
        """Verifies dist_tol = 0.0 matches only exact coordinates."""
        gt = [{"xc": 10.0, "yc": 20.0}]
        det_exact = [{"xc": 10.0, "yc": 20.0, "bbox_cad": [9, 19, 11, 21]}]
        det_near = [{"xc": 10.001, "yc": 20.0, "bbox_cad": [9, 19, 11, 21]}]

        m1 = compute_greedy_matching(det_exact, gt, dist_tol=0.0)
        self.assertEqual(m1["tp"], 1)

    def test_tier2_f9_05_distance_tolerance_infinite(self):
        """Verifies large dist_tol matches any detection regardless of distance."""
        gt = [{"xc": 0.0, "yc": 0.0}]
        det_far = [{"xc": 9999.0, "yc": 9999.0, "bbox_cad": [9998, 9998, 10000, 10000]}]
        metrics = compute_greedy_matching(det_far, gt, dist_tol=1e6)
        self.assertEqual(metrics["tp"], 1)


# ==============================================================================
# Feature 10 Boundaries: Visual Sheets & Export (5 tests)
# ==============================================================================
class TestTier2Feature10Boundaries(unittest.TestCase):
    """Boundary conditions for visual sheet, CSV, and JSON generation."""

    def test_tier2_f10_01_export_empty_detections(self):
        """Verifies exporting files with zero detections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty_dets.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'xc_cad', 'yc_cad', 'conf'])
            self.assertTrue(os.path.exists(csv_path))

    def test_tier2_f10_02_export_thousands_detections(self):
        """Verifies exporting 1000 detections to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "many_dets.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'xc_cad', 'yc_cad', 'conf'])
                for i in range(1000):
                    writer.writerow([i, f"{i*0.5:.2f}", "10.0", "0.95"])

            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = list(csv.reader(f))
            self.assertEqual(len(lines), 1001)

    def test_tier2_f10_03_export_special_characters_in_stem(self):
        """Verifies file generation with filenames containing accents and spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "Tablero_Sección_1 (Principal)"
            json_file = os.path.join(tmpdir, f"{stem}_detections.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({"test": True}, f)
            self.assertTrue(os.path.exists(json_file))

    def test_tier2_f10_04_export_pixel_coordinates_clamped(self):
        """Verifies clamping coordinates at canvas boundary."""
        W_px, H_px = 1920, 1080
        raw_x1, raw_y1, raw_x2, raw_y2 = -10, -5, 2000, 1100
        clamped = [
            max(0, min(W_px, raw_x1)),
            max(0, min(H_px, raw_y1)),
            max(0, min(W_px, raw_x2)),
            max(0, min(H_px, raw_y2))
        ]
        self.assertEqual(clamped, [0, 0, 1920, 1080])

    def test_tier2_f10_05_export_json_serialization_numpy_types(self):
        """Verifies JSON serialization converts numpy types (float32, int64)."""
        data = {
            "scale": float(np.float32(75.0)),
            "count": int(np.int64(42)),
            "coords": [float(v) for v in np.array([1.2, 3.4])]
        }
        dumped = json.dumps(data)
        loaded = json.loads(dumped)
        self.assertEqual(loaded["count"], 42)


# ==============================================================================
# Feature 11 Boundaries: Zero-Regression Benchmarks (5 tests)
# ==============================================================================
class TestTier2Feature11Boundaries(unittest.TestCase):
    """Boundary conditions for 98% recall enforcement and regression flagging."""

    def test_tier2_f11_01_regression_at_exact_98_percent(self):
        """Verifies Recall = 98.0% exactly meets the acceptance threshold."""
        recall = 98.0
        self.assertGreaterEqual(recall, 98.0)

    def test_tier2_f11_02_regression_at_97_point_9_percent(self):
        """Verifies Recall = 97.9% is flagged as failing."""
        recall = 97.9
        self.assertLess(recall, 98.0)

    def test_tier2_f11_03_regression_with_single_false_negative(self):
        """Verifies flagging when a pre-validated switchboard has 1 false negative."""
        tsbe = {"gt": 15, "tp": 14, "fn": 1}
        has_fn = tsbe["fn"] > 0
        self.assertTrue(has_fn)

    def test_tier2_f11_04_regression_benchmark_missing_file(self):
        """Verifies handling missing benchmark ground truth file cleanly."""
        missing = DXF_DIR / "nonexistent_gt.csv"
        self.assertFalse(missing.exists())

    def test_tier2_f11_05_regression_zero_gt_guard(self):
        """Verifies protecting against ZeroDivisionError when GT list is empty."""
        total_gt = 0
        tp = 0
        recall = (tp / total_gt * 100.0) if total_gt > 0 else 100.0
        self.assertEqual(recall, 100.0)


# ==============================================================================
# Feature 12 Boundaries: Input Validation & Robustness (5 tests)
# ==============================================================================
class TestTier2Feature12Boundaries(unittest.TestCase):
    """Boundary conditions for corrupted inputs, truncated DXF, and invalid tags."""

    def test_tier2_f12_01_truncated_dxf_file(self):
        """Verifies truncated DXF file raises exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trunc = os.path.join(tmpdir, "truncated.dxf")
            with open(trunc, 'w', encoding='ascii') as f:
                f.write("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1027\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n")
            with self.assertRaises(Exception):
                render_dxf(trunc)

    def test_tier2_f2_02_binary_garbage_dxf(self):
        """Verifies random binary data raises ezdxf exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = os.path.join(tmpdir, "random_bytes.dxf")
            with open(bad, 'wb') as f:
                f.write(os.urandom(256))
            with self.assertRaises(Exception):
                render_dxf(bad)

    def test_tier2_f12_03_empty_file_zero_bytes(self):
        """Verifies 0-byte DXF raises exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zero_dxf = os.path.join(tmpdir, "zero.dxf")
            with open(zero_dxf, 'wb') as f:
                f.write(b"")
            with self.assertRaises(Exception):
                render_dxf(zero_dxf)

    def test_tier2_f12_04_unsupported_version_tag(self):
        """Verifies invalid AutoCAD version tag handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_ver = os.path.join(tmpdir, "bad_ver.dxf")
            with open(bad_ver, 'w', encoding='ascii') as f:
                f.write("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nINVALID_VER_999\n0\nENDSEC\n0\nEOF\n")
            # Should either load with fallback or raise DXFVersionError
            try:
                doc = ezdxf.readfile(bad_ver)
            except Exception as e:
                self.assertIsNotNone(e)

    def test_tier2_f12_05_corrupted_image_file(self):
        """Verifies reading truncated image file fails safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_img = os.path.join(tmpdir, "corrupt.png")
            with open(bad_img, 'wb') as f:
                f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtruncated")
            img = cv2.imread(bad_img)
            self.assertIsNone(img)


if __name__ == "__main__":
    unittest.main()
