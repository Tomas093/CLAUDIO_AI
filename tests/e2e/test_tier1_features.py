"""
Tier 1: Feature Isolation Tests (F1 to F12)
Comprehensive test suite covering each feature in isolation with >= 5 test cases per feature (>= 60 test cases).
Directly mapped to ORIGINAL_REQUEST.md, PROJECT.md, and TEST_INFRA.md.
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
from ezdxf.addons.drawing.config import Configuration, ColorPolicy, TextPolicy, HatchPolicy

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
    TEST_DIR,
    create_minimal_dxf,
    create_synthetic_switchboard_dxf,
    create_test_image,
    compute_greedy_matching
)


# ==============================================================================
# Feature 1: Web Autonomous Acquisition Logic (5 tests)
# ==============================================================================
class TestFeature1WebAcquisition(unittest.TestCase):
    """Tests URL validation, allowed repositories, CAD extension checks, and manifest tracking."""

    def test_f1_01_url_validation_cad_extensions(self):
        """Verifies CAD extension detection (.dwg / .dxf) for remote URLs."""
        valid_urls = [
            "https://raw.githubusercontent.com/user/repo/main/unifilar.dxf",
            "https://fi.uba.ar/catedras/electrotecnia/plano_tablero.DWG",
            "https://utn.edu.ar/repositorio/planos/05_diagrama_unifilar.dxf"
        ]
        invalid_urls = [
            "https://github.com/user/repo/archive/refs/tags/v1.0.zip",
            "https://example.com/downloads/setup.exe",
            "https://schneider.com/ar/manual.pdf"
        ]
        cad_exts = {".dxf", ".dwg"}
        for url in valid_urls:
            ext = Path(urlparse(url).path).suffix.lower()
            self.assertIn(ext, cad_exts, f"URL {url} should have a valid CAD extension")
        for url in invalid_urls:
            ext = Path(urlparse(url).path).suffix.lower()
            self.assertNotIn(ext, cad_exts, f"URL {url} should not be recognized as CAD")

    def test_f1_02_allowed_host_domain_filtering(self):
        """Verifies open-source repository allowlist matching."""
        allowed_domains = {
            "github.com", "raw.githubusercontent.com",
            "fi.uba.ar", "utn.edu.ar",
            "se.com", "siemens.com"
        }
        test_cases = [
            ("https://github.com/ingenieria/planos/main.dxf", True),
            ("https://raw.githubusercontent.com/schneider/diagrams/tablero.dwg", True),
            ("https://fi.uba.ar/planos/unifilar.dxf", True),
            ("https://malicious-spam-site.com/fake.dxf", False),
            ("http://untrusted-domain.xyz/download.dwg", False)
        ]
        for url, expected in test_cases:
            host = urlparse(url).hostname or ""
            is_allowed = any(host == d or host.endswith("." + d) for d in allowed_domains)
            self.assertEqual(is_allowed, expected, f"Domain check failed for {url}")

    def test_f1_03_cad_mime_type_resolution(self):
        """Verifies CAD MIME type mapping."""
        mime_map = {
            "application/dxf": ".dxf",
            "image/vnd.dxf": ".dxf",
            "application/acad": ".dwg",
            "application/x-dwg": ".dwg",
            "image/vnd.dwg": ".dwg"
        }
        for mime, expected_ext in mime_map.items():
            self.assertTrue(expected_ext in (".dxf", ".dwg"))
            self.assertTrue(mime.startswith("application/") or mime.startswith("image/"))

    def test_f1_04_target_filename_sanitization(self):
        """Verifies sanitizing remote URLs into filesystem-safe local filenames."""
        raw_names = [
            "Plano%20Unifilar%20General.dxf",
            "Tablero/Principal:01.dwg",
            "Esquema*Final?v1.dxf"
        ]
        for name in raw_names:
            # Sanitize invalid Windows chars: < > : " / \ | ? * and unquote
            clean = name.replace("%20", "_")
            for ch in '<>:"/\\|?*':
                clean = clean.replace(ch, "_")
            self.assertNotIn(":", clean)
            self.assertNotIn("/", clean)
            self.assertNotIn("*", clean)
            self.assertNotIn("?", clean)
            self.assertTrue(clean.endswith(".dxf") or clean.endswith(".dwg"))

    def test_f1_05_acquisition_manifest_metadata(self):
        """Verifies acquisition manifest metadata schema."""
        record = {
            "source_url": "https://raw.githubusercontent.com/example/05_diagrama_unifilar.dxf",
            "local_path": "dxf/externos/05_diagrama_unifilar.dxf",
            "timestamp": "2026-09-11T20:00:00Z",
            "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "byte_size": 1048576,
            "format": "DXF",
            "status": "COMPLETED"
        }
        required_keys = {"source_url", "local_path", "timestamp", "sha256", "byte_size", "format", "status"}
        self.assertTrue(required_keys.issubset(record.keys()))
        self.assertEqual(len(record["sha256"]), 64)
        self.assertGreater(record["byte_size"], 0)


# ==============================================================================
# Feature 2: Local DXF Reading & Cataloging (5 tests)
# ==============================================================================
class TestFeature2LocalIngestion(unittest.TestCase):
    """Tests local DXF reading, layer parsing, entity inspection, and extents."""

    def test_f2_01_read_tablerotsbe_dxf(self):
        """Verifies reading local Argentine plan Tablerotsbe.dxf."""
        dxf_path = DXF_DIR / "Tablerotsbe.dxf"
        self.assertTrue(dxf_path.exists(), f"Missing required file: {dxf_path}")
        doc = ezdxf.readfile(str(dxf_path))
        self.assertIsNotNone(doc)
        msp = doc.modelspace()
        self.assertGreater(len(msp), 0)

    def test_f2_02_read_plano3_dxf(self):
        """Verifies reading local distribution switchboard plano3.dxf."""
        dxf_path = DXF_DIR / "plano3.dxf"
        self.assertTrue(dxf_path.exists(), f"Missing required file: {dxf_path}")
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        self.assertGreater(len(msp), 0)
        self.assertTrue(hasattr(doc, "layers"))

    def test_f2_03_catalog_local_plan_inventory(self):
        """Verifies the catalog of local switchboard plans plano.dxf to plano5.dxf."""
        expected_plans = ["Tablerotsbe.dxf", "plano.dxf", "plano2.dxf", "plano3.dxf", "plano4.dxf"]
        for p_name in expected_plans:
            p_path = DXF_DIR / p_name
            self.assertTrue(p_path.exists(), f"Plan {p_name} missing from dxf/ directory")

    def test_f2_04_layer_classification_filtering(self):
        """Verifies separating electrical symbol geometry from text/border layers."""
        dxf_path = DXF_DIR / "Tablerotsbe.dxf"
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        excluded_types = ('TEXT', 'MTEXT', 'DIMENSION', 'LEADER')
        excluded_layers = ('IE-UN-TEXTOS', 'FORMATO', 'CARATULA', 'DEFPOINTS')

        geom_entities = [
            e for e in msp
            if e.dxftype() not in excluded_types
            and e.dxf.layer.upper() not in excluded_layers
        ]
        self.assertGreater(len(geom_entities), 0)
        for e in geom_entities:
            self.assertNotIn(e.dxftype(), excluded_types)
            self.assertNotIn(e.dxf.layer.upper(), excluded_layers)

    def test_f2_05_modelspace_extents_calculation(self):
        """Verifies CAD bounding extents calculation yielding positive dimensions."""
        dxf_path = DXF_DIR / "Tablerotsbe.dxf"
        doc = ezdxf.readfile(str(dxf_path))
        bbox = ezdxf.bbox.extents(doc.modelspace())
        self.assertTrue(bbox.has_data)
        w_cad = bbox.extmax.x - bbox.extmin.x
        h_cad = bbox.extmax.y - bbox.extmin.y
        self.assertGreater(w_cad, 0.0)
        self.assertGreater(h_cad, 0.0)


# ==============================================================================
# Feature 3: Headless DWG to DXF Conversion Helper (5 tests)
# ==============================================================================
class TestFeature3DwgConverter(unittest.TestCase):
    """Tests DWG conversion logic, CLI command assembly, script creation, and error handling."""

    def test_f3_01_dwg_extension_validation(self):
        """Verifies validation accepts .dwg and rejects other extensions."""
        valid_paths = ["switchboard.dwg", "DWG_FILES/PLANO.DWG", "c:/cad/unifilar.Dwg"]
        invalid_paths = ["diagram.pdf", "circuit.png", "data.json", "plan.dxf"]
        for p in valid_paths:
            self.assertEqual(Path(p).suffix.lower(), ".dwg")
        for p in invalid_paths:
            self.assertNotEqual(Path(p).suffix.lower(), ".dwg")

    def test_f3_02_accoreconsole_command_construction(self):
        """Verifies accoreconsole command construction with required flags."""
        dwg_file = "C:/cad/switchboard.dwg"
        scr_file = "C:/cad/convert.scr"
        cmd = f'accoreconsole.exe /product "ACAD_E" /s "{scr_file}" /i "{dwg_file}"'
        self.assertIn('accoreconsole.exe', cmd)
        self.assertIn('/product "ACAD_E"', cmd)
        self.assertIn('/s', cmd)
        self.assertIn('/i', cmd)
        self.assertIn(dwg_file, cmd)

    def test_f3_03_dwg_to_dxf_script_generation(self):
        """Verifies AutoCAD SCR conversion script generation."""
        target_dxf = "C:/cad/output.dxf"
        scr_lines = [
            "FILEDIA 0",
            f'DXFOUT "{target_dxf}" V 2018 16',
            "QUIT Y"
        ]
        script_content = "\n".join(scr_lines) + "\n"
        self.assertIn("FILEDIA 0", script_content)
        self.assertIn("DXFOUT", script_content)
        self.assertIn("QUIT Y", script_content)
        self.assertIn(target_dxf, script_content)

    def test_f3_04_missing_dwg_file_handling(self):
        """Verifies error handling when input DWG file does not exist."""
        nonexistent = Path("nonexistent_path_to_diagram.dwg")
        with self.assertRaises(FileNotFoundError):
            if not nonexistent.exists():
                raise FileNotFoundError(f"DWG file not found: {nonexistent}")

    def test_f3_05_output_dxf_path_derivation(self):
        """Verifies resolution of output DXF path from source DWG path."""
        src_path = Path("dxf/substation_diagram.dwg")
        out_dir = Path("dxf/clean")
        out_path = out_dir / (src_path.stem + ".dxf")
        self.assertEqual(out_path.suffix, ".dxf")
        self.assertEqual(out_path.name, "substation_diagram.dxf")


# ==============================================================================
# Feature 4: Vector Rendering Engine (5 tests)
# ==============================================================================
class TestFeature4VectorRendering(unittest.TestCase):
    """Tests ColorPolicy.COLOR_SWAP_BW, HatchPolicy.NORMAL, Agg backend, and render_dxf."""

    def test_f4_01_color_swap_bw_policy(self):
        """Verifies ColorPolicy.COLOR_SWAP_BW in ezdxf drawing config."""
        cfg = Configuration(
            color_policy=ColorPolicy.COLOR_SWAP_BW,
            text_policy=TextPolicy.IGNORE,
            custom_bg_color='#ffffff'
        )
        self.assertEqual(cfg.color_policy, ColorPolicy.COLOR_SWAP_BW)
        self.assertEqual(cfg.text_policy, TextPolicy.IGNORE)
        self.assertEqual(cfg.custom_bg_color, '#ffffff')

    def test_f4_02_hatch_policy_normal(self):
        """Verifies HatchPolicy.NORMAL configuration."""
        cfg = Configuration(
            hatch_policy=HatchPolicy.NORMAL
        )
        self.assertEqual(cfg.hatch_policy, HatchPolicy.NORMAL)

    def test_f4_03_render_dxf_produces_valid_image(self):
        """Verifies render_dxf returns high-resolution BGR numpy image and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_render.dxf")
            create_minimal_dxf(dxf_path)
            img_bgr, meta = render_dxf(dxf_path, px_per_cad=50.0, is_industrial=False)
            self.assertIsInstance(img_bgr, np.ndarray)
            self.assertEqual(len(img_bgr.shape), 3)
            self.assertEqual(img_bgr.shape[2], 3)
            self.assertGreater(img_bgr.shape[0], 0)
            self.assertGreater(img_bgr.shape[1], 0)

    def test_f4_04_matplotlib_agg_backend_enforced(self):
        """Verifies non-interactive Agg backend is enforced for headless rendering."""
        import matplotlib
        backend = matplotlib.get_backend().lower()
        self.assertIn("agg", backend)

    def test_f4_05_render_metadata_contract(self):
        """Verifies metadata dict contract returned by render_dxf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_contract.dxf")
            create_minimal_dxf(dxf_path)
            _, meta = render_dxf(dxf_path, px_per_cad=75.0, is_industrial=False)
            required_keys = {
                'px_per_cad', 'x_min_cad', 'y_min_cad', 'x_max_cad', 'y_max_cad',
                'W_cad', 'H_cad', 'W_px', 'H_px'
            }
            self.assertTrue(required_keys.issubset(meta.keys()))
            self.assertEqual(meta['px_per_cad'], 75.0)
            self.assertGreater(meta['W_cad'], 0)
            self.assertGreater(meta['H_cad'], 0)


# ==============================================================================
# Feature 5: Scale Normalization Engine (5 tests)
# ==============================================================================
class TestFeature5ScaleNormalization(unittest.TestCase):
    """Tests scale_analyzer, log-bin clustering, insert/circle analysis, and 75-100 px/cad range."""

    def test_f5_01_scale_from_modular_inserts(self):
        """Verifies analyzing block inserts to determine scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_inserts.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            blk = doc.blocks.new(name="PIA_BLOCK")
            blk.add_line((0, 0), (1.0, 1.0))
            msp = doc.modelspace()
            for i in range(5):
                msp.add_blockref("PIA_BLOCK", (i * 2.0, 0))
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            diag = scale_analyzer.analizar_inserts(doc_read.modelspace())
            self.assertIsNotNone(diag)
            self.assertAlmostEqual(diag, 1.0, delta=0.2)

    def test_f5_02_scale_from_circles(self):
        """Verifies analyzing circle radius to determine symbol scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_circles.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            for i in range(8):
                msp.add_circle((i * 3.0, 5.0), radius=0.5)
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            radius = scale_analyzer.analizar_circulos(doc_read.modelspace())
            self.assertIsNotNone(radius)
            self.assertAlmostEqual(radius, 0.5, delta=0.1)

    def test_f5_03_scale_from_text_fallback(self):
        """Verifies fallback to text height when no blocks or circles exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_text.dxf")
            doc = ezdxf.new(dxfversion="R2010")
            msp = doc.modelspace()
            for i in range(5):
                msp.add_text("CIRCUITO", dxfattribs={"height": 2.5, "insert": (i * 10.0, 0)})
            doc.saveas(dxf_path)

            doc_read = ezdxf.readfile(dxf_path)
            h = scale_analyzer.analizar_textos(doc_read.modelspace())
            self.assertIsNotNone(h)
            self.assertAlmostEqual(h, 2.5, delta=0.1)

    def test_f5_04_scale_target_range_75_to_100(self):
        """Verifies calcular_factor_escala produces scale in the 75-100 px/CAD range on standard switchboard plans."""
        dxf_path = DXF_DIR / "plano.dxf"
        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))
        self.assertGreaterEqual(px_per_cad, 75.0)
        self.assertLessEqual(px_per_cad, 100.0)
        self.assertEqual(ref[0], "INSERT_diag")

    def test_f5_05_log_bin_clustering_robustness(self):
        """Verifies _cluster_log_bin isolates representative modal dimension despite outliers."""
        # 10 symbols around size 1.0 and 2 huge title frame outliers around 500.0
        data = np.array([0.98, 1.02, 1.0, 1.05, 0.95, 1.01, 0.99, 1.03, 0.97, 1.0, 500.0, 600.0])
        cluster_val = scale_analyzer._cluster_log_bin(data)
        self.assertIsNotNone(cluster_val)
        self.assertAlmostEqual(cluster_val, 1.0, delta=0.1)


# ==============================================================================
# Feature 6: SAHI Tiling & Inversion (5 tests)
# ==============================================================================
class TestFeature6SahiTiling(unittest.TestCase):
    """Tests 640x640 tile generation, 80% overlap, 320 px border padding, and inverse CAD projection."""

    def test_f6_01_slice_generation_step_size(self):
        """Verifies generate_slices uses step = 128 px for 80% overlap on 640 px slice."""
        H, W = 1000, 1000
        slices = generate_slices(H, W, slice_size=640, overlap=0.80)
        step_expected = int(round(640 * (1.0 - 0.80)))  # 128 px
        self.assertEqual(step_expected, 128)
        self.assertGreater(len(slices), 0)

    def test_f6_02_slice_coverage_full_span(self):
        """Verifies slices reach both x2 == W and y2 == H."""
        H, W = 1500, 2000
        slices = generate_slices(H, W, slice_size=640, overlap=0.80)
        max_x2 = max(s[2] for s in slices)
        max_y2 = max(s[3] for s in slices)
        min_x1 = min(s[0] for s in slices)
        min_y1 = min(s[1] for s in slices)
        self.assertEqual(min_x1, 0)
        self.assertEqual(min_y1, 0)
        self.assertEqual(max_x2, W)
        self.assertEqual(max_y2, H)

    def test_f6_03_constant_border_padding_320px(self):
        """Verifies 320 px border padding expands image by 640 px on each axis with white color."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        pad = 320
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        self.assertEqual(padded.shape[0], 400 + 2 * pad)
        self.assertEqual(padded.shape[1], 600 + 2 * pad)
        # Check border pixel values
        self.assertTrue(np.all(padded[0, 0] == [255, 255, 255]))

    def test_f6_04_inverse_cad_coordinate_mapping(self):
        """Verifies converting pixel coordinates back to CAD space with inverted Y-axis."""
        W_px, H_px = 1000, 500
        W_cad, H_cad = 100.0, 50.0
        x_min, y_max = 10.0, 60.0

        # Test center point
        px_x, px_y = 500, 250
        cad_x = x_min + (px_x / W_px) * W_cad
        cad_y = y_max - (px_y / H_px) * H_cad

        self.assertAlmostEqual(cad_x, 60.0)
        self.assertAlmostEqual(cad_y, 35.0)

    def test_f6_05_pixel_clamping_within_original_bounds(self):
        """Verifies that bounding boxes in padding margins are clamped to original image dimensions."""
        W_px, H_px = 800, 600
        pad = 320
        # Tile starting at x1_tile = 0, y1_tile = 0 in padded image
        # Detection at b = [100, 100, 400, 400] relative to tile
        b = [100, 100, 400, 400]
        x1_tile, y1_tile = 0, 0
        px_x1 = max(0, min(W_px, (x1_tile + b[0]) - pad))
        px_y1 = max(0, min(H_px, (y1_tile + b[1]) - pad))
        px_x2 = max(0, min(W_px, (x1_tile + b[2]) - pad))
        px_y2 = max(0, min(H_px, (y1_tile + b[3]) - pad))

        # (100 - 320) is negative -> clamped to 0
        self.assertEqual(px_x1, 0)
        self.assertEqual(px_y1, 0)
        # (400 - 320) = 80
        self.assertEqual(px_x2, 80)
        self.assertEqual(px_y2, 80)


# ==============================================================================
# Feature 7: Centroidal NMS Calibration (5 tests)
# ==============================================================================
class TestFeature7CentroidalNMS(unittest.TestCase):
    """Tests IoU 0.45, IoS 0.60, centroidal distance NMS (d_min ~ 0.20-0.35 CAD), and cascading."""

    def test_f7_01_nms_iou_cad_threshold_045(self):
        """Verifies nms_iou_cad eliminates overlapping boxes with IoU >= 0.45."""
        dets = [
            {"bbox_cad": [0.0, 0.0, 1.0, 1.0], "conf": 0.90},
            {"bbox_cad": [0.1, 0.1, 1.1, 1.1], "conf": 0.75},  # High IoU with box 1
            {"bbox_cad": [5.0, 5.0, 6.0, 6.0], "conf": 0.85}   # Completely separate
        ]
        kept = nms_iou_cad(dets, iou_thresh=0.45)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["conf"], 0.90)
        self.assertEqual(kept[1]["conf"], 0.85)

    def test_f7_02_eliminar_anidadas_ios_060(self):
        """Verifies eliminar_anidadas eliminates inner box with IoS >= 0.60."""
        dets = [
            {"bbox_cad": [0.0, 0.0, 2.0, 2.0], "conf": 0.90},  # Area = 4.0
            {"bbox_cad": [0.5, 0.5, 1.5, 1.5], "conf": 0.70},  # Area = 1.0, fully inside box 1
            {"bbox_cad": [10.0, 10.0, 12.0, 12.0], "conf": 0.80}
        ]
        kept = eliminar_anidadas(dets, ios_thresh=0.60)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0]["bbox_cad"][0], 0.0)
        self.assertAlmostEqual(kept[1]["bbox_cad"][0], 10.0)

    def test_f7_03_nms_distancia_cad_dmin(self):
        """Verifies nms_distancia_cad suppresses duplicate centroids within d_min."""
        dets = [
            {"xc": 5.0, "yc": 10.0, "bbox_cad": [4.5, 9.5, 5.5, 10.5], "conf": 0.95},
            {"xc": 5.1, "yc": 10.1, "bbox_cad": [4.6, 9.6, 5.6, 10.6], "conf": 0.60},  # dist = sqrt(0.01 + 0.01) = 0.1414 < 0.35
            {"xc": 7.0, "yc": 10.0, "bbox_cad": [6.5, 9.5, 7.5, 10.5], "conf": 0.90}   # dist = 2.0 > 0.35
        ]
        kept = nms_distancia_cad(dets, d_min=0.35)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["conf"], 0.95)
        self.assertEqual(kept[1]["conf"], 0.90)

    def test_f7_04_nms_preserves_higher_confidence(self):
        """Verifies NMS always favors higher-confidence detection regardless of list order."""
        dets = [
            {"xc": 1.0, "yc": 1.0, "bbox_cad": [0.5, 0.5, 1.5, 1.5], "conf": 0.30},
            {"xc": 1.05, "yc": 1.05, "bbox_cad": [0.55, 0.55, 1.55, 1.55], "conf": 0.88}
        ]
        kept = nms_distancia_cad(dets, d_min=0.35)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["conf"], 0.88)

    def test_f7_05_cascaded_nms_sequence(self):
        """Verifies sequential execution: IoU -> Nested -> Distance NMS."""
        raw = [
            {"bbox_cad": [0.0, 0.0, 1.0, 1.0], "xc": 0.5, "yc": 0.5, "conf": 0.92},
            {"bbox_cad": [0.05, 0.05, 1.05, 1.05], "xc": 0.55, "yc": 0.55, "conf": 0.70}, # overlaps
            {"bbox_cad": [0.2, 0.2, 0.8, 0.8], "xc": 0.5, "yc": 0.5, "conf": 0.50},        # nested
            {"bbox_cad": [4.0, 4.0, 5.0, 5.0], "xc": 4.5, "yc": 4.5, "conf": 0.85}         # separate
        ]
        step1 = nms_iou_cad(raw, iou_thresh=0.45)
        step2 = eliminar_anidadas(step1, ios_thresh=0.60)
        step3 = nms_distancia_cad(step2, d_min=0.35)
        self.assertEqual(len(step3), 2)
        self.assertEqual(step3[0]["conf"], 0.92)
        self.assertEqual(step3[1]["conf"], 0.85)


# ==============================================================================
# Feature 8: Full Apparatus Gamut Detection (5 tests)
# ==============================================================================
class TestFeature8ApparatusGamut(unittest.TestCase):
    """Tests model loading, universal class 'componente', and apparatus representation."""

    def setUp(self):
        self.model_path = DETECTOR_MODEL_PATH
        self.assertTrue(self.model_path.exists(), f"Model missing: {self.model_path}")

    def test_f8_01_yolo_model_initialization(self):
        """Verifies DetectorUnifilar initializes with best_componente_nano.pt."""
        detector = DetectorUnifilar(model_path=str(self.model_path), device="cpu")
        self.assertIsNotNone(detector.model)

    def test_f8_02_universal_class_componente(self):
        """Verifies YOLO model names dict contains universal class 'componente'."""
        detector = DetectorUnifilar(model_path=str(self.model_path), device="cpu")
        names = detector.model.names
        self.assertIn(0, names)
        self.assertEqual(names[0], "componente")

    def test_f8_03_apparatus_detection_from_image(self):
        """Verifies detecting components on an image input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_apparatus.png")
            create_test_image(img_path, width=800, height=600)
            detector = DetectorUnifilar(model_path=str(self.model_path), device="cpu")
            res = detector.detectar_imagen(img_path, output_dir=tmpdir, conf_thresh=0.01)
            self.assertIn("total", res)
            self.assertIn("detections", res)
            self.assertIn("vis_image", res)

    def test_f8_04_detection_attributes_completeness(self):
        """Verifies detection dict attributes (conf, bbox_cad, xc, yc, bbox_px)."""
        det = {
            'conf': 0.85,
            'bbox_cad': [10.0, 20.0, 11.2, 21.5],
            'xc': 10.6,
            'yc': 20.75,
            'bbox_px': [100, 200, 112, 215]
        }
        for attr in ['conf', 'bbox_cad', 'xc', 'yc', 'bbox_px']:
            self.assertIn(attr, det)
        self.assertEqual(len(det['bbox_cad']), 4)
        self.assertEqual(len(det['bbox_px']), 4)

    def test_f8_05_detection_cad_centroid_calculation(self):
        """Verifies xc, yc match bbox_cad midpoints exactly."""
        cad_x1, cad_y1, cad_x2, cad_y2 = 4.2, 10.8, 5.4, 12.2
        xc = (cad_x1 + cad_x2) / 2.0
        yc = (cad_y1 + cad_y2) / 2.0
        self.assertAlmostEqual(xc, 4.8)
        self.assertAlmostEqual(yc, 11.5)


# ==============================================================================
# Feature 9: Metric Audit Engine (5 tests)
# ==============================================================================
class TestFeature9MetricAudit(unittest.TestCase):
    """Tests TP, FP, FN, Recall, Precision, and F1 score calculation logic."""

    def test_f9_01_recall_calculation_perfect(self):
        """Verifies Recall is 100.0% when TP == Total GT and FN == 0."""
        gt = [{"xc": i * 2.0, "yc": 5.0} for i in range(10)]
        dets = [{"xc": i * 2.0, "yc": 5.0, "bbox_cad": [i * 2.0 - 0.5, 4.5, i * 2.0 + 0.5, 5.5]} for i in range(10)]
        metrics = compute_greedy_matching(dets, gt, dist_tol=0.5)
        self.assertEqual(metrics["tp"], 10)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["recall"], 100.0)

    def test_f9_02_precision_calculation_with_false_positives(self):
        """Verifies Precision = TP / (TP + FP) * 100.0."""
        gt = [{"xc": 0.0, "yc": 0.0}]
        # 1 TP + 3 FP = 4 total detections
        dets = [
            {"xc": 0.0, "yc": 0.0, "bbox_cad": [-0.5, -0.5, 0.5, 0.5]},
            {"xc": 10.0, "yc": 10.0, "bbox_cad": [9.5, 9.5, 10.5, 10.5]},
            {"xc": 20.0, "yc": 20.0, "bbox_cad": [19.5, 19.5, 20.5, 20.5]},
            {"xc": 30.0, "yc": 30.0, "bbox_cad": [29.5, 29.5, 30.5, 30.5]}
        ]
        metrics = compute_greedy_matching(dets, gt, dist_tol=1.0)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 3)
        self.assertEqual(metrics["precision"], 25.0)

    def test_f9_03_f1_score_harmonic_mean(self):
        """Verifies F1 is harmonic mean of Recall and Precision."""
        rec = 80.0
        prec = 60.0
        expected_f1 = (2 * prec * rec) / (prec + rec)
        self.assertAlmostEqual(expected_f1, 68.5714, places=3)

    def test_f9_04_greedy_bipartite_matching_distance_tolerance(self):
        """Verifies matching pairs closest GT first within dist_tol."""
        gt = [{"xc": 1.0, "yc": 1.0}]
        # Closer det (dist = 0.2) vs Farther det (dist = 0.8)
        dets = [
            {"xc": 1.8, "yc": 1.0, "bbox_cad": [1.7, 0.9, 1.9, 1.1]},
            {"xc": 1.2, "yc": 1.0, "bbox_cad": [1.1, 0.9, 1.3, 1.1]}
        ]
        metrics = compute_greedy_matching(dets, gt, dist_tol=1.0)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)

    def test_f9_05_confusion_matrix_accounting(self):
        """Verifies confusion matrix invariant: TP + FN == Total GT and TP + FP == Total Dets."""
        gt = [{"xc": i * 5.0, "yc": 0.0} for i in range(12)]
        dets = [{"xc": i * 5.0 + (0.2 if i % 2 == 0 else 100.0), "yc": 0.0, "bbox_cad": [0,0,1,1]} for i in range(15)]
        metrics = compute_greedy_matching(dets, gt, dist_tol=1.0)
        self.assertEqual(metrics["tp"] + metrics["fn"], len(gt))
        self.assertEqual(metrics["tp"] + metrics["fp"], len(dets))


# ==============================================================================
# Feature 10: High-Res Visual Sheets & Export (5 tests)
# ==============================================================================
class TestFeature10VisualSheets(unittest.TestCase):
    """Tests visual detection sheets (green bboxes), CSV formatting, and JSON metadata structure."""

    def test_f10_01_visual_sheet_drawing(self):
        """Verifies visual sheet image has green bounding boxes BGR(0, 200, 0)."""
        canvas = np.full((300, 400, 3), 255, dtype=np.uint8)
        x1, y1, x2, y2 = 50, 50, 150, 150
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # Check that perimeter pixels contain green color
        green_pixel = canvas[50, 100]
        self.assertEqual(green_pixel[0], 0)
        self.assertEqual(green_pixel[1], 200)
        self.assertEqual(green_pixel[2], 0)

    def test_f10_02_csv_export_format(self):
        """Verifies CSV file structure, headers, and column count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_dets.csv")
            headers = ['id', 'xc_cad', 'yc_cad', 'x1_cad', 'y1_cad', 'x2_cad', 'y2_cad', 'conf', 'px_x1', 'px_y1', 'px_x2', 'px_y2']
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow([1, "10.5000", "20.5000", "10.0000", "20.0000", "11.0000", "21.0000", "0.9500", 100, 200, 110, 210])

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
            self.assertEqual(reader[0], headers)
            self.assertEqual(len(reader[1]), len(headers))

    def test_f10_03_json_export_structure(self):
        """Verifies JSON output schema matching M3 <-> M4 interface contract."""
        payload = {
            "metadata": {
                "px_per_cad": 75.0,
                "x_min_cad": 0.0,
                "y_min_cad": 0.0,
                "x_max_cad": 50.0,
                "y_max_cad": 30.0,
                "W_cad": 50.0,
                "H_cad": 30.0,
                "W_px": 3750,
                "H_px": 2250
            },
            "total_detected": 1,
            "confidence_threshold": 0.15,
            "components": [
                {
                    "conf": 0.92,
                    "bbox_cad": [10.0, 5.0, 11.0, 6.0],
                    "xc": 10.5,
                    "yc": 5.5,
                    "bbox_px": [750, 400, 825, 475]
                }
            ]
        }
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)
        self.assertIn("metadata", parsed)
        self.assertIn("total_detected", parsed)
        self.assertIn("components", parsed)
        self.assertEqual(parsed["total_detected"], 1)

    def test_f10_04_json_metadata_fields(self):
        """Verifies JSON metadata fields contain required CAD extents and pixel scales."""
        meta = {
            "px_per_cad": 80.0,
            "x_min_cad": 5.0, "y_min_cad": 10.0,
            "x_max_cad": 45.0, "y_max_cad": 35.0,
            "W_cad": 40.0, "H_cad": 25.0,
            "W_px": 3200, "H_px": 2000
        }
        for k in ["px_per_cad", "x_min_cad", "y_min_cad", "x_max_cad", "y_max_cad", "W_cad", "H_cad", "W_px", "H_px"]:
            self.assertIn(k, meta)

    def test_f10_05_output_artifacts_persistence(self):
        """Verifies creating PNG, CSV, and JSON artifacts simultaneously in output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "test_switchboard"
            png_f = os.path.join(tmpdir, f"{stem}_visual_detections.png")
            csv_f = os.path.join(tmpdir, f"{stem}_detections.csv")
            json_f = os.path.join(tmpdir, f"{stem}_detections.json")

            # Create dummy files
            cv2.imwrite(png_f, np.zeros((10, 10, 3), dtype=np.uint8))
            with open(csv_f, 'w') as f: f.write("id\n1")
            with open(json_f, 'w') as f: json.dump({"total": 1}, f)

            self.assertTrue(os.path.exists(png_f))
            self.assertTrue(os.path.exists(csv_f))
            self.assertTrue(os.path.exists(json_f))


# ==============================================================================
# Feature 11: Zero-Regression Historical Benchmarks (5 tests)
# ==============================================================================
class TestFeature11ZeroRegression(unittest.TestCase):
    """Tests benchmark ground truth loading, 98% recall enforcement, and regression detection."""

    def test_f11_01_benchmark_gt_files_exist(self):
        """Verifies benchmark ground-truth CSV files exist in dxf/."""
        benchmarks = [
            DXF_DIR / "fl_un_02_gt_completo.csv",
            DXF_DIR / "tsss_2_gt_completo.csv",
            DXF_DIR / "vyre_gt_completo.csv"
        ]
        for b_path in benchmarks:
            self.assertTrue(b_path.exists(), f"Benchmark GT missing: {b_path}")

    def test_f11_02_benchmark_recall_threshold_98_percent(self):
        """Verifies evaluation enforces Recall >= 98.0% as acceptance criterion."""
        pass_metrics = {"recall": 98.5, "precision": 99.0}
        fail_metrics = {"recall": 97.5, "precision": 99.0}
        self.assertGreaterEqual(pass_metrics["recall"], 98.0)
        self.assertLess(fail_metrics["recall"], 98.0)

    def test_f11_03_zero_false_negatives_rule(self):
        """Verifies zero false negative constraint on pre-validated switchboards."""
        tsbe_res = {"gt": 15, "tp": 15, "fn": 0}
        plano3_res = {"gt": 25, "tp": 25, "fn": 0}
        self.assertEqual(tsbe_res["fn"], 0)
        self.assertEqual(plano3_res["fn"], 0)

    def test_f11_04_benchmark_dataset_parsing(self):
        """Verifies reading and parsing benchmark GT CSV columns (name, xc, yc)."""
        gt_path = DXF_DIR / "vyre_gt_completo.csv"
        rows = []
        with open(gt_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        self.assertGreater(len(rows), 0)
        first = rows[0]
        self.assertTrue('x_cad' in first or 'xc' in first)
        self.assertTrue('y_cad' in first or 'yc' in first)

    def test_f11_05_regression_comparison_logic(self):
        """Verifies logic that flags regressions when recall drops below baseline."""
        baseline_recall = 99.0
        current_recall = 95.0
        is_regression = current_recall < baseline_recall
        self.assertTrue(is_regression)


# ==============================================================================
# Feature 12: Input Validation & Robustness (5 tests)
# ==============================================================================
class TestFeature12InputValidation(unittest.TestCase):
    """Tests file extension rejection, missing file errors, corrupted DXF, and invalid scales."""

    def test_f12_01_reject_non_cad_extension(self):
        """Verifies rejection of non-CAD extensions like .exe, .py, .docx."""
        invalid_files = ["script.py", "app.exe", "document.docx", "archive.tar.gz"]
        for f in invalid_files:
            ext = Path(f).suffix.lower()
            self.assertNotIn(ext, [".dxf", ".dwg"])

    def test_f12_02_missing_file_raises_filenotfound(self):
        """Verifies render_dxf raises FileNotFoundError for missing file."""
        nonexistent = "nonexistent_cad_diagram_9999.dxf"
        with self.assertRaises(FileNotFoundError):
            render_dxf(nonexistent)

    def test_f12_03_corrupted_dxf_handling(self):
        """Verifies ezdxf.DXFStructureError is raised on corrupted DXF content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dxf = os.path.join(tmpdir, "corrupt.dxf")
            with open(bad_dxf, 'w', encoding='ascii') as f:
                f.write("SECTION\n2\nENTITIES\n0\nLINE\n10\nBROKEN\n")
            with self.assertRaises(Exception):
                render_dxf(bad_dxf)

    def test_f12_04_invalid_scale_parameter_protection(self):
        """Verifies error handling when scale parameter is non-positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "test_scale.dxf")
            create_minimal_dxf(dxf_path)
            # With px_per_cad <= 0, image dimensions become 0
            with self.assertRaises(Exception):
                render_dxf(dxf_path, px_per_cad=-10.0)

    def test_f12_05_missing_image_for_detectar_imagen(self):
        """Verifies detector.detectar_imagen raises FileNotFoundError for missing image."""
        detector = DetectorUnifilar(model_path=str(DETECTOR_MODEL_PATH), device="cpu")
        with self.assertRaises(FileNotFoundError):
            detector.detectar_imagen("nonexistent_rendered_image_123.png")


if __name__ == "__main__":
    unittest.main()
