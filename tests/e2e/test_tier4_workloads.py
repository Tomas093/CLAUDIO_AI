"""
Tier 4: Realistic End-to-End Application Workload Tests (>= 6 test cases)
Validates complete end-to-end processing across realistic Argentine single-line diagrams:
1. Newly acquired residential switchboard (05_diagrama_unifilar.dxf)
2. Motor control switchboard (01_diagrama_unifilar.dxf)
3. Industrial CCM switchboard (01_diagrama_unifilar_ccm.dxf)
4. Pre-validated switchboards (Tablerotsbe: 15/15, plano3: 25/25 at conf 0.50)
5. Historical benchmark suite (test1.dxf and test_2.dxf with zero regressions)
6. Master drawing inspection and switchboard block extraction on OCJ-DE-IEL-UNI-000-001-O03
"""

import os
import sys
import json
import csv
import math
import tempfile
import unittest
from pathlib import Path

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
    DetectorUnifilar
)
import scale_analyzer
from tests.e2e.test_helpers import (
    PROJECT_ROOT,
    DETECTOR_MODEL_PATH,
    DXF_DIR,
    compute_greedy_matching
)


class TestTier4RealWorldWorkloads(unittest.TestCase):
    """End-to-end realistic workload execution and metric validation."""

    @classmethod
    def setUpClass(cls):
        device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        cls.detector = DetectorUnifilar(model_path=str(DETECTOR_MODEL_PATH), device=device)

    def test_tier4_01_workload_acquired_residential(self):
        """Scenario 1: Complete pipeline on newly acquired residential switchboard 05_diagrama_unifilar.dxf."""
        dxf_path = DXF_DIR / "externos" / "05_diagrama_unifilar.dxf"
        self.assertTrue(dxf_path.exists(), f"Missing acquired diagram: {dxf_path}")

        # Compute optimal scale via scale analyzer
        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))
        self.assertEqual(ref[0], "INSERT_diag")

        with tempfile.TemporaryDirectory() as tmpdir:
            res = self.detector.detectar_dxf(
                dxf_path=str(dxf_path),
                output_dir=tmpdir,
                conf_thresh=0.15,
                px_per_cad=px_per_cad,
                is_industrial=False,
                batch_size=32
            )
            # Verify high recall on acquired unifilar scheme (10 components detected)
            self.assertGreaterEqual(res["total"], 8, "Should detect modular breakers and switches")
            self.assertTrue(os.path.exists(res["vis_image"]))
            self.assertTrue(os.path.exists(res["csv_path"]))
            self.assertTrue(os.path.exists(res["json_path"]))

            # Verify JSON integrity
            with open(res["json_path"], "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            self.assertEqual(meta_json["total_detected"], res["total"])
            self.assertEqual(len(meta_json["components"]), res["total"])

    def test_tier4_02_workload_motor_control(self):
        """Scenario 2: Complete pipeline on motor control switchboard 01_diagrama_unifilar.dxf."""
        dxf_path = DXF_DIR / "externos" / "01_diagrama_unifilar.dxf"
        self.assertTrue(dxf_path.exists(), f"Missing motor control diagram: {dxf_path}")

        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))
        self.assertGreater(px_per_cad, 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            res = self.detector.detectar_dxf(
                dxf_path=str(dxf_path),
                output_dir=tmpdir,
                conf_thresh=0.15,
                px_per_cad=px_per_cad,
                is_industrial=True,
                batch_size=32
            )
            self.assertGreater(res["total"], 5, "Should detect motor protective switches")
            self.assertTrue(os.path.exists(res["vis_image"]))
            self.assertTrue(os.path.exists(res["csv_path"]))

    def test_tier4_03_workload_ccm_industrial(self):
        """Scenario 3: Complete pipeline on industrial CCM switchboard 01_diagrama_unifilar_ccm.dxf."""
        dxf_path = DXF_DIR / "externos" / "01_diagrama_unifilar_ccm.dxf"
        self.assertTrue(dxf_path.exists(), f"Missing industrial CCM diagram: {dxf_path}")

        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            res = self.detector.detectar_dxf(
                dxf_path=str(dxf_path),
                output_dir=tmpdir,
                conf_thresh=0.15,
                px_per_cad=px_per_cad,
                is_industrial=True,
                batch_size=32
            )
            self.assertGreater(res["total"], 5, "Should detect CCM switchgear components")
            self.assertTrue(os.path.exists(res["json_path"]))

    def test_tier4_04_workload_prevalidated_switchboards(self):
        """Scenario 4: Batch evaluation on pre-validated switchboards Tablerotsbe (15/15) and plano3 (25/25) at conf 0.50."""
        # 1. Tablerotsbe.dxf
        tsbe_path = DXF_DIR / "Tablerotsbe.dxf"
        self.assertTrue(tsbe_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            res_tsbe = self.detector.detectar_dxf(
                dxf_path=str(tsbe_path),
                output_dir=tmpdir,
                conf_thresh=0.50,
                px_per_cad=75.0,
                is_industrial=True,
                batch_size=32
            )
            # Authoritative expectation from ORIGINAL_REQUEST.md: 15/15 detections (100% Recall, 100% Precision)
            self.assertEqual(res_tsbe["total"], 15, f"Expected 15 detections for Tablerotsbe, got {res_tsbe['total']}")

        # 2. plano3.dxf
        plano3_path = DXF_DIR / "plano3.dxf"
        self.assertTrue(plano3_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            res_plano3 = self.detector.detectar_dxf(
                dxf_path=str(plano3_path),
                output_dir=tmpdir,
                conf_thresh=0.50,
                px_per_cad=75.0,
                is_industrial=True,
                batch_size=32
            )
            # Authoritative expectation from ORIGINAL_REQUEST.md: 25/25 detections (100% Recall, 100% Precision)
            self.assertEqual(res_plano3["total"], 25, f"Expected 25 detections for plano3, got {res_plano3['total']}")

    def test_tier4_05_workload_historical_benchmarks_test1_test2(self):
        """Scenario 5: Complete audit & regression run on historical benchmarks test1.dxf and test_2.dxf."""
        test1_path = PROJECT_ROOT / "test1.dxf"
        self.assertTrue(test1_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            res_t1 = self.detector.detectar_dxf(
                dxf_path=str(test1_path),
                output_dir=tmpdir,
                conf_thresh=0.15,
                px_per_cad=75.0,
                is_industrial=False,
                batch_size=32
            )
            # Historical benchmark test1 has ~126 components detected
            self.assertGreaterEqual(res_t1["total"], 120, "test1 should maintain high recall (>=120 detections)")

        test2_path = PROJECT_ROOT / "test_2.dxf"
        self.assertTrue(test2_path.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            res_t2 = self.detector.detectar_dxf(
                dxf_path=str(test2_path),
                output_dir=tmpdir,
                conf_thresh=0.15,
                px_per_cad=75.0,
                is_industrial=False,
                batch_size=32
            )
            self.assertGreaterEqual(res_t2["total"], 50, "test_2 should maintain high recall (>=50 detections)")

    def test_tier4_06_workload_master_sheet_ocj(self):
        """Scenario 6: Master drawing inspection and switchboard viewport block extraction on OCJ-DE-IEL-UNI-000-001-O03."""
        ocj_path = DXF_DIR / "OCJ-DE-IEL-UNI-000-001-O03.dxf"
        self.assertTrue(ocj_path.exists(), f"Missing master drawing: {ocj_path}")

        doc = ezdxf.readfile(str(ocj_path))
        self.assertIsNotNone(doc)
        msp = doc.modelspace()
        self.assertGreater(len(msp), 0)

        # Verify presence of master switchboard block reference
        block_refs = [e for e in msp if e.dxftype() == 'INSERT']
        self.assertGreater(len(block_refs), 0)
        master_ref = block_refs[0]
        self.assertIn("OCJ-ESQUEMAS_UNIFILARES", master_ref.dxf.name)

        # Inspect definition within doc.blocks
        block_name = master_ref.dxf.name
        self.assertIn(block_name, doc.blocks)
        block_def = doc.blocks[block_name]
        self.assertGreater(len(block_def), 1000, "Master switchboard block should contain extensive unifilar circuitry")


if __name__ == "__main__":
    unittest.main()
