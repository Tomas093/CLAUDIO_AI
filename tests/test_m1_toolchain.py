"""
tests/test_m1_toolchain.py
Unit and integration test suite for Worker M1 deliverables:
- tools/acquire_plans.py and downloaded plans in dxf/externos/
- tools/convert_dwg_to_dxf.py with accoreconsole.exe
- Local Argentine CAD plans ingestion
- detector_pack/detector_unifilar.py updates (--d_min and HatchPolicy.NORMAL)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import ezdxf
import ezdxf.bbox

# Workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.acquire_plans import CURATED_PLANS, validate_dxf
from tools.convert_dwg_to_dxf import convert_dwg_to_dxf, find_accoreconsole
from detector_pack.detector_unifilar import DetectorUnifilar, render_dxf


class TestWorkerM1Deliverables(unittest.TestCase):

    def test_curated_web_plans_acquired(self):
        """Verifies that all 4 web plans exist in dxf/externos/ and validate with ezdxf."""
        externos_dir = WORKSPACE_ROOT / "dxf" / "externos"
        self.assertTrue(externos_dir.exists(), "dxf/externos directory does not exist")

        expected_files = [
            "05_diagrama_unifilar.dxf",
            "01_diagrama_unifilar.dxf",
            "01_diagrama_unifilar_ccm.dxf",
            "03_diagrama_unifilar_ca.dxf"
        ]

        for fname in expected_files:
            fpath = externos_dir / fname
            self.assertTrue(fpath.exists(), f"Expected plan not found: {fname}")
            self.assertGreater(fpath.stat().st_size, 50000, f"File {fname} is too small (<50KB)")

            val = validate_dxf(fpath)
            self.assertTrue(val["valid"], f"Validation failed for {fname}")
            self.assertEqual(val["dxf_version"], "AC1027", f"Unexpected DXF version for {fname}")
            self.assertGreater(val["total_msp_entities"], 0, f"No entities in {fname}")
            self.assertIsNotNone(val["bbox"], f"No bounding box for {fname}")
            self.assertGreater(val["bbox"]["width"], 0, f"Zero width for {fname}")
            self.assertGreater(val["bbox"]["height"], 0, f"Zero height for {fname}")

    def test_local_argentine_plans_exist_and_valid(self):
        """Verifies that local Argentine plans in dxf/ are valid DXF files."""
        local_plans = [
            "Tablerotsbe.dxf",
            "plano.dxf",
            "plano2.dxf",
            "plano3.dxf",
            "plano4.dxf"
        ]
        for p_name in local_plans:
            p_path = WORKSPACE_ROOT / "dxf" / p_name
            self.assertTrue(p_path.exists(), f"Local plan {p_name} does not exist")
            doc = ezdxf.readfile(str(p_path))
            self.assertIsNotNone(doc.header, f"Header missing in {p_name}")
            msp = doc.modelspace()
            self.assertGreater(len(list(msp)), 0, f"No entities in {p_name}")

    def test_dwg_to_dxf_converter(self):
        """Verifies accoreconsole DWG to DXF conversion pipeline."""
        accore_exe = find_accoreconsole()
        self.assertTrue(accore_exe.exists(), f"accoreconsole.exe not found at {accore_exe}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_dwg = tmp_path / "test_input.dwg"
            sample_dxf = tmp_path / "test_output.dxf"

            # Create a synthetic DWG using accoreconsole
            sample_dwg_str = str(sample_dwg).replace("\\", "/")
            scr_content = f'_LINE\n0,0\n50,50\n\n_CIRCLE\n25,25\n10\n_SAVEAS\n2018\n"{sample_dwg_str}"\n_QUIT\n_Y\n'
            scr_file = tmp_path / "create.scr"
            with open(scr_file, "w") as f:
                f.write(scr_content)

            import subprocess
            res_create = subprocess.run(
                [str(accore_exe), "/product", "ACAD_E", "/s", str(scr_file)],
                capture_output=True,
                timeout=30
            )
            self.assertEqual(res_create.returncode, 0, "Failed to create synthetic test DWG")
            self.assertTrue(sample_dwg.exists(), "Synthetic DWG file not created")

            # Convert synthetic DWG to DXF using our convert_dwg_to_dxf function
            success = convert_dwg_to_dxf(
                dwg_path=str(sample_dwg),
                dxf_path=str(sample_dxf),
                product="ACAD_E",
                verbose=False
            )
            self.assertTrue(success, "convert_dwg_to_dxf returned False")
            self.assertTrue(sample_dxf.exists(), "Output DXF was not generated")
            self.assertGreater(sample_dxf.stat().st_size, 0, "Output DXF is empty")

            # Verify DXF structure
            doc = ezdxf.readfile(str(sample_dxf))
            self.assertIn(doc.dxfversion, ["AC1032", "AC1027", "AC1024"])
            msp = doc.modelspace()
            entity_types = [e.dxftype() for e in msp]
            self.assertIn("LINE", entity_types)
            self.assertIn("CIRCLE", entity_types)

    def test_detector_configuration_and_d_min(self):
        """Verifies that detector_unifilar.py renders with HatchPolicy.NORMAL and supports d_min."""
        dxf_tsbe = WORKSPACE_ROOT / "dxf" / "Tablerotsbe.dxf"
        self.assertTrue(dxf_tsbe.exists())

        # Test render_dxf
        img_bgr, meta = render_dxf(dxf_tsbe, px_per_cad=75.0, is_industrial=True)
        self.assertIsNotNone(img_bgr)
        self.assertEqual(len(img_bgr.shape), 3)
        self.assertGreater(img_bgr.shape[0], 500)
        self.assertGreater(img_bgr.shape[1], 1000)
        self.assertIn("px_per_cad", meta)
        self.assertEqual(meta["px_per_cad"], 75.0)

        # Test detector with d_min=0.20
        model_path = WORKSPACE_ROOT / "detector_pack" / "best_componente_nano.pt"
        self.assertTrue(model_path.exists())

        detector = DetectorUnifilar(model_path=str(model_path))
        with tempfile.TemporaryDirectory() as tmpdir:
            res = detector.detectar_dxf(
                dxf_path=str(dxf_tsbe),
                output_dir=tmpdir,
                conf_thresh=0.50,
                px_per_cad=75.0,
                d_min=0.20
            )
            self.assertEqual(res["total"], 15, f"Expected 15 components on Tablerotsbe, got {res['total']}")
            self.assertTrue(Path(res["vis_image"]).exists())
            self.assertTrue(Path(res["csv_path"]).exists())
            self.assertTrue(Path(res["json_path"]).exists())


if __name__ == "__main__":
    unittest.main()
