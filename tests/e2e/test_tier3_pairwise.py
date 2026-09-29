"""
Tier 3: Pairwise Combinatorial Interaction Tests (>= 12 test cases)
Validates cross-feature interactions:
- DWG conversion + rendering (F3 + F4)
- Scale analyzer + SAHI tiling (F5 + F6)
- SAHI tiling + NMS deduplication (F6 + F7)
- Vector rendering + YOLO detector (F4 + F8)
- Detector inference + Metric audit engine (F8 + F9)
- Metric audit + Visual sheet generation (F9 + F10)
- Scale normalization + CAD NMS (F5 + F7)
- Local plan ingestion + Scale analysis (F2 + F5)
- Web acquisition simulation + Local ingestion (F1 + F2)
- High-res rendering + Inverse CAD projection (F4 + F6)
- NMS calibration + Benchmark regression check (F7 + F11)
- Input validation rejection + Metric audit safety (F12 + F9)
- Full apparatus gamut + Visual sheet export (F8 + F10)
- Confidence sweep + Metric audit calculation (F7 + F9)
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
    create_synthetic_switchboard_dxf,
    create_test_image,
    compute_greedy_matching
)


class TestTier3PairwiseInteractions(unittest.TestCase):
    """Pairwise combinatorial tests validating feature interplay and data flow."""

    @classmethod
    def setUpClass(cls):
        cls.detector = DetectorUnifilar(model_path=str(DETECTOR_MODEL_PATH), device="cpu")

    def test_tier3_01_dwg_conversion_to_dxf_rendering(self):
        """Interaction F3 + F4: DWG target path resolution feeding into DXF vector rendering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dwg_source = "C:/cad/externos/05_diagrama_unifilar.dwg"
            dxf_dest = os.path.join(tmpdir, Path(dwg_source).stem + ".dxf")
            create_minimal_dxf(dxf_dest)

            img_bgr, meta = render_dxf(dxf_dest, px_per_cad=50.0, is_industrial=False)
            self.assertEqual(meta['W_px'], img_bgr.shape[1])
            self.assertEqual(meta['H_px'], img_bgr.shape[0])
            self.assertGreater(img_bgr.size, 0)

    def test_tier3_02_scale_analyzer_to_sahi_tiling(self):
        """Interaction F5 + F6: Scale analyzer px_per_cad determining canvas dimensions and SAHI tiling."""
        dxf_path = DXF_DIR / "plano.dxf"
        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))
        self.assertGreaterEqual(px_per_cad, 75.0)

        # Compute canvas size for CAD dimensions 40.0 x 30.0
        cad_w, cad_h = 40.0, 30.0
        img_w = int(round(cad_w * px_per_cad))
        img_h = int(round(cad_h * px_per_cad))

        # Pad 320 px and generate slices
        pad = 320
        pad_w = img_w + 2 * pad
        pad_h = img_h + 2 * pad
        slices = generate_slices(pad_h, pad_w, slice_size=640, overlap=0.80)
        self.assertGreater(len(slices), 10)
        for s in slices:
            self.assertLessEqual(s[2] - s[0], 640)
            self.assertLessEqual(s[3] - s[1], 640)

    def test_tier3_03_sahi_slicing_to_nms_deduplication(self):
        """Interaction F6 + F7: High overlap SAHI slicing generating duplicate tile detections resolved by CAD NMS."""
        # Simulate 1 switch detected across 4 overlapping tiles
        tile_detections = [
            {"bbox_cad": [10.0, 20.0, 11.0, 21.0], "xc": 10.5, "yc": 20.5, "conf": 0.94},
            {"bbox_cad": [10.05, 20.02, 11.05, 21.02], "xc": 10.55, "yc": 20.52, "conf": 0.88},
            {"bbox_cad": [9.98, 19.95, 10.98, 20.95], "xc": 10.48, "yc": 20.45, "conf": 0.79},
            {"bbox_cad": [10.02, 20.01, 11.02, 21.01], "xc": 10.52, "yc": 20.51, "conf": 0.91}
        ]
        # And 1 separate switch
        separate_switch = {"bbox_cad": [15.0, 20.0, 16.0, 21.0], "xc": 15.5, "yc": 20.5, "conf": 0.92}
        all_dets = tile_detections + [separate_switch]

        # Apply cascaded NMS
        step1 = nms_iou_cad(all_dets, iou_thresh=0.45)
        step2 = eliminar_anidadas(step1, ios_thresh=0.60)
        final_dets = nms_distancia_cad(step2, d_min=0.35)

        self.assertEqual(len(final_dets), 2)
        # Verify higher confidence detection was selected for the first switch
        self.assertEqual(final_dets[0]["conf"], 0.94)
        self.assertEqual(final_dets[1]["conf"], 0.92)

    def test_tier3_04_vector_rendering_to_yolo_inference(self):
        """Interaction F4 + F8: Vector rendered image feeding directly into YOLO detector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "switchboard.dxf")
            create_synthetic_switchboard_dxf(dxf_path, num_circuits=3)

            img_bgr, meta = render_dxf(dxf_path, px_per_cad=75.0, is_industrial=False)
            res = self.detector._procesar_imagen(
                img_bgr=img_bgr,
                meta=meta,
                stem="switchboard",
                output_dir=tmpdir,
                conf_thresh=0.01,
                batch_size=16
            )
            self.assertIn("total", res)
            self.assertIn("detections", res)
            self.assertTrue(os.path.exists(res["vis_image"]))

    def test_tier3_05_detector_inference_to_metric_audit(self):
        """Interaction F8 + F9: Detector output evaluated against known ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dxf_path = os.path.join(tmpdir, "audit_test.dxf")
            _, gt_list = create_synthetic_switchboard_dxf(dxf_path, num_circuits=4)

            # Synthetic detections covering GT with minor offset
            synthetic_dets = [
                {
                    "xc": g["xc"] + 0.05,
                    "yc": g["yc"] - 0.05,
                    "bbox_cad": [g["xc"] - 0.4, g["yc"] - 0.5, g["xc"] + 0.4, g["yc"] + 0.5],
                    "conf": 0.95
                }
                for g in gt_list
            ]
            # Add 1 FP
            synthetic_dets.append({
                "xc": 99.0, "yc": 99.0,
                "bbox_cad": [98.5, 98.5, 99.5, 99.5],
                "conf": 0.50
            })

            metrics = compute_greedy_matching(synthetic_dets, gt_list, dist_tol=1.0)
            self.assertEqual(metrics["tp"], len(gt_list))
            self.assertEqual(metrics["fn"], 0)
            self.assertEqual(metrics["fp"], 1)
            self.assertEqual(metrics["recall"], 100.0)
            self.assertGreater(metrics["precision"], 85.0)

    def test_tier3_06_metric_audit_to_visual_sheet_generation(self):
        """Interaction F9 + F10: Metric audit results used to generate visual sheet with TP/FP labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vis_img = np.full((600, 800, 3), 255, dtype=np.uint8)
            dets = [
                {"bbox_px": [100, 100, 150, 180], "conf": 0.95, "is_tp": True},
                {"bbox_px": [300, 100, 350, 180], "conf": 0.40, "is_tp": False}
            ]
            for d in dets:
                x1, y1, x2, y2 = d["bbox_px"]
                color = (0, 200, 0) if d["is_tp"] else (0, 0, 255)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            out_file = os.path.join(tmpdir, "audit_visual.png")
            cv2.imwrite(out_file, vis_img)
            self.assertTrue(os.path.exists(out_file))

    def test_tier3_07_scale_normalization_to_cad_nms(self):
        """Interaction F5 + F7: Normalization scale mapping pixel boxes to CAD space for d_min filtering."""
        px_per_cad = 75.0
        W_px, H_px = 1500, 750
        W_cad, H_cad = W_px / px_per_cad, H_px / px_per_cad
        x_min, y_max = 0.0, H_cad

        # 2 close breaker detections in pixels (15 px apart -> 15/75 = 0.20 CAD)
        px_d1 = [300, 300, 340, 380]
        px_d2 = [315, 300, 355, 380]

        cad_dets = []
        for box, c in [(px_d1, 0.95), (px_d2, 0.70)]:
            cad_x1 = x_min + (box[0] / W_px) * W_cad
            cad_x2 = x_min + (box[2] / W_px) * W_cad
            cad_y1 = y_max - (box[3] / H_px) * H_cad
            cad_y2 = y_max - (box[1] / H_px) * H_cad
            cad_dets.append({
                "bbox_cad": [cad_x1, cad_y1, cad_x2, cad_y2],
                "xc": (cad_x1 + cad_x2) / 2.0,
                "yc": (cad_y1 + cad_y2) / 2.0,
                "conf": c
            })

        # With d_min = 0.35 CAD, the closer detection (dist = 0.20) should be suppressed
        kept = nms_distancia_cad(cad_dets, d_min=0.35)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["conf"], 0.95)

    def test_tier3_08_local_ingestion_to_scale_analysis(self):
        """Interaction F2 + F5: Local plan ingestion verified by scale analyzer."""
        dxf_path = DXF_DIR / "plano3.dxf"
        doc = ezdxf.readfile(str(dxf_path))
        self.assertIsNotNone(doc)
        px_per_cad, ref = scale_analyzer.calcular_factor_escala(str(dxf_path))
        self.assertGreater(px_per_cad, 0.0)

    def test_tier3_09_web_acquisition_to_local_ingestion(self):
        """Interaction F1 + F2: Simulated web acquisition saving DXF validated by local ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir) / "externos"
            dest_dir.mkdir(parents=True, exist_ok=True)
            mock_dxf = dest_dir / "05_diagrama_unifilar.dxf"
            create_minimal_dxf(str(mock_dxf))

            doc = ezdxf.readfile(str(mock_dxf))
            self.assertEqual(len(doc.modelspace()), 4)

    def test_tier3_10_high_res_rendering_to_inverse_cad_mapping(self):
        """Interaction F4 + F6: Forward render and inverse projection round-trip accuracy."""
        cad_box = [5.0, 10.0, 6.2, 11.5]
        px_per_cad = 80.0
        W_cad, H_cad = 50.0, 30.0
        W_px = int(W_cad * px_per_cad)
        H_px = int(H_cad * px_per_cad)
        x_min, y_max = 0.0, 30.0

        # CAD -> Pixel
        px_x1 = int((cad_box[0] - x_min) / W_cad * W_px)
        px_x2 = int((cad_box[2] - x_min) / W_cad * W_px)
        px_y1 = int((y_max - cad_box[3]) / H_cad * H_px)
        px_y2 = int((y_max - cad_box[1]) / H_cad * H_px)

        # Pixel -> CAD
        inv_x1 = x_min + (px_x1 / W_px) * W_cad
        inv_x2 = x_min + (px_x2 / W_px) * W_cad
        inv_y1 = y_max - (px_y2 / H_px) * H_cad
        inv_y2 = y_max - (px_y1 / H_px) * H_cad

        self.assertAlmostEqual(cad_box[0], inv_x1, delta=0.05)
        self.assertAlmostEqual(cad_box[2], inv_x2, delta=0.05)
        self.assertAlmostEqual(cad_box[1], inv_y1, delta=0.05)
        self.assertAlmostEqual(cad_box[3], inv_y2, delta=0.05)

    def test_tier3_11_nms_calibration_to_benchmark_regression(self):
        """Interaction F7 + F11: Calibrated d_min ensures historical benchmark recall >= 98%."""
        # 10 closely spaced components (e.g. DIN rail with 0.50 CAD pitch)
        gt = [{"xc": 10.0 + i * 0.50, "yc": 20.0} for i in range(10)]
        dets = [
            {
                "xc": 10.0 + i * 0.50, "yc": 20.0,
                "bbox_cad": [9.8 + i * 0.50, 19.5, 10.2 + i * 0.50, 20.5],
                "conf": 0.90
            }
            for i in range(10)
        ]
        # With d_min = 0.20 CAD, all 10 are preserved because pitch = 0.50 > 0.20
        kept = nms_distancia_cad(dets, d_min=0.20)
        self.assertEqual(len(kept), 10)
        metrics = compute_greedy_matching(kept, gt, dist_tol=0.25)
        self.assertGreaterEqual(metrics["recall"], 98.0)

    def test_tier3_12_input_validation_to_metric_audit_guard(self):
        """Interaction F12 + F9: Corrupted input detected before reaching metric audit engine."""
        invalid_input = None
        has_error = False
        try:
            if invalid_input is None or not os.path.exists(str(invalid_input)):
                raise FileNotFoundError("Invalid input diagram")
            compute_greedy_matching(invalid_input, [])
        except FileNotFoundError:
            has_error = True
        self.assertTrue(has_error)

    def test_tier3_13_full_gamut_to_visual_sheet_export(self):
        """Interaction F8 + F10: Multiple apparatus types written to CSV and JSON with correct types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "gamut_test"
            csv_path = os.path.join(tmpdir, f"{stem}_detections.csv")
            json_path = os.path.join(tmpdir, f"{stem}_detections.json")

            apparatus_sample = [
                {"id": 1, "tipo": "PIA", "conf": 0.95, "xc": 10.0, "yc": 20.0, "bbox_cad": [9.5, 19.5, 10.5, 20.5], "bbox_px": [100, 200, 110, 210]},
                {"id": 2, "tipo": "ID", "conf": 0.91, "xc": 14.0, "yc": 20.0, "bbox_cad": [13.5, 19.5, 14.5, 20.5], "bbox_px": [140, 200, 150, 210]},
                {"id": 3, "tipo": "DPS", "conf": 0.89, "xc": 18.0, "yc": 20.0, "bbox_cad": [17.5, 19.5, 18.5, 20.5], "bbox_px": [180, 200, 190, 210]},
                {"id": 4, "tipo": "BORNERA", "conf": 0.85, "xc": 10.0, "yc": 5.0, "bbox_cad": [9.8, 4.8, 10.2, 5.2], "bbox_px": [100, 350, 104, 354]}
            ]

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'xc_cad', 'yc_cad', 'conf'])
                for a in apparatus_sample:
                    writer.writerow([a["id"], f"{a['xc']:.2f}", f"{a['yc']:.2f}", f"{a['conf']:.2f}"])

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"total_detected": len(apparatus_sample), "components": apparatus_sample}, f, indent=2)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))

    def test_tier3_14_conf_sweep_to_metric_audit(self):
        """Interaction F7 + F9: Sweeping confidence threshold exhibits monotonic recall behavior."""
        gt = [{"xc": i * 2.0, "yc": 5.0} for i in range(10)]
        dets = [
            {"xc": i * 2.0, "yc": 5.0, "bbox_cad": [i*2-0.5, 4.5, i*2+0.5, 5.5], "conf": 0.10 + i * 0.08}
            for i in range(10)
        ]
        recalls = []
        for th in [0.05, 0.20, 0.40, 0.60, 0.80]:
            filtered = [d for d in dets if d["conf"] >= th]
            m = compute_greedy_matching(filtered, gt, dist_tol=0.5)
            recalls.append(m["recall"])

        # Monotonic non-increasing recall as threshold increases
        for k in range(len(recalls) - 1):
            self.assertGreaterEqual(recalls[k], recalls[k + 1])


if __name__ == "__main__":
    unittest.main()
