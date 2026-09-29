"""
Unit and integration tests for tools/audit_engine.py and CLAUDIO_AI audit pipeline.
"""

import os
import sys
import csv
import json
import unittest
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_engine import AuditEngine, PlanSpec


class TestAuditEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = PROJECT_ROOT / "output_eval"
        cls.engine = AuditEngine(
            model_path="detector_pack/best_componente_nano.pt",
            output_dir=str(cls.output_dir)
        )

    def test_plan_inventory(self):
        """Verifies that all required plans across web, local, and benchmarks are registered."""
        plans = self.engine.get_default_plans()
        categories = {p.category for p in plans}
        self.assertIn("web", categories)
        self.assertIn("local_argentine", categories)
        self.assertIn("benchmark", categories)

        keys = [p.key for p in plans]
        # Web plans
        self.assertIn("web_residencial_05", keys)
        self.assertIn("web_motor_01", keys)
        self.assertIn("web_ccm_01", keys)
        self.assertIn("web_fotovoltaico_03", keys)
        # Local plans
        self.assertIn("local_tsbe", keys)
        self.assertIn("local_plano", keys)
        self.assertIn("local_plano2", keys)
        self.assertIn("local_plano3", keys)
        self.assertIn("local_plano4", keys)
        self.assertIn("local_plano5", keys)
        self.assertIn("local_ocj_master", keys)
        # Benchmarks
        self.assertIn("bench_test1_comp", keys)
        self.assertIn("bench_test2_comp", keys)
        self.assertIn("bench_fl_un_02_comp", keys)
        self.assertIn("bench_tsss_2", keys)
        self.assertIn("bench_vyre_comp", keys)

    def test_evaluation_prevalidated_tsbe(self):
        """Evaluates Tablerotsbe.dxf and asserts 100% recall with zero false negatives."""
        tsbe_spec = next(p for p in self.engine.get_default_plans() if p.key == "local_tsbe")
        res = self.engine.evaluate_plan(tsbe_spec)

        self.assertEqual(res['total_detected'], 15)
        self.assertIsNotNone(res['metrics'])
        self.assertGreaterEqual(res['metrics']['recall'], 98.0)
        self.assertEqual(res['metrics']['fn'], 0)
        self.assertEqual(res['metrics']['tp'], 15)

    def test_artifact_compliance(self):
        """Verifies CSV and JSON artifacts match interface contracts in PROJECT.md."""
        csv_path = self.output_dir / "Tablerotsbe_detections.csv"
        json_path = self.output_dir / "Tablerotsbe_detections.json"
        png_path = self.output_dir / "Tablerotsbe_visual_detections.png"

        self.assertTrue(csv_path.exists(), "CSV detection table must exist")
        self.assertTrue(json_path.exists(), "JSON detection metadata must exist")
        self.assertTrue(png_path.exists(), "Visual sheet PNG must exist")

        # Verify CSV format
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected_headers = ['id', 'xc_cad', 'yc_cad', 'x1_cad', 'y1_cad', 'x2_cad', 'y2_cad', 'conf', 'px_x1', 'px_y1', 'px_x2', 'px_y2']
            self.assertEqual(headers, expected_headers)
            rows = list(reader)
            self.assertEqual(len(rows), 15)

        # Verify JSON format
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('metadata', data)
            self.assertIn('total_detected', data)
            self.assertIn('components', data)
            self.assertEqual(data['total_detected'], 15)
            self.assertEqual(len(data['components']), 15)

    def test_audit_report_generation(self):
        """Verifies that AUDIT_REPORT.md exists at project root and passes certification."""
        report_path = PROJECT_ROOT / "AUDIT_REPORT.md"
        self.assertTrue(report_path.exists(), "AUDIT_REPORT.md must exist at project root")

        content = report_path.read_text(encoding='utf-8')
        self.assertIn("100% APROBADO (CERO REGRESIONES)", content)
        self.assertIn("100.0%", content)
        self.assertIn("Tablerotsbe.dxf", content)
        self.assertIn("FL-UN-02_tablero_1.dxf", content)
        self.assertIn("UNIFILAR TABLERO GENERAL Vyre 09 09 2026.dxf", content)


if __name__ == '__main__':
    unittest.main()
