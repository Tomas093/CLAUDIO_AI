"""
Test Helpers and Fixture Generators for CLAUDIO_AI E2E Test Suite.
Provides synthetic DXF generators, image mocks, ground-truth metrics calculators,
and path constants.
"""

import os
import sys
import math
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import ezdxf

# Project root and key paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DETECTOR_PACK_DIR = PROJECT_ROOT / "detector_pack"
DETECTOR_MODEL_PATH = DETECTOR_PACK_DIR / "best_componente_nano.pt"
DXF_DIR = PROJECT_ROOT / "dxf"
TEST_DIR = PROJECT_ROOT / "test"

# Ensure detector_pack and root are in sys.path
for p in [str(PROJECT_ROOT), str(DETECTOR_PACK_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def create_minimal_dxf(file_path: str, dxfversion: str = "R2010") -> str:
    """Creates a minimal valid ASCII DXF with a few lines."""
    doc = ezdxf.new(dxfversion=dxfversion)
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    msp.add_line((10, 10), (0, 10))
    msp.add_line((0, 10), (0, 0))
    doc.saveas(file_path)
    return file_path


def create_synthetic_switchboard_dxf(file_path: str, num_circuits: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Creates a realistic synthetic single-line diagram DXF following AEA 90364 conventions:
    - Main disconnector (Q0)
    - Differential protection (ID1)
    - Branch miniature circuit breakers (PIA1 to PIAn)
    - Terminals (X1 to Xn)
    Returns file_path and ground-truth component locations.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Define layers
    doc.layers.new(name="UNIFILAR", dxfattribs={"color": 7})
    doc.layers.new(name="IE-UN-TEXTOS", dxfattribs={"color": 2})

    gt_components = []

    # Busbar
    bus_x1 = 5.0
    bus_x2 = 5.0 + num_circuits * 4.0
    bus_y = 20.0
    msp.add_line((bus_x1 - 2.0, bus_y), (bus_x2 + 2.0, bus_y), dxfattribs={"layer": "UNIFILAR"})

    # Main incoming feeder
    feed_x = bus_x1 + 1.0
    msp.add_line((feed_x, bus_y + 10.0), (feed_x, bus_y), dxfattribs={"layer": "UNIFILAR"})

    # Main breaker Q0 at (feed_x, bus_y + 6.0)
    q0_xc, q0_yc = feed_x, bus_y + 6.0
    msp.add_lwpolyline([
        (q0_xc - 0.5, q0_yc - 0.7),
        (q0_xc + 0.5, q0_yc - 0.7),
        (q0_xc + 0.5, q0_yc + 0.7),
        (q0_xc - 0.5, q0_yc + 0.7)
    ], close=True, dxfattribs={"layer": "UNIFILAR"})
    gt_components.append({
        "name": "PIA_GENERAL",
        "xc": q0_xc, "yc": q0_yc,
        "bbox": [q0_xc - 0.5, q0_yc - 0.7, q0_xc + 0.5, q0_yc + 0.7],
        "type": "PIA"
    })

    # Main differential ID0 at (feed_x, bus_y + 3.0)
    id0_xc, id0_yc = feed_x, bus_y + 3.0
    msp.add_circle((id0_xc, id0_yc), radius=0.6, dxfattribs={"layer": "UNIFILAR"})
    gt_components.append({
        "name": "ID_GENERAL",
        "xc": id0_xc, "yc": id0_yc,
        "bbox": [id0_xc - 0.6, id0_yc - 0.6, id0_xc + 0.6, id0_yc + 0.6],
        "type": "ID"
    })

    # Branch circuits
    for i in range(num_circuits):
        cx = bus_x1 + 2.0 + i * 4.0
        # Line from busbar down
        msp.add_line((cx, bus_y), (cx, bus_y - 12.0), dxfattribs={"layer": "UNIFILAR"})

        # Circuit breaker PIA_i
        pia_y = bus_y - 4.0
        msp.add_lwpolyline([
            (cx - 0.4, pia_y - 0.6),
            (cx + 0.4, pia_y - 0.6),
            (cx + 0.4, pia_y + 0.6),
            (cx - 0.4, pia_y + 0.6)
        ], close=True, dxfattribs={"layer": "UNIFILAR"})
        gt_components.append({
            "name": f"PIA_C{i+1}",
            "xc": cx, "yc": pia_y,
            "bbox": [cx - 0.4, pia_y - 0.6, cx + 0.4, pia_y + 0.6],
            "type": "PIA"
        })

        # Terminal block X_i at bottom
        term_y = bus_y - 11.0
        msp.add_circle((cx, term_y), radius=0.3, dxfattribs={"layer": "UNIFILAR"})
        gt_components.append({
            "name": f"BORNERA_C{i+1}",
            "xc": cx, "yc": term_y,
            "bbox": [cx - 0.3, term_y - 0.3, cx + 0.3, term_y + 0.3],
            "type": "BORNERA"
        })

        # Circuit text (in excluded text layer)
        msp.add_text(f"Cto {i+1}: Tomas 2x16A", dxfattribs={"layer": "IE-UN-TEXTOS", "height": 0.8, "insert": (cx + 0.6, pia_y)})

    doc.saveas(file_path)
    return file_path, gt_components


def create_test_image(file_path: str, width: int = 1200, height: int = 800, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> str:
    """Creates a blank test image and saves as PNG."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Draw simple circuit shapes
    cv2.line(img, (100, 400), (1100, 400), (0, 0, 0), 2)
    cv2.rectangle(img, (200, 350), (250, 450), (0, 0, 0), 2)
    cv2.circle(img, (500, 400), 30, (0, 0, 0), 2)
    cv2.imwrite(file_path, img)
    return file_path


def compute_greedy_matching(detections: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], dist_tol: float = 1.0) -> Dict[str, Any]:
    """
    Evaluates detections against ground truth using standard greedy 1-to-1 matching.
    Calculates TP, FN, FP, Recall, Precision, and F1.
    """
    candidates = []
    for g_idx, g in enumerate(ground_truth):
        for d_idx, d in enumerate(detections):
            dist = math.hypot(g['xc'] - d['xc'], g['yc'] - d['yc'])
            inside = False
            if 'bbox_cad' in d:
                b = d['bbox_cad']
                inside = (b[0] <= g['xc'] <= b[2] and b[1] <= g['yc'] <= b[3])
            if dist <= dist_tol or inside:
                candidates.append((dist, g_idx, d_idx))

    candidates.sort(key=lambda x: x[0])
    matched_gt = set()
    matched_det = set()

    for dist, g_idx, d_idx in candidates:
        if g_idx not in matched_gt and d_idx not in matched_det:
            matched_gt.add(g_idx)
            matched_det.add(d_idx)

    tp = len(matched_gt)
    fn = len(ground_truth) - tp
    fp = len(detections) - len(matched_det)
    rec = (tp / len(ground_truth)) * 100.0 if ground_truth else 100.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "total_gt": len(ground_truth),
        "total_det": len(detections)
    }
