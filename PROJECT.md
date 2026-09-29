# Project: CLAUDIO_AI - Detector Universal de Planos Eléctricos Unifilares Argentinos (AEA 90364 / IRAM)

## Architecture
CLAUDIO_AI provides an autonomous end-to-end intelligence system for electrical single-line computer-aided design (CAD) diagrams (DWG and DXF) according to Argentine electrical regulations (AEA 90364 and IRAM standards).

```
[Web Open Repos / Local DWG/DXF]
           │
           ▼
[Milestone 1: Acquisition & Ingestion] ──► [DWG->DXF Headless Converter (accoreconsole)]
           │
           ▼ (Clean DXF)
[Milestone 2: Vector Rendering & Normalization]
    ├── ColorPolicy.COLOR_SWAP_BW + HatchPolicy.NORMAL
    └── Spatial Scale Normalization (75-100 px/CAD)
           │
           ▼ (High-Res Rendered Plan + CAD Metadata)
[Milestone 3: Universal YOLO Nano Inference & NMS]
    ├── best_componente_nano.pt (Universal Class: componente)
    ├── SAHI Tiling: 640x640, 80% Overlap, 320 px Pad
    └── Cascaded CAD NMS: IoU 0.45, IoS 0.60, Centroidal d_min ~ 0.20 CAD
           │
           ▼ (Detections in CAD & Pixel Coordinates)
[Milestone 4: Metric Auditing, Sheets & Regression]
    ├── Metric Engine: Recall, Precision, F1 against Ground Truth
    ├── Visual Sheets ({stem}_visual_detections.png)
    └── Historical Benchmark Regression Verifier (TEST 1, TEST 2, FL-UN-02, TSSS_2, Vyre)
           │
           ▼
[Milestone 5: E2E Acceptance & Adversarial Hardening]
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1: Web Autonomous Acquisition | Autonomous search and download of >= 3 open-source CAD electrical unifilar diagrams (DWG/DXF) from public repositories | M1 | Survey E3 / Req R1 |
| 2 | F2: Local Ingestion & Cataloging | Ingestion and structural processing of local Argentine plans (`Tablerotsbe`, `plano.dxf` to `plano5.dxf`, `OCJ-DE-IEL-...`) | M1 | Survey E2 / Req R1 |
| 3 | F3: Headless DWG to DXF Converter | Automated conversion from DWG to clean DXF using `accoreconsole.exe /product "ACAD_E"` | M1 | Survey E3 / Req R2 |
| 4 | F4: Vector Rendering Engine | Rasterization with `ColorPolicy.COLOR_SWAP_BW`, `HatchPolicy.NORMAL`, and matplotlib `Agg` backend | M2 | Survey E1 / Req R2 |
| 5 | F5: Scale Normalization | Normalization to 75-100 px/CAD based on modular switchgear dimensions (0.8 - 1.2 CAD units) | M2 | Survey E1, E2 / Req R2 |
| 6 | F6: SAHI Tiling & Inversion | 640x640 sliding tiles with 80% overlap, 320 px constant white padding, batch GPU inference, and inverse CAD projection | M3 | Survey E1 / Req R3 |
| 7 | F7: Centroidal NMS Calibration | Cascaded NMS in CAD space with calibrated $d_{\min} \approx 0.20$ CAD for Argentine DIN compact rails | M3 | Survey E1 / Req R3 |
| 8 | F8: Full Apparatus Gamut | Detection of PIAs, IDs, seccionadores, DPS, generator sets, CT/VT transformers, pilot lights, and terminals (ø and solid) | M3 | Survey E1 / Req R3 |
| 9 | F9: Metric Audit Engine | Computation of Recall, Precision, F1, TP, FN, FP per diagram evaluated against ground truth | M4 | Survey E2 / Req R4 |
| 10 | F10: High-Res Visual Sheets | Generation of `{stem}_visual_detections.png` with green bounding boxes and confidence labels, CSV and JSON outputs | M4 | Survey E1 / Req R4 |
| 11 | F11: Zero-Regression Benchmarks | Verification of no regression (Recall >= 98%, zero FN) on `TEST 1`, `TEST 2`, `FL-UN-02`, `TSSS_2`, `Vyre` | M4 | Survey E2 / Req R4 |
| 12 | F12: E2E Acceptance & Stress Testing | Verification against full E2E test suite (Tiers 1-4) and adversarial coverage hardening (Tier 5) | M5 | Dual Track / Req R1-R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Planos Acquisition & Ingestion | Download >=3 web unifilar plans, local CAD ingestion, DWG->DXF converter utility | None | DONE |
| M2 | Vector Rendering & Normalization | Verify & standardize `ColorPolicy.COLOR_SWAP_BW`, `HatchPolicy.NORMAL`, 75-100 px/CAD scale engine | M1 | DONE |
| M3 | Universal Detection Pipeline | Refine `detector_unifilar.py`, configure $d_{\min} \approx 0.20$ CAD, sliding tiles, full gamut inference | M2 | DONE |
| M4 | Metric Auditing & Visual Sheets | Run full audit across all plans, generate visual sheets, verify historical benchmarks (zero-regression) | M3 | DONE |
| M5 | E2E Acceptance & Adversarial Hardening | Validate against E2E test suite published in TEST_READY.md and adversarial stress testing | M4 | DONE |

## Interface Contracts

### M1 ↔ M2 (Acquisition/Conversion ↔ Rendering)
- Input: `.dwg` or `.dxf` files placed in `dxf/` or `dxf/externos/`.
- Output: Valid, readable ASCII DXF files (AC1009 to AC1032) verified by `ezdxf.readfile()`.
- Metadata: File paths, file sizes, version string.

### M2 ↔ M3 (Rendering ↔ Detection)
- Rendering output: High-resolution PNG image `{stem}.png` rendered on white background with inverted black line work.
- Transformation metadata: `{stem}_meta.json` containing:
  ```json
  {
    "img_w": int, "img_h": int,
    "cad_xmin": float, "cad_ymin": float, "cad_xmax": float, "cad_ymax": float,
    "px_per_cad": float,
    "scale_factor": float
  }
  ```

### M3 ↔ M4 (Detection ↔ Auditing & Sheets)
- Detection results: `{stem}_detections.json` and `{stem}_detections.csv`:
  ```json
  {
    "total_detected": int,
    "confidence_threshold": float,
    "detections": [
      {
        "id": int,
        "cad_bbox": [x1, y1, x2, y2],
        "cad_centroid": [xc, yc],
        "pixel_bbox": [px1, py1, px2, py2],
        "confidence": float,
        "class": "componente"
      }
    ]
  }
  ```
- Visual sheet: `{stem}_visual_detections.png` with BGR(0, 200, 0) bounding boxes and text labels.

### M4 ↔ M5 (Auditing ↔ E2E Acceptance)
- Audit summary report: `AUDIT_REPORT.md` documenting Recall, Precision, and regression analysis for all plans.
- All evaluation criteria met: Recall >= 98%, zero regressions on historical benchmarks, zero systematic false positives on blank cables or text.

## Code Layout
- `detector_pack/`: Packaged detector core (`detector_unifilar.py`, `best_componente_nano.pt`).
- `tools/` / workspace scripts:
  - `convert_dwg_to_dxf.py`: Headless DWG to DXF converter.
  - `acquire_plans.py`: Autonomous web plan downloader.
  - `dxf_to_image.py`: DXF vector renderer.
  - `scale_analyzer.py`: Optimal px/CAD analyzer.
  - `audit_engine.py` / `eval_benchmark.py`: Metric computation and regression verification.
- `dxf/`: Local and acquired DXF files (`Tablerotsbe.dxf`, `plano.dxf`, `plano2.dxf`, `plano3.dxf`, `plano4.dxf`, `plano5.dxf`, `OCJ-DE-IEL-UNI-000-001-O03.dxf`, `externos/`).
- `test/`: Historical ground truth datasets (`test_1/`, `test_2/`, etc.).
- `output_eval/` / `visual_sheets/`: Generated visual sheets and detection artifacts.
- `.agents/`: Agent working directories and metadata.
