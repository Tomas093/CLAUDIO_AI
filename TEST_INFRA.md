# E2E Test Infra: CLAUDIO_AI Argentine Single-Line Electrical Diagram Intelligence System

## Test Philosophy
- **Requirement-Driven & Opaque-Box**: Tests validate functional compliance directly against `ORIGINAL_REQUEST.md` (AEA 90364 / IRAM standards, DWG/DXF ingestion, YOLO Nano inference, >=98% Recall, zero regressions) without depending on internal implementation details.
- **Methodology**: 4-tier systematic approach (Category-Partition, Boundary Value Analysis, Pairwise Combinatorial, and Real-World Workload Scenarios).

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) |
|---|---------|--------|:----------------:|:-----------------:|:-----------------:|
| F1 | Web Autonomous Acquisition (>= 3 plans) | Req R1 | 5 | 5 | ✓ |
| F2 | Local Plan Ingestion (`Tablerotsbe`, `plano1-5`, `OCJ`) | Req R1 | 5 | 5 | ✓ |
| F3 | Headless DWG->DXF Conversion (`accoreconsole`) | Req R2 | 5 | 5 | ✓ |
| F4 | Rendering with `COLOR_SWAP_BW` & `HatchPolicy.NORMAL` | Req R2 | 5 | 5 | ✓ |
| F5 | Scale Normalization (75 - 100 px/CAD) | Req R2 | 5 | 5 | ✓ |
| F6 | SAHI Tiling (640x640, 80% overlap, 320 px pad) | Req R3 | 5 | 5 | ✓ |
| F7 | Centroidal NMS Calibration ($d_{\min} \approx 0.20$ CAD) | Req R3 | 5 | 5 | ✓ |
| F8 | Full Apparatus Detection Gamut (PIA, ID, DPS, etc.) | Req R3 | 5 | 5 | ✓ |
| F9 | Metric Audit Engine (Recall, Precision, F1, TP/FP/FN) | Req R4 | 5 | 5 | ✓ |
| F10 | High-Res Visual Sheets & Export Format (PNG, CSV, JSON) | Req R4 | 5 | 5 | ✓ |
| F11 | Zero Regression on Historical Benchmarks (TEST 1/2, FL, TSSS, Vyre) | Req R4 | 5 | 5 | ✓ |
| F12 | System Robustness & Non-CAD Rejection | R1-R4 | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**: Python pytest runner / standalone runner `run_e2e_tests.py` executable via `py -3.9 run_e2e_tests.py`.
- **Exit Code**: Exit code 0 indicates 100% pass across all enabled tiers. Non-zero indicates failures with detailed diagnosis.
- **Test Artifact Locations**:
  - Test suites: `tests/e2e/` (e.g. `test_tier1_features.py`, `test_tier2_boundaries.py`, `test_tier3_pairwise.py`, `test_tier4_workloads.py`).
  - Output artifacts: `output_eval/`, `visual_sheets/`.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full pipeline on newly acquired residential switchboard (`05_diagrama_unifilar.dxf`) | F1, F4, F5, F6, F7, F8, F10 | High |
| 2 | Full pipeline on motor control switchboard (`01_diagrama_unifilar.dxf`) | F1, F4, F5, F6, F7, F8, F10 | High |
| 3 | Full pipeline on industrial CCM switchboard (`01_diagrama_unifilar_ccm.dxf`) | F1, F4, F5, F6, F7, F8, F10 | High |
| 4 | Batch evaluation on pre-validated switchboards (`Tablerotsbe` & `plano3`) at conf 0.50 | F2, F4, F5, F6, F7, F8, F9, F10 | High |
| 5 | Complete audit & regression run on historical benchmark suite (`TEST 1`, `TEST 2`, `FL-UN-02`, `TSSS_2`, `Vyre`) | F2, F6, F7, F8, F9, F10, F11 | Very High |
| 6 | Master drawing inspection and switchboard viewport extraction on `OCJ-DE-IEL-UNI-000-001-O03` | F2, F4, F5, F6, F7, F8, F10 | Very High |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: >= 60 test cases (5 per feature).
- **Tier 2 (Boundary & Corner Cases)**: >= 60 test cases (5 per feature).
- **Tier 3 (Pairwise Interactions)**: >= 12 test cases covering critical feature pairs (e.g. DWG conversion + rendering, tiling + NMS, scale + inference).
- **Tier 4 (Real-World Scenarios)**: >= 6 comprehensive end-to-end workload cases.
- **Total Minimum Test Count**: >= 138 test cases.
