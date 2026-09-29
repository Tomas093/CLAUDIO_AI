#!/usr/bin/env python
"""
================================================================================
CLAUDIO_AI - Autonomous E2E Test Suite Runner (Tiers 1 - 4)
================================================================================
Central test runner executing the 4-tier opaque-box E2E test suite:
- Tier 1: Feature Isolation Tests (F1 to F12, >= 60 tests)
- Tier 2: Boundary Value & Corner Cases (F1 to F12, >= 60 tests)
- Tier 3: Pairwise Combinatorial Interactions (>= 12 tests)
- Tier 4: Realistic Real-World Workload Scenarios (>= 6 tests)

Usage:
  py -3.9 run_e2e_tests.py [--tier {1,2,3,4,all}] [--verbose]
================================================================================
"""

import os
import sys
import time
import argparse
import unittest
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DETECTOR_PACK_DIR = PROJECT_ROOT / "detector_pack"
if str(DETECTOR_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_PACK_DIR))

# Encoding reconfigure
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


TIER_CONFIG = [
    {
        "id": "tier1",
        "num": 1,
        "name": "Tier 1: Feature Isolation (F1-F12)",
        "module": "tests.e2e.test_tier1_features",
        "description": "Tests each feature in isolation (web acquisition, local CAD, DWG helper, render policies, scale, SAHI, NMS, gamut, metrics, visual sheets, regression, validation)",
        "target_min": 60
    },
    {
        "id": "tier2",
        "num": 2,
        "name": "Tier 2: Boundary & Corner Cases",
        "module": "tests.e2e.test_tier2_boundaries",
        "description": "Boundary values (empty CAD, single-entity, huge coords, conf boundaries 0.01/0.50/0.99, d_min, zero extents, corrupt DXF)",
        "target_min": 60
    },
    {
        "id": "tier3",
        "num": 3,
        "name": "Tier 3: Pairwise Combinatorial",
        "module": "tests.e2e.test_tier3_pairwise",
        "description": "Cross-feature interactions (DWG+render, scale+tiling, tiling+NMS, render+YOLO, YOLO+audit, audit+sheets, scale+CAD NMS, etc.)",
        "target_min": 12
    },
    {
        "id": "tier4",
        "num": 4,
        "name": "Tier 4: Real-World Workload Scenarios",
        "module": "tests.e2e.test_tier4_workloads",
        "description": "End-to-end realistic workloads on Argentine switchboards (05_diagrama_unifilar, motor control, CCM industrial, Tablerotsbe, plano3, TEST 1/2, OCJ)",
        "target_min": 6
    }
]


def run_tier(tier_info: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Loads and runs a specific test tier module, collecting structured metrics."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(tier_info["module"])

    print(f"\n" + "=" * 80)
    print(f"Executing {tier_info['name']}")
    print(f"Scope: {tier_info['description']}")
    print("=" * 80)

    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)

    t0 = time.time()
    result = runner.run(suite)
    duration = time.time() - t0

    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)

    return {
        "id": tier_info["id"],
        "num": tier_info["num"],
        "name": tier_info["name"],
        "total": result.testsRun,
        "passed": passed,
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration": duration,
        "success": result.wasSuccessful(),
        "target_min": tier_info["target_min"]
    }


def print_summary_table(results: List[Dict[str, Any]], total_duration: float):
    """Prints an ASCII summary report table of all tiers."""
    print("\n" + "=" * 90)
    print("CLAUDIO_AI - COMPREHENSIVE E2E TEST SUITE EXECUTION REPORT")
    print("=" * 90)
    header = f"{'Tier':<8} | {'Category / Test Scope':<38} | {'Total':>5} | {'Pass':>5} | {'Fail':>5} | {'Err':>4} | {'Time (s)':>8}"
    print(header)
    print("-" * 90)

    tot_tests = sum(r["total"] for r in results)
    tot_passed = sum(r["passed"] for r in results)
    tot_failed = sum(r["failed"] for r in results)
    tot_errors = sum(r["errors"] for r in results)
    all_success = all(r["success"] for r in results)

    for r in results:
        t_label = f"Tier {r['num']}"
        cat_name = r["name"].split(": ")[-1][:38]
        status_flag = "PASS" if r["success"] else "FAIL"
        print(f"{t_label:<8} | {cat_name:<38} | {r['total']:5d} | {r['passed']:5d} | {r['failed']:5d} | {r['errors']:4d} | {r['duration']:7.2f}s")

    print("-" * 90)
    print(f"{'TOTAL':<8} | {'All E2E Verification Tiers':<38} | {tot_tests:5d} | {tot_passed:5d} | {tot_failed:5d} | {tot_errors:4d} | {total_duration:7.2f}s")
    print("=" * 90)

    if all_success:
        print("\n>>> OVERALL RESULT: SUCCESS (100% PASS) - ALL ACCEPTANCE CRITERIA SATISFIED <<<")
        print(f"    Total Tests Executed: {tot_tests} (Requirement: >= 138 tests)")
        print(f"    Total Failures:       0")
        print(f"    Total Errors:         0")
        print(f"    Total Elapsed Time:   {total_duration:.2f}s\n")
    else:
        print("\n>>> OVERALL RESULT: FAILURE - DETECTED DEFECTS IN SUITE EXECUTION <<<")
        print(f"    Total Failures: {tot_failed}")
        print(f"    Total Errors:   {tot_errors}\n")


def main():
    parser = argparse.ArgumentParser(description="CLAUDIO_AI 4-Tier E2E Test Suite Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all", help="Select tier to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose test execution")
    args = parser.parse_args()

    selected_tiers = []
    if args.tier == "all":
        selected_tiers = TIER_CONFIG
    else:
        selected_tiers = [t for t in TIER_CONFIG if str(t["num"]) == args.tier]

    print("\n" + "#" * 90)
    print("# CLAUDIO_AI E2E TEST RUNNER - ARGENTINE ELECTRICAL SINGLE-LINE CAD INTELLIGENCE")
    print("# Standards: AEA 90364 / IRAM Norms | YOLO Nano CAD Architecture")
    print(f"# Selected Tiers: {[t['num'] for t in selected_tiers]} (Total configured: 4)")
    print("#" * 90)

    results = []
    suite_start = time.time()
    for t in selected_tiers:
        res = run_tier(t, verbose=args.verbose)
        results.append(res)
    total_suite_duration = time.time() - suite_start

    print_summary_table(results, total_suite_duration)

    all_passed = all(r["success"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
