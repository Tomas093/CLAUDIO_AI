# compare_models.py
import os
import sys
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from evaluate_all_plans import evaluate_dxf

def run_evaluation(model_path, tag=""):
    print("\n" + "=" * 80)
    print(f"EVALUANDO MODELO: {model_path} ({tag})")
    print("=" * 80)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    
    plans = [
        {
            'name': 'TEST 2 (Base INSERTs)',
            'dxf': 'test_2.dxf',
            'gt': 'test/test_2/verdad_terreno/test_2_inserts.csv',
            'out': f'eval_multitest/cmp_{tag}_test2_base',
            'scale': 84.31,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 2 (Completo Aparatos)',
            'dxf': 'test_2.dxf',
            'gt': 'test/test_2/verdad_terreno/test_2_completo.csv',
            'out': f'eval_multitest/cmp_{tag}_test2_comp',
            'scale': 84.31,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 1 (Base INSERTs)',
            'dxf': 'test1.dxf',
            'gt': 'test/test_1/verdad_terreno/test1_inserts.csv',
            'out': f'eval_multitest/cmp_{tag}_test1_base',
            'scale': 75.0,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 1 (Completo Aparatos)',
            'dxf': 'test1.dxf',
            'gt': 'test/test_1/verdad_terreno/test1_completo.csv',
            'out': f'eval_multitest/cmp_{tag}_test1_comp',
            'scale': 75.0,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'FL-UN-02 (Industrial 258)',
            'dxf': 'dxf/FL-UN-02_tablero_1.dxf',
            'gt': 'dxf/fl_un_02_gt.csv',
            'out': f'eval_multitest/cmp_{tag}_fl_un_02',
            'scale': 75.0,
            'is_ind': True,
            'dist_tol': 1.2
        }
    ]
    
    results = {}
    for p in plans:
        res_conf, _, _ = evaluate_dxf(
            dxf_path=p['dxf'],
            gt_csv_path=p['gt'],
            model=model,
            out_dir=p['out'],
            px_per_cad_default=p['scale'],
            is_industrial=p['is_ind'],
            conf_sweep=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            dist_tol=p['dist_tol'],
            device=device
        )
        results[p['name']] = res_conf
    return results

def print_comparison(orig_res, boost_res):
    print("\n" + "#" * 105)
    print("COMPARATIVA COMPLETA ANTES (ORIGINAL) vs DESPUES (BOOSTED)")
    print("#" * 105)
    
    for th in [0.10, 0.15, 0.20]:
        print(f"\n--- UMBRAL DE CONFIANZA: Conf >= {th:.2f} ---")
        header = f"{'Plano / Benchmark':<30} | {'GT':>4} | {'TP (Orig)':>9} {'TP (Boost)':>10} | {'FP (Orig)':>9} {'FP (Boost)':>10} | {'Rec (Orig)':>10} {'Rec (Boost)':>11} | {'Prec (Orig)':>11} {'Prec (Boost)':>12} | {'F1 (Orig)':>9} {'F1 (Boost)':>10}"
        print(header)
        print("-" * len(header))
        
        for name in orig_res.keys():
            o = orig_res[name][th]
            b = boost_res[name][th]
            gt = o['gt']
            print(f"{name:<30} | {gt:4d} | {o['tp']:9d} {b['tp']:10d} | {o['fp']:9d} {b['fp']:10d} | {o['recall']:9.1f}% {b['recall']:10.1f}% | {o['precision']:10.1f}% {b['precision']:11.1f}% | {o['f1']/100:9.3f} {b['f1']/100:10.3f}")

if __name__ == '__main__':
    prev_p = 'train-maker/models/best_componente_nano_prev35ep.pt'
    curr_p = 'train-maker/models/best_componente_nano.pt'
    
    print("Iniciando benchmark comparativo...")
    orig = run_evaluation(prev_p, tag="orig")
    boost = run_evaluation(curr_p, tag="boost")
    print_comparison(orig, boost)
