#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f work/runs/V_ft/best_limpio.pt ]; do sleep 30; done
sleep 30
cp -f work/runs/V_ft/best_limpio.pt best_componente_v17_V.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v17_V.pt 0.05 V4_v17_V_005 >  log_eval_V.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v17_V.pt 0.05 V4_v17_V_marcelo >> log_eval_V.txt 2>&1
echo "EVAL_V_LISTO" >> log_eval_V.txt
py -3 -u src/rescore.py V3 V4 > log_cierre_V.txt 2>&1
echo "--- fusion ---" >> log_cierre_V.txt
py -3 -u src/metrica_fusion.py V4_v13_R_marcelo V4_v17_V_marcelo >> log_cierre_V.txt 2>&1
echo "CIERRE_V_LISTO" >> log_cierre_V.txt
