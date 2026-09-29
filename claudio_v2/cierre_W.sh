#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f work/runs/W_ft/best_limpio.pt ]; do sleep 30; done
sleep 30
cp -f work/runs/W_ft/best_limpio.pt best_componente_v18_W.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v18_W.pt 0.05 V4_v18_W_005 >  log_eval_W.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v18_W.pt 0.05 V4_v18_W_marcelo >> log_eval_W.txt 2>&1
echo "EVAL_W_LISTO" >> log_eval_W.txt
py -3 -u src/rescore.py V3 V4 > log_cierre_W.txt 2>&1
echo "--- fusion ---" >> log_cierre_W.txt
py -3 -u src/metrica_fusion.py V4_v13_R_marcelo V4_v18_W_marcelo >> log_cierre_W.txt 2>&1
echo "CIERRE_W_LISTO" >> log_cierre_W.txt
