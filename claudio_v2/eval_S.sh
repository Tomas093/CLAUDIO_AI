#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f work/runs/S_full/best_limpio.pt ]; do sleep 60; done
sleep 30
cp -f work/runs/S_full/best_limpio.pt best_componente_v14_S.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v14_S.pt 0.05 V4_v14_S_005 >  log_eval_S.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v14_S.pt 0.05 V4_v14_S_marcelo >> log_eval_S.txt 2>&1
echo "EVAL_S_LISTO" >> log_eval_S.txt
