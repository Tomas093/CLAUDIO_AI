#!/bin/sh
# Espera a que el plan R termine y lo evalua con el GT v4. Pensado para dejarlo corriendo solo.
cd "$(dirname "$0")"
while [ ! -f work/runs/R_full/best_limpio.pt ]; do sleep 60; done
sleep 30
cp -f work/runs/R_full/best_limpio.pt best_componente_v13_R.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v13_R.pt 0.05 V4_v13_R_005    >  log_eval_R.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v13_R.pt 0.05 V4_v13_R_marcelo >> log_eval_R.txt 2>&1
echo "EVAL_R_LISTO" >> log_eval_R.txt
