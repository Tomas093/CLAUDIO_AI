#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f work/runs/T_full/best_limpio.pt ]; do sleep 60; done
sleep 60
cp -f work/runs/T_full/best_limpio.pt best_componente_v15_T.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v15_T.pt 0.05 V4_v15_T_005 >  log_eval_T.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v15_T.pt 0.05 V4_v15_T_marcelo >> log_eval_T.txt 2>&1
echo "EVAL_T_LISTO" >> log_eval_T.txt
py -3 -u src/rescore.py V3 V4 > log_cierre_T.txt 2>&1
echo "--- fusion ---" >> log_cierre_T.txt
py -3 -u src/metrica_fusion.py V4_v13_R_marcelo V4_v15_T_marcelo >> log_cierre_T.txt 2>&1
echo "--- confianza ---" >> log_cierre_T.txt
py -3 -u src/metrica_confianza.py V4_v15_T V4_v13_R >> log_cierre_T.txt 2>&1
echo "CIERRE_T_LISTO" >> log_cierre_T.txt
