#!/bin/sh
cd "$(dirname "$0")"
while [ ! -f work/runs/U_full/best_limpio.pt ]; do sleep 60; done
sleep 60
cp -f work/runs/U_full/best_limpio.pt best_componente_v16_U.pt
export GT_V4=1
py -3 -u src/evaluate.py best_componente_v16_U.pt 0.05 V4_v16_U_005 >  log_eval_U.txt 2>&1
EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v16_U.pt 0.05 V4_v16_U_marcelo >> log_eval_U.txt 2>&1
echo "EVAL_U_LISTO" >> log_eval_U.txt
py -3 -u src/rescore.py V3 V4 > log_cierre_U.txt 2>&1
echo "--- fusion ---" >> log_cierre_U.txt
py -3 -u src/metrica_fusion.py V4_v13_R_marcelo V4_v15_T_marcelo V4_v16_U_marcelo >> log_cierre_U.txt 2>&1
echo "--- confianza ---" >> log_cierre_U.txt
py -3 -u src/metrica_confianza.py V4_v16_U V4_v13_R >> log_cierre_U.txt 2>&1
echo "CIERRE_U_LISTO" >> log_cierre_U.txt
