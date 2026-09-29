#!/bin/sh
# Espera a que termine el plan S, se asegura de que este evaluado y corre las tres metricas.
# Es idempotente: si la cadena eval_S.sh sigue viva, la deja trabajar; si murio, evalua aca.
cd "$(dirname "$0")"
while [ ! -f work/runs/S_full/best_limpio.pt ]; do sleep 60; done
sleep 60
cp -f work/runs/S_full/best_limpio.pt best_componente_v14_S.pt
export GT_V4=1
# esperar hasta 25 min a que la cadena original evalue; si no, evaluar aca
i=0
while [ $i -lt 25 ]; do
  if [ -f log_eval_S.txt ] && grep -aq "EVAL_S_LISTO" log_eval_S.txt; then break; fi
  i=$((i+1)); sleep 60
done
if ! grep -aq "EVAL_S_LISTO" log_eval_S.txt 2>/dev/null; then
  py -3 -u src/evaluate.py best_componente_v14_S.pt 0.05 V4_v14_S_005 >  log_eval_S.txt 2>&1
  EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v14_S.pt 0.05 V4_v14_S_marcelo >> log_eval_S.txt 2>&1
  echo "EVAL_S_LISTO" >> log_eval_S.txt
fi
py -3 -u src/rescore.py V3 V4            > log_cierre.txt 2>&1
echo "--- fusion ---"                   >> log_cierre.txt
py -3 -u src/metrica_fusion.py V3_v9_N_marcelo V3_v12_Q_marcelo V4_v13_R_marcelo V4_v14_S_marcelo >> log_cierre.txt 2>&1
echo "--- confianza ---"                >> log_cierre.txt
py -3 -u src/metrica_confianza.py V4_v14_S V4_v13_R V3_v9_N >> log_cierre.txt 2>&1
echo "CIERRE_LISTO"                     >> log_cierre.txt
