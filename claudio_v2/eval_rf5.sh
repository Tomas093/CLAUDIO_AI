#!/bin/sh
# 28/09: re-evaluacion de RF5 (la de cadena_orden fallo: rfdetr 1.11 buscaba el .pth relativo en ~/.roboflow/models)
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd "$(dirname "$0")"
export GT_V6=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 EVAL_LOTE=8
PY11=/d/CLAUDIO_RFDETR/venv311/Scripts/python.exe
[ -f work/eval/V6_RF5_nano_005/resumen.json ] || $PY11 -u src/evaluate.py best_componente_v25_RF5.pth 0.05 V6_RF5_nano_005 > log_eval_RF5_nano.txt 2>&1
[ -f work/eval/V6_RF5_nano_marcelo/resumen.json ] || EVAL_SET=qet QET_DIR=test_marcelo $PY11 -u src/evaluate.py best_componente_v25_RF5.pth 0.05 V6_RF5_nano_marcelo >> log_eval_RF5_nano.txt 2>&1
echo "EVAL_RF5_LISTA $(date)" >> log_eval_RF5_nano.txt
