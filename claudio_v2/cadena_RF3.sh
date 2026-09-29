#!/bin/sh
# 25/09. RF3 = RF2 con rfdetr 1.11 (Python 3.11 en D:\CLAUDIO_RFDETR). Pedido de Tomas: probar la version
# nueva "de ultima", despues de RF2, entrenando en el disco D:. Mismo dataset e hiperparametros que RF2:
# la unica diferencia es la version de rfdetr. Idempotente.
cd "$(dirname "$0")"
PY=/d/CLAUDIO_RFDETR/venv311/Scripts/python.exe
OUT=/d/CLAUDIO_RFDETR/runs/RF3_nano
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
until grep -qE "RF2_nano (listo|fallo)" log_cadena_RF2.txt 2>/dev/null; do sleep 600; done
if [ ! -f $OUT/best_limpio.pth ]; then
    echo "[cadena_RF3] $(date) arranca RF3_nano (rfdetr 1.11)" >> log_cadena_RF3.txt
    $PY -u src/train_rfdetr11.py > /d/CLAUDIO_RFDETR/log_RF3_nano.txt 2>&1
fi
[ -f $OUT/best_limpio.pth ] || { echo "[cadena_RF3] RF3_nano fallo" >> log_cadena_RF3.txt; exit 1; }
cp -f $OUT/best_limpio.pth best_componente_v25_RF3.pth
export GT_V4=1
if [ ! -f work/eval/V4_RF3_nano_005/resumen.json ]; then
    $PY -u src/evaluate.py best_componente_v25_RF3.pth 0.05 V4_RF3_nano_005 > log_eval_RF3_nano.txt 2>&1
    EVAL_SET=qet QET_DIR=test_marcelo $PY -u src/evaluate.py best_componente_v25_RF3.pth 0.05 V4_RF3_nano_marcelo >> log_eval_RF3_nano.txt 2>&1
fi
echo "[cadena_RF3] $(date) RF3_nano listo" >> log_cadena_RF3.txt
