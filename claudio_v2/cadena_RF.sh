#!/bin/sh
# 24/09. RF-DETR Nano sobre ds18 (COCO), DESPUES de que termine cadena_XYZ.sh (la GPU es una sola).
# Idempotente. Evalua con la receta de siempre: GT_V4=1, 2 escalas, conf 0,05, los 7 planos.
cd "$(dirname "$0")"
PY=venv_rfdetr/Scripts/python.exe
until grep -q CADENA_XYZ_LISTA log_cadena_XYZ.txt 2>/dev/null; do sleep 300; done
if [ ! -f work/runs/RF_nano/best_limpio.pth ]; then
    echo "[cadena_RF] $(date) arranca RF_nano" >> log_cadena_RF.txt
    $PY -u src/train_rfdetr.py > log_RF_nano.txt 2>&1
fi
[ -f work/runs/RF_nano/best_limpio.pth ] || { echo "[cadena_RF] RF_nano fallo" >> log_cadena_RF.txt; exit 1; }
cp -f work/runs/RF_nano/best_limpio.pth best_componente_v23_RF.pth
export GT_V4=1
if [ ! -f work/eval/V4_RF_nano_005/resumen.json ]; then
    $PY -u src/evaluate.py best_componente_v23_RF.pth 0.05 V4_RF_nano_005 > log_eval_RF_nano.txt 2>&1
    EVAL_SET=qet QET_DIR=test_marcelo $PY -u src/evaluate.py best_componente_v23_RF.pth 0.05 V4_RF_nano_marcelo >> log_eval_RF_nano.txt 2>&1
fi
echo "[cadena_RF] $(date) RF_nano listo" >> log_cadena_RF.txt
