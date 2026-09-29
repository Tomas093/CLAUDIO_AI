#!/bin/sh
# 25/09. Objetivo de Tomas: 100% de recall a conf ~0,25. RF_nano (ds18, 40 ep.) llego a 100% hasta 0,13
# y seguia mejorando; no detecta bien el PLC ni amperimetros a 0,25. RF2 = RF-DETR Nano sobre ds19
# (ds18 + marco punteado, la receta de U) y 60 epocas. Idempotente.
cd "$(dirname "$0")"
PY=venv_rfdetr/Scripts/python.exe
[ -f work/ds19/data.yaml ] || { echo "[cadena_RF2] falta ds19" >> log_cadena_RF2.txt; exit 1; }
[ -f work/ds19_coco/train/_annotations.coco.json ] || py -3 src/yolo2coco.py work/ds19 >> log_cadena_RF2.txt 2>&1
if [ ! -f work/runs/RF2_nano/best_limpio.pth ]; then
    echo "[cadena_RF2] $(date) arranca RF2_nano" >> log_cadena_RF2.txt
    RFDETR_NAME=RF2_nano RFDETR_DS=ds19_coco RFDETR_EPOCHS=60 RFDETR_PATIENCE=60 RFDETR_LR_DROP=48 \
        $PY -u src/train_rfdetr.py > log_RF2_nano.txt 2>&1
fi
[ -f work/runs/RF2_nano/best_limpio.pth ] || { echo "[cadena_RF2] RF2_nano fallo" >> log_cadena_RF2.txt; exit 1; }
cp -f work/runs/RF2_nano/best_limpio.pth best_componente_v24_RF2.pth
export GT_V4=1
if [ ! -f work/eval/V4_RF2_nano_005/resumen.json ]; then
    $PY -u src/evaluate.py best_componente_v24_RF2.pth 0.05 V4_RF2_nano_005 > log_eval_RF2_nano.txt 2>&1
    EVAL_SET=qet QET_DIR=test_marcelo $PY -u src/evaluate.py best_componente_v24_RF2.pth 0.05 V4_RF2_nano_marcelo >> log_eval_RF2_nano.txt 2>&1
fi
echo "[cadena_RF2] $(date) RF2_nano listo" >> log_cadena_RF2.txt
