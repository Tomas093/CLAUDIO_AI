#!/bin/sh
# 30/09: RF11 = RF4 -> ds26 (ds23 + tiles REALES del GT nuevo de Tomas x2, 2 clases). Pasa adelante de RF9:
# cuando cadena_ds24 termina RF8 ("RF8 listo") se corta esa cadena (iba a entrenar RF9) y corre RF11; RF9 despues.
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd "$(dirname "$0")"
L=log_cadena_ds26.txt; PY=venv_rfdetr/Scripts/python.exe
until grep -q "RF8 listo" log_cadena_ds24.txt 2>/dev/null; do sleep 120; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'cadena_ds24\.sh|train_rfdetr\.py' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >> $L 2>&1
[ -d work/runs/RF9_nano ] && mv work/runs/RF9_nano "_para_borrar/RF9_nano_cortado_$(date +%H%M%S)"
until [ -f work/reales_tomas/LISTO ] || grep -q "tiles/cajas" log_reales_tomas.txt 2>/dev/null; do sleep 120; done
[ -f work/ds26/data.yaml ] || py -3 src/armar_ds26.py >> $L 2>&1
[ -f work/ds26_coco/COMPLETO ] || py -3 src/yolo2coco.py work/ds26 >> $L 2>&1
if [ ! -f work/runs/RF11_nano/best_limpio.pth ]; then
    echo "[ds26] $(date) arranca/retoma RF11_nano (RF4 -> ds26)" >> $L
    RFDETR_NAME=RF11_nano RFDETR_DS=ds26_coco RFDETR_EPOCHS=16 RFDETR_PATIENCE=16 RFDETR_LR_DROP=12 \
        RFDETR_PRE="$(pwd)/best_componente_v24_RF4.pth" $PY -u src/train_rfdetr.py >> log_RF11_nano.txt 2>&1
fi
if [ -f work/runs/RF11_nano/best_limpio.pth ]; then
    cp -f work/runs/RF11_nano/best_limpio.pth best_componente_v34_RF11.pth
    [ -f work/eval/V9_RF11_005/resumen.json ] || EVAL_CLASES_FUERA=2 $PY -u src/evaluate.py best_componente_v34_RF11.pth 0.05 V9_RF11_005 > log_eval_RF11.txt 2>&1
    [ -f work/eval/V9_RF11_marcelo/resumen.json ] || EVAL_CLASES_FUERA=2 EVAL_SET=qet QET_DIR=test_marcelo $PY -u src/evaluate.py best_componente_v34_RF11.pth 0.05 V9_RF11_marcelo >> log_eval_RF11.txt 2>&1
    GT_BARRIDO=GT_V9 py -3 src/barrido.py V9_RF11 >> log_eval_RF11.txt 2>&1
    echo "[ds26] $(date) CADENA_DS26_LISTA" >> $L
else echo "[ds26] $(date) RF11 FALLO" >> $L; fi
sh cadena_ds24.sh     # RF9 (retoma; RF8 ya esta)
