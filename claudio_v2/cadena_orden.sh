#!/bin/sh
# 26/09. Orden pedido por Tomas: primero yolo11m (compararlo contra RF-DETR y decidir si seguir con RF o
# con un medium), despues RF4 (retoma desde su checkpoint: murio en la epoca 2 al cerrarse la sesion) y
# al final RF5 (rfdetr 1.11). Todo sobre ds20, evaluado con GT_V5=1. Reemplaza a cadena_RF45.sh y
# cadena_M11.sh. Idempotente: si se corta, relanzar y saltea/retoma lo que ya esta.
# Lanzar DESACOPLADO de la sesion (si no, muere al cerrarla):
#   powershell Start-Process 'C:\Program Files\Git\usr\bin\sh.exe' -ArgumentList 'cadena_orden.sh' -WindowStyle Hidden
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH   # lanzado con Start-Process no trae /usr/bin (date, cp)
cd "$(dirname "$0")"
export GT_V5=1
L=log_cadena_orden.txt

# 1) yolo11m, 100 epocas (~13 min/epoca: 200 serian ~43 h; 100 = ~22 h, el mismo tiempo de GPU que RF4)
if [ ! -f work/runs/M11_20/best_limpio.pt ]; then
    if [ -f work/runs/M11_20/weights/last.pt ]; then
        echo "[orden] $(date) reanuda M11_20" >> $L; py -3 -u src/reanudar.py M11_20 >> log_M11_20.txt 2>&1
    else
        echo "[orden] $(date) arranca M11_20 (yolo11m, ds20)" >> $L; M_EPOCHS=100 py -3 -u src/train.py --name M11_20 > log_M11_20.txt 2>&1
    fi
fi
if [ -f work/runs/M11_20/best_limpio.pt ]; then
    cp -f work/runs/M11_20/best_limpio.pt best_componente_v26_M11.pt
    if [ ! -f work/eval/V5_M11_20_005/resumen.json ]; then
        py -3 -u src/evaluate.py best_componente_v26_M11.pt 0.05 V5_M11_20_005 > log_eval_M11_20.txt 2>&1
        EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py best_componente_v26_M11.pt 0.05 V5_M11_20_marcelo >> log_eval_M11_20.txt 2>&1
    fi
    echo "[orden] $(date) M11_20 listo" >> $L
else
    echo "[orden] $(date) M11_20 fallo" >> $L
fi

# 2) RF4 (rfdetr 1.3; train_rfdetr.py retoma solo si hay checkpoint.pth)
PY=venv_rfdetr/Scripts/python.exe
if [ ! -f work/runs/RF4_nano/best_limpio.pth ]; then
    echo "[orden] $(date) arranca/retoma RF4_nano (rfdetr 1.3, ds20)" >> $L
    RFDETR_NAME=RF4_nano RFDETR_DS=ds20_coco RFDETR_EPOCHS=60 RFDETR_PATIENCE=60 RFDETR_LR_DROP=48 \
        $PY -u src/train_rfdetr.py >> log_RF4_nano.txt 2>&1
fi
if [ -f work/runs/RF4_nano/best_limpio.pth ]; then
    cp -f work/runs/RF4_nano/best_limpio.pth best_componente_v24_RF4.pth
    if [ ! -f work/eval/V5_RF4_nano_005/resumen.json ]; then
        $PY -u src/evaluate.py best_componente_v24_RF4.pth 0.05 V5_RF4_nano_005 > log_eval_RF4_nano.txt 2>&1
        EVAL_SET=qet QET_DIR=test_marcelo $PY -u src/evaluate.py best_componente_v24_RF4.pth 0.05 V5_RF4_nano_marcelo >> log_eval_RF4_nano.txt 2>&1
    fi
    echo "[orden] $(date) RF4_nano listo" >> $L
else
    echo "[orden] $(date) RF4_nano fallo" >> $L
fi

# 3) RF5 (rfdetr 1.11, Python 3.11 en D:)
PY11=/d/CLAUDIO_RFDETR/venv311/Scripts/python.exe
OUT=/d/CLAUDIO_RFDETR/runs/RF5_nano
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [ ! -f $OUT/best_limpio.pth ]; then
    echo "[orden] $(date) arranca RF5_nano (rfdetr 1.11, ds20)" >> $L
    RFDETR_NAME=RF5_nano RFDETR_DS=ds20_coco RFDETR_EPOCHS=60 RFDETR_LR_DROP=48 \
        $PY11 -u src/train_rfdetr11.py > /d/CLAUDIO_RFDETR/log_RF5_nano.txt 2>&1
fi
if [ -f $OUT/best_limpio.pth ]; then
    cp -f $OUT/best_limpio.pth best_componente_v25_RF5.pth
    if [ ! -f work/eval/V5_RF5_nano_005/resumen.json ]; then
        $PY11 -u src/evaluate.py best_componente_v25_RF5.pth 0.05 V5_RF5_nano_005 > log_eval_RF5_nano.txt 2>&1
        EVAL_SET=qet QET_DIR=test_marcelo $PY11 -u src/evaluate.py best_componente_v25_RF5.pth 0.05 V5_RF5_nano_marcelo >> log_eval_RF5_nano.txt 2>&1
    fi
    echo "[orden] $(date) RF5_nano listo" >> $L
else
    echo "[orden] $(date) RF5_nano fallo" >> $L
fi
echo "CADENA_ORDEN_LISTA" >> $L
