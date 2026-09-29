#!/bin/sh
# 28/09, pedido de Tomas: dataset ds21 (fallas de RF4: letra en recuadro, rotulos verticales como NEGATIVO,
# negativos nuevos, fondo blanco, VIS_MIN 0,85) y, "cuando pongas a entrenar el rf, pone un yolo nano adelante":
#   0) espera a que termine cadena_orden.sh (RF5 usa la GPU)
#   1) N21 = yolo11n sobre ds21 (train.py arma ds21 si falta), 200 epocas
#   2) RF6 = RF-DETR Nano (rfdetr 1.3, como RF4) sobre ds21_coco, 60 epocas, LR a 1/10 en la 48
# Todo evaluado con GT_V6=1. Idempotente: retoma desde last.pt / checkpoint.pth. Lanzar via WMI
# (memoria "entrenamientos-desacoplados"); arranque_cadena.ps1 la relanza tras un corte de luz, salvo que
# este log diga CADENA_DS21_LISTA (solo se escribe si N21 y RF6 terminaron bien).
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd "$(dirname "$0")"
export GT_V6=1   # 28/09: sin los rotulos de test1 (no son componentes)
L=log_cadena_ds21.txt
mkdir -p _para_borrar

until grep -q CADENA_ORDEN_LISTA log_cadena_orden.txt 2>/dev/null; do sleep 300; done

# ds21 a medio armar (corte de luz o error durante build_all): sin data.yaml se aparta y se rearma
if [ -d work/ds21 ] && [ ! -f work/ds21/data.yaml ] && [ ! -f work/runs/N21/weights/last.pt ]; then
    mv work/ds21 "_para_borrar/ds21_incompleto_$(date +%Y%m%d_%H%M%S)"
fi

# evaluacion de un modelo en los 7 planos; cada tanda se rehace sola si falta su resumen.json
evaluar() {  # $1 python  $2 pesos  $3 tag  $4 log
    [ -f work/eval/${3}_005/resumen.json ] || $1 -u src/evaluate.py $2 0.05 ${3}_005 > $4 2>&1
    [ -f work/eval/${3}_marcelo/resumen.json ] || EVAL_SET=qet QET_DIR=test_marcelo $1 -u src/evaluate.py $2 0.05 ${3}_marcelo >> $4 2>&1
}

# 1) yolo nano sobre ds21
if [ ! -f work/runs/N21/best_limpio.pt ]; then
    if [ -f work/runs/N21/weights/last.pt ]; then
        echo "[ds21] $(date) reanuda N21" >> $L; py -3 -u src/reanudar.py N21 >> log_N21.txt 2>&1
    else
        echo "[ds21] $(date) arma ds21 y arranca N21 (yolo11n)" >> $L; py -3 -u src/train.py --name N21 > log_N21.txt 2>&1
    fi
fi
if [ -f work/runs/N21/best_limpio.pt ]; then
    cp -f work/runs/N21/best_limpio.pt best_componente_v27_N21.pt
    evaluar "py -3" best_componente_v27_N21.pt V6_N21 log_eval_N21.txt
    echo "[ds21] $(date) N21 listo" >> $L
else
    echo "[ds21] $(date) N21 fallo" >> $L
fi

# 2) RF6 (rfdetr 1.3) sobre ds21. La conversion a COCO se rehace si no termino (marca COMPLETO).
if [ -f work/ds21/data.yaml ] && [ ! -f work/ds21_coco/COMPLETO ] && [ ! -f work/runs/RF6_nano/checkpoint.pth ]; then
    [ -d work/ds21_coco ] && mv work/ds21_coco "_para_borrar/ds21_coco_incompleto_$(date +%Y%m%d_%H%M%S)"
    py -3 src/yolo2coco.py work/ds21 >> $L 2>&1
fi
PY=venv_rfdetr/Scripts/python.exe
if [ -f work/ds21_coco/COMPLETO ] && [ ! -f work/runs/RF6_nano/best_limpio.pth ]; then
    echo "[ds21] $(date) arranca/retoma RF6_nano (rfdetr 1.3, ds21)" >> $L
    RFDETR_NAME=RF6_nano RFDETR_DS=ds21_coco RFDETR_EPOCHS=60 RFDETR_PATIENCE=60 RFDETR_LR_DROP=48 \
        $PY -u src/train_rfdetr.py >> log_RF6_nano.txt 2>&1
fi
if [ -f work/runs/RF6_nano/best_limpio.pth ]; then
    cp -f work/runs/RF6_nano/best_limpio.pth best_componente_v28_RF6.pth
    evaluar $PY best_componente_v28_RF6.pth V6_RF6_nano log_eval_RF6_nano.txt
    echo "[ds21] $(date) RF6_nano listo" >> $L
else
    echo "[ds21] $(date) RF6_nano fallo" >> $L
fi

if [ -f work/eval/V6_N21_marcelo/resumen.json ] && [ -f work/eval/V6_RF6_nano_marcelo/resumen.json ]; then
    echo "CADENA_DS21_LISTA" >> $L
else
    echo "CADENA_DS21_FALLO $(date) (arranque_cadena.ps1 la vuelve a intentar al iniciar sesion)" >> $L
fi
