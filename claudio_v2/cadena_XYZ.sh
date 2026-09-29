#!/bin/sh
# 22/09. R2 (yolo11n) -> X (yolo26n-p2) -> Z (yolo26n) -> Y (yolo11s), todos sobre ds18
# (dataset de R corregido). ds18 tiene que estar armado y auditado antes de lanzar.
# Idempotente: si se corta, relanzar y saltea lo que ya tiene best_limpio / eval.
cd "$(dirname "$0")"
export GT_V4=1
corrida() {  # $1 plan  $2 archivo de salida
    if [ ! -f work/runs/$1/best_limpio.pt ]; then
        if [ -f work/runs/$1/weights/last.pt ]; then
            # corrida cortada: se reanuda con los argumentos guardados en last.pt
            echo "[cadena] $(date) reanuda $1" >> log_cadena_XYZ.txt
            py -3 -u src/reanudar.py $1 >> log_$1.txt 2>&1
        else
            echo "[cadena] $(date) arranca $1" >> log_cadena_XYZ.txt
            py -3 -u src/train.py --name $1 > log_$1.txt 2>&1
        fi
    fi
    [ -f work/runs/$1/best_limpio.pt ] || { echo "[cadena] $1 fallo" >> log_cadena_XYZ.txt; return 1; }
    cp -f work/runs/$1/best_limpio.pt $2
    if [ ! -f work/eval/V4_${1}_005/resumen.json ]; then
        py -3 -u src/evaluate.py $2 0.05 V4_${1}_005 > log_eval_$1.txt 2>&1
        EVAL_SET=qet QET_DIR=test_marcelo py -3 -u src/evaluate.py $2 0.05 V4_${1}_marcelo >> log_eval_$1.txt 2>&1
    fi
    echo "[cadena] $(date) $1 listo" >> log_cadena_XYZ.txt
}
[ -f work/ds18/data.yaml ] || { echo "[cadena] falta work/ds18" >> log_cadena_XYZ.txt; exit 1; }
corrida R2_full best_componente_v19_R2.pt
corrida X_26p2 best_componente_v20_X.pt
corrida Z_26n best_componente_v21_Z.pt
if [ -f yolo11s.pt ]; then corrida Y_11s best_componente_v22_Y.pt
else echo "[cadena] Y_11s salteado: falta yolo11s.pt" >> log_cadena_XYZ.txt; fi
py -3 -u src/rescore.py V3 V4 > log_cierre_XYZ.txt 2>&1
echo "CADENA_XYZ_LISTA" >> log_cadena_XYZ.txt
