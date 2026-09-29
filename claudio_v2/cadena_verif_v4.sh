#!/bin/sh
# 30/09: verificador v4 = datos con negativos de objeto (PAT, PAT en cuadrado, flecha, rotulo) + v2 + v3
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd "$(dirname "$0")"
PY=venv_rfdetr/Scripts/python.exe
[ -f work/verif/rf4_negobj_v4.npz ] || VERIF_NOMBRE=rf4_negobj_v4 VERIF_NEGOBJ=5 VERIF_SEMILLA=6000000 VERIF_SPLITS= $PY -u src/verif_datos.py best_componente_v24_RF4.pth 3000 > log_verif_datos_v4.txt 2>&1
[ -f work/verif/verif_v4.pt ] || $PY -u src/verif_train.py work/verif/rf4_ds21_v2.npz,work/verif/rf4_reales_v3.npz,work/verif/rf4_negobj_v4.npz work/verif/verif_v4.pt > log_verif_train_v4.txt 2>&1
grep -q aplicado log_verif_aplicar_v4.txt 2>/dev/null || { VERIF_PESOS=work/verif/verif_v4.pt $PY -u src/verif_aplicar.py V6_RF4 V6_RF4v4 > log_verif_aplicar_v4.txt 2>&1 && echo aplicado >> log_verif_aplicar_v4.txt; }
echo VERIF_V4_LISTO >> log_verif_aplicar_v4.txt
