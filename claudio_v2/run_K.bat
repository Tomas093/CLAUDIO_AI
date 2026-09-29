@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan K (largo, desde cero, criterio final) inicio %date% %time% ===== >> log_K.txt

REM Dataset ds6: fusible y ojo de buey SEPARADOS, con los 21 simbolos reales de Tomas.
set K_EPOCHS=200
set PATIENCE=40
set J_NP=12000
set J_NN=3500
set J_NR=900
set PESO_TOMAS=6
set P_SPM=0.45
set P_FUSIBLE=0.35
set P_POLO=0.25
set P_DENSA=0.28
set P_BARRA=0.10
set P_TRAFO=0.15
set P_CELDAS=0.18

%PY% src\train.py --name K_full >> log_K.txt 2>&1
copy /Y work\runs\K_full\best_limpio.pt best_componente_v6_K.pt >> log_K.txt 2>&1

%PY% src\evaluate.py best_componente_v6_K.pt 0.05 K_005 >> log_K.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v6_K.pt 0.20 K_marcelo >> log_K.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan K fin %date% %time% ===== >> log_K.txt
