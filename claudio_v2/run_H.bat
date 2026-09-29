@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan H (fine tuning sobre G) inicio %date% %time% ===== >> log_H.txt

REM Dataset chico y enfocado (ds4, ~4500 tiles) con los simbolos reales de data/diferencial.dxf.
set H_EPOCHS=25
set PATIENCE=10
set H_NP=3000
set H_NN=600
set H_NR=250
set PESO_TOMAS=8
set P_SPM=0.55
set P_FUSIBLE=0.45
set P_POLO=0.30
set P_DENSA=0.20
set P_BARRA=0.08
set P_TRAFO=0.10
set P_CELDAS=0.12

%PY% src\train.py --name H_ft >> log_H.txt 2>&1

copy /Y work\runs\H_ft\best_limpio.pt best_componente_v4.pt >> log_H.txt 2>&1

REM Evaluaciones: los 4 planos de test y los 3 del cliente
%PY% src\evaluate.py best_componente_v4.pt 0.20 H_test >> log_H.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v4.pt 0.20 H_marcelo >> log_H.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan H fin %date% %time% ===== >> log_H.txt
