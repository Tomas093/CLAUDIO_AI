@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== H2 + I inicio %date% %time% ===== >> log_H2.txt

REM Segunda vuelta del fine tuning. Dataset ds5: el ojo de buey y el fusible tambien se
REM generan SOLOS, mas negativos que en H, y menos epocas.
set H2_EPOCHS=12
set PATIENCE=6
set H2_NP=6000
set H2_NN=2000
set H2_NR=500
set PESO_TOMAS=5
set P_SPM=0.50
set P_FUSIBLE=0.35
set P_POLO=0.25
set P_DENSA=0.28
set P_BARRA=0.10
set P_TRAFO=0.15
set P_CELDAS=0.18

REM --- H2: parte de H (que ya tiene el testigo resuelto) ---
%PY% src\train.py --name H2_ft >> log_H2.txt 2>&1
copy /Y work\runs\H2_ft\best_limpio.pt best_componente_v5_H2.pt >> log_H2.txt 2>&1
%PY% src\evaluate.py best_componente_v5_H2.pt 0.05 H2_005 >> log_H2.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v5_H2.pt 0.20 H2_marcelo >> log_H2.txt 2>&1
set EVAL_SET=
set QET_DIR=

REM --- I: parte de G (limpio) con el mismo dataset ---
%PY% src\train.py --name I_ft >> log_H2.txt 2>&1
copy /Y work\runs\I_ft\best_limpio.pt best_componente_v5_I.pt >> log_H2.txt 2>&1
%PY% src\evaluate.py best_componente_v5_I.pt 0.05 I_005 >> log_H2.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v5_I.pt 0.20 I_marcelo >> log_H2.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== H2 + I fin %date% %time% ===== >> log_H2.txt
