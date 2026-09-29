@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan P (rotulos multilinea) inicio %date% %time% ===== >> log_P.txt

REM ds11 = ds10 + caja_nombre con 20 rotulos y peso doble + negativo de nota multilinea.
set P_EPOCHS=200
set PATIENCE=40
set L_NP=12000
set L_NN=3500
set L_NR=900
set PESO_TOMAS=6
set USAR_PSEUDO=1
set PSEUDO_ALTA=0.80
set P_SPM=0.45
set P_FUSIBLE=0.35
set P_POLO=0.25
set P_DENSA=0.28
set P_BARRA=0.10
set P_TRAFO=0.15
set P_CELDAS=0.18

%PY% src\train.py --name P_full >> log_P.txt 2>&1
copy /Y work\runs\P_full\best_limpio.pt best_componente_v11_P.pt >> log_P.txt 2>&1

%PY% src\evaluate.py best_componente_v11_P.pt 0.05 P_005 >> log_P.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v11_P.pt 0.20 P_marcelo >> log_P.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan P fin %date% %time% ===== >> log_P.txt
