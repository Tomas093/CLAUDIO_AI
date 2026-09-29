@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan S (R + recuadro punteado con nombre) inicio %date% %time% ===== >> log_S.txt

REM ds14 = ds13 + procsym.caja_punteada. Es el UNICO cambio sobre R: el recuadro de linea
REM punteada con el nombre de un equipo adentro (el PLC de nyw-un-01, ultimo FN real).
set S_EPOCHS=200
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
set P_CELDAS=0.25
set P_TERNA=0.35
set P_PULS=0.30
set P_PAT=0.30
set GENS_NOMBRE_SIMPLE=1

%PY% src\train.py --name S_full >> log_S.txt 2>&1
copy /Y work\runs\S_full\best_limpio.pt best_componente_v14_S.pt >> log_S.txt 2>&1
echo ===== plan S fin %date% %time% ===== >> log_S.txt
