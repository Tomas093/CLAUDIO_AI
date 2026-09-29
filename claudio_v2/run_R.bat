@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan R (Q sin comerse las planillas) inicio %date% %time% ===== >> log_R.txt

REM ds13 = ds12 + lo que falta para juntar las virtudes de N y de Q (ver el hook R_full).
REM   puls_contactor (P_PULS): contactor y pulsador con UNA CAJA CADA UNO.
REM   caja_nombre a peso simple: con peso doble el modelo fusiona simbolos vecinos.
REM   P_CELDAS 0.18 -> 0.25 y vocabulario real de planilla (esto es lo de menos: 0,9%).
set R_EPOCHS=200
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
set GENS_NOMBRE_SIMPLE=1

%PY% src\train.py --name R_full >> log_R.txt 2>&1
copy /Y work\runs\R_full\best_limpio.pt best_componente_v13_R.pt >> log_R.txt 2>&1

set GT_V4=1
%PY% src\evaluate.py best_componente_v13_R.pt 0.05 V4_v13_R_005 >> log_R.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v13_R.pt 0.20 V4_v13_R_marcelo >> log_R.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan R fin %date% %time% ===== >> log_R.txt
