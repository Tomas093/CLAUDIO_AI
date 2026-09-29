@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan F inicio %date% %time% ===== >> log_F.txt

REM Ajuste fino sobre los fallos vistos en planos reales del cliente (16/09).
REM Parte del mejor de D. El dataset ds3 se genera dentro de train.py con estas probabilidades.
set F_EPOCHS=18
set P_SPM=0.30
set P_DENSA=0.30
set P_BARRA=0.12
set P_TRAFO=0.18
set P_CELDAS=0.22

%PY% src\train.py --name F_spm >> log_F.txt 2>&1

REM Pesos limpios del plan F (no pisa best_componente_v2.pt: la comparacion con D la hace Claude)
copy /Y work\runs\F_spm\best_limpio.pt best_componente_v2_F.pt >> log_F.txt 2>&1

REM Evaluaciones: los 4 planos de test de siempre y los 3 del cliente
%PY% src\evaluate.py best_componente_v2_F.pt 0.20 F_test >> log_F.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v2_F.pt 0.20 F_marcelo >> log_F.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan F fin %date% %time% ===== >> log_F.txt
