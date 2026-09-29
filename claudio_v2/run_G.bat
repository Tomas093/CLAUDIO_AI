@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan G (desde cero) inicio %date% %time% ===== >> log_G.txt

REM Modelo nuevo, NO parte de D. Dataset ds3: incluye SPM, filas densas, barras colectoras,
REM trafo vs par de interruptores y celdas de tabla; sin PAT.
REM Corta por epocas y por paciencia, sin limite de reloj.
set G_EPOCHS=200
set PATIENCE=40
set P_SPM=0.30
set P_DENSA=0.30
set P_BARRA=0.12
set P_TRAFO=0.18
set P_CELDAS=0.22

%PY% src\train.py --name G_scratch >> log_G.txt 2>&1

copy /Y work\runs\G_scratch\best_limpio.pt best_componente_v3.pt >> log_G.txt 2>&1

REM Evaluaciones: los 4 planos de test de siempre y los 3 del cliente
%PY% src\evaluate.py best_componente_v3.pt 0.20 G_test >> log_G.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v3.pt 0.20 G_marcelo >> log_G.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan G fin %date% %time% ===== >> log_G.txt
