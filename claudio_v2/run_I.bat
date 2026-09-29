@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan I (fine tuning desde G) inicio %date% %time% ===== >> log_I.txt

REM Mismo dataset ds5 que H2, pero partiendo de G en vez de H.
set H2_EPOCHS=12
set PATIENCE=6

%PY% src\train.py --name I_ft >> log_I.txt 2>&1
copy /Y work\runs\I_ft\best_limpio.pt best_componente_v5_I.pt >> log_I.txt 2>&1

%PY% src\evaluate.py best_componente_v5_I.pt 0.05 I_005 >> log_I.txt 2>&1
set EVAL_SET=qet
set QET_DIR=test_marcelo
%PY% src\evaluate.py best_componente_v5_I.pt 0.20 I_marcelo >> log_I.txt 2>&1
set EVAL_SET=
set QET_DIR=

echo ===== plan I fin %date% %time% ===== >> log_I.txt
