@echo off
cd /d "%~dp0"
set PY=py -3
set GT_V3=1
set MODELOS=v9_N v12_Q v7_L v8_M v10_O v6_K v6_J v5_I v5_H2 v4 v3 v2

set EVAL_SET=
set QET_DIR=
for %%M in (%MODELOS%) do (
  echo ===== %%M base =====
  %PY% src\evaluate.py best_componente_%%M.pt 0.05 V3_%%M_005
)

set EVAL_SET=qet
set QET_DIR=test_marcelo
for %%M in (%MODELOS%) do (
  echo ===== %%M marcelo =====
  %PY% src\evaluate.py best_componente_%%M.pt 0.20 V3_%%M_marcelo
)
set EVAL_SET=
set QET_DIR=
echo ===== TODOS LISTO =====
