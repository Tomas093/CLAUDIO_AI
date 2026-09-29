@echo off
setlocal
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== [0] dependencias ===== > log_todo.txt
%PY% -m pip install -q ezdxf ultralytics opencv-python matplotlib pillow >> log_todo.txt 2>&1
echo ===== [1] dataset ===== >> log_todo.txt
%PY% src\build_all.py >> log_todo.txt 2>&1 || goto :err
echo ===== [2] entrenamiento ===== >> log_todo.txt
%PY% src\train.py >> log_todo.txt 2>&1 || goto :err
echo ===== [3] evaluacion sobre DXF ===== >> log_todo.txt
%PY% src\evaluate.py best_componente_v2.pt 0.20 >> log_todo.txt 2>&1 || goto :err
echo ===== LISTO ===== >> log_todo.txt
exit /b 0
:err
echo ===== ERROR (ver arriba) ===== >> log_todo.txt
exit /b 1
