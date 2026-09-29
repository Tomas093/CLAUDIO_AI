@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== inicio %date% %time% ===== >> log_noche.txt
%PY% -m pip install -q ezdxf ultralytics opencv-python matplotlib pillow >> log_noche.txt 2>&1
%PY% -c "import torch;print('CUDA',torch.cuda.is_available())" >> log_noche.txt 2>&1
%PY% src\noche.py >> log_noche.txt 2>&1
echo ===== fin %date% %time% ===== >> log_noche.txt
