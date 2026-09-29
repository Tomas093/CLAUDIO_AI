@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan D inicio %date% %time% ===== >> log_D.txt
set D_HOURS=6
%PY% src\train.py --name B_s_largo >> log_D.txt 2>&1
%PY% src\evaluate.py work\runs\B_s_largo\best_limpio.pt 0.20 D_final >> log_D.txt 2>&1
copy /Y work\runs\B_s_largo\best_limpio.pt best_componente_v2.pt >> log_D.txt 2>&1
echo ===== plan D fin %date% %time% ===== >> log_D.txt
