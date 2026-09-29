@echo off
cd /d "%~dp0"
set PY=py -3
%PY% -c "import sys" 2>nul || set PY=python
echo ===== plan E: esperando que termine D %date% %time% ===== >> log_E.txt
:espera
findstr /C:"plan D fin" log_D.txt >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto espera
)
echo ===== plan E (largo) inicio %date% %time% ===== >> log_E.txt
set PATIENCE=60
%PY% src\train.py --name E_largo >> log_E.txt 2>&1
%PY% src\evaluate.py work\runs\E_largo\best_limpio.pt 0.20 E_largo >> log_E.txt 2>&1
copy /Y work\runs\E_largo\best_limpio.pt best_componente_v2_largo.pt >> log_E.txt 2>&1
echo ===== plan E fin %date% %time% ===== >> log_E.txt
