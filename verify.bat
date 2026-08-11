@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :not_installed
.venv\Scripts\python.exe selftest.py
if errorlevel 1 goto :failed
.venv\Scripts\python.exe hanzistyleforge.py --config config.json check
if errorlevel 1 goto :failed
if exist "work\dataset\index.csv" (
  .venv\Scripts\python.exe hanzistyleforge.py --config config.json contract
  if errorlevel 1 goto :failed
)
echo HanziStyleForge project verification completed successfully.
pause
exit /b 0
:not_installed
echo Run install.bat first.
pause
exit /b 1
:failed
echo Verification failed. Review the error above.
pause
exit /b 1
