@echo off
setlocal
cd /d "%~dp0"
if exist STOP_AFTER_CHECKPOINT (
  del /q STOP_AFTER_CHECKPOINT
  echo Safe-stop request cleared.
) else (
  echo No safe-stop request was pending.
)
echo run_months_resilient.bat also clears it on launch, so this is only needed
echo when starting a run some other way.
pause
