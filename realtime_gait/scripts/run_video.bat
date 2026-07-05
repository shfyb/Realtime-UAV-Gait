@echo off
setlocal

if "%~1"=="" (
  echo Usage: run_video.bat ^<video_path^>
  exit /b 1
)

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

python -m realtime_gait.main --video "%~1" --display

endlocal
