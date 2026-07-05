@echo off
setlocal

if "%~1"=="" (
  set STREAM_URL=rtsp://127.0.0.1:8554/home
) else (
  set STREAM_URL=%~1
)

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

python -m realtime_gait.run_stream --stream "%STREAM_URL%"

endlocal
