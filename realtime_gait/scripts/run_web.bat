@echo off
REM Realtime gait web console
cd /d "%~dp0..\.."
call conda activate pytorch
python -m realtime_gait.web --host 127.0.0.1 --port 7860 --stream rtsp://127.0.0.1:8554/home
pause
