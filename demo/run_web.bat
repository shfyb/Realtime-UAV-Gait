@echo off
setlocal
cd /d %~dp0..
call realtime_gait\scripts\run_web.bat %*
endlocal
