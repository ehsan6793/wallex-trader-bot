@echo off
cd /d %~dp0
start "Wallex dashboard" python -m bot dashboard
python -m bot shadow
pause
