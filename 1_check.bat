@echo off
cd /d %~dp0
python -m pip install -q -r requirements.txt
python -m bot check > check_output.txt 2>&1
type check_output.txt
pause
