@echo off
REM اجرای زنده: سفارش واقعی ثبت می‌کند. کامپیوتر نباید Sleep شود. داشبورد خودکار باز می‌شود.
cd /d %~dp0
start "Wallex dashboard" python -m bot dashboard
python -m bot live --yes
pause
