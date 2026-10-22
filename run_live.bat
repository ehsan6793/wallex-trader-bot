@echo off
REM اجرای زنده ربات روی ویندوز. قبلش: فایل .env را پر کنید و کامپیوتر را از حالت Sleep خارج کنید.
cd /d %~dp0
python -m bot live --yes
pause
