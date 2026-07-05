@echo off
cd /d %~dp0
call venv\Scripts\activate
set HF_HUB_OFFLINE=1
python elara_tray.py