@echo off
setlocal

cd /d "%~dp0"

set "HF_HOME=%~dp0.cache\huggingface"
set "TRANSFORMERS_CACHE=%~dp0.cache\huggingface\transformers"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"

".\venv\Scripts\python.exe" launch.py %*
pause
