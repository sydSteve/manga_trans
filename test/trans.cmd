@echo off
setlocal

cd /d "%~dp0"
call "%~dp0..\venv\Scripts\activate.bat"
python "%~dp0trans.py" %*
