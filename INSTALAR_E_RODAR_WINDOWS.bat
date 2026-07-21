@echo off
chcp 65001 >nul
echo ==========================================
echo GERADOR EXCEL FIFA 2026 - EDGE/SELENIUM v5
echo ==========================================
py -3 -m venv .venv
if errorlevel 1 goto :erro
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :erro
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :erro
.venv\Scripts\python.exe gerador_excel_fifa2026_edge_v5.py
if errorlevel 1 goto :erro
exit /b 0

:erro
echo Falha na instalacao ou execucao. Revise as mensagens acima.
pause
exit /b 1
