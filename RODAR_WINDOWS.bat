@echo off
chcp 65001 >nul
if not exist .venv\Scripts\python.exe (
    echo Ambiente virtual nao encontrado. Execute INSTALAR_E_RODAR_WINDOWS.bat primeiro.
    pause
    exit /b 1
)
.venv\Scripts\python.exe gerador_excel_fifa2026_edge_v5.py
pause
