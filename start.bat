@echo off
title AuraVision.AI - Industrial Visual Anomaly Detection
echo ======================================================================
echo   Launching AuraVision.AI Platform
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_project.py
) else (
    python run_project.py
)

pause
