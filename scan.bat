@echo off
REM RSI(2) scanner - double-click to run, or call from Task Scheduler.
REM Runs the scanner, then updates the paper-trading journal and opens it.
REM Pass --no-paper to skip the journal (use this for headless scheduled runs).
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "RUN_PAPER=1"
set "SCAN_ARGS="
for %%A in (%*) do (
    if /I "%%~A"=="--no-paper" (
        set "RUN_PAPER=0"
    ) else (
        set "SCAN_ARGS=!SCAN_ARGS! %%~A"
    )
)

".venv\Scripts\python.exe" "scanner\rsi2_scanner.py"!SCAN_ARGS!

if "!RUN_PAPER!"=="1" (
    echo.
    echo ============================================================
    echo   Updating paper-trading journal
    echo ============================================================
    ".venv\Scripts\python.exe" "scanner\paper_options.py" --universe strong
    REM HTML, not the .md - Windows has no reliable handler for markdown but always has a browser.
    if exist "output\PAPER_OPTIONS.html" (
        echo.
        echo Opening journal...
        start "" "output\PAPER_OPTIONS.html"
    )
)

endlocal
echo.
pause
