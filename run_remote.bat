@echo off
setlocal enabledelayedexpansion

:: Get the dragged file path (the first argument)
set "WIN_PATH=%~1"

:: If no file was dragged, prompt the user to drag and drop it here
if "%WIN_PATH%"=="" (
    echo === CorridorKey v2 Remote SSH Launcher ===
    set /p "WIN_PATH=Drag and drop your Windows video file (or frame directory) here and press Enter: "
)

:: Remove surrounding quotes if manually entered
set "WIN_PATH=!WIN_PATH:"=!"

:: Run the interactive script on the Linux GPU workstation via SSH.
:: We use the -t flag to force pseudo-terminal allocation so the interactive prompts
:: and colored terminal styling function correctly over the SSH connection.
ssh -t corridor@10.10.10.109 "cd /home/corridor/CK2.0/corridorkey_v2_runtime && ./venv/bin/python3 run_interactive.py '!WIN_PATH!'"

echo.
echo SSH connection closed.
pause
