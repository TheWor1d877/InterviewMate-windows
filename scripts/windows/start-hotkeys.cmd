@echo off
rem Start the InterviewMate AutoHotkey v2 global hotkey script.
rem Called by interviewmate.cmd only for the "start" subcommand.
rem This script must never block the application start: callers ignore failures.

setlocal

rem Resolve the repository root relative to this script: scripts\windows -> repo root
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "HOTKEY_SCRIPT=%REPO_ROOT%\scripts\windows\hotkeys.ahk"

if not exist "%HOTKEY_SCRIPT%" (
  echo [hotkeys] Script not found: "%HOTKEY_SCRIPT%"
  exit /b 1
)

rem Do not start a second instance when AutoHotkey already runs this script.
set "RUNNING=0"
for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "Get-CimInstance Win32_Process ^| Where-Object { ($_.Name -like 'AutoHotkey*') -and ($_.CommandLine -like '*hotkeys.ahk*') } ^| Measure-Object ^| Select-Object -ExpandProperty Count" 2^>nul`) do set "RUNNING=%%C"
if not "%RUNNING%"=="0" (
  echo [hotkeys] AutoHotkey is already running hotkeys.ahk.
  exit /b 0
)

rem Probe common AutoHotkey v2 installation locations.
set "AHK_EXE="
for %%P in (
  "%ProgramFiles%\AutoHotkey\v2\AutoHotkey.exe"
  "%ProgramFiles%\AutoHotkey\v2\AutoHotkey64.exe"
  "%ProgramFiles%\AutoHotkey\AutoHotkey.exe"
  "%LocalAppData%\Programs\AutoHotkey\v2\AutoHotkey.exe"
) do (
  if not defined AHK_EXE if exist "%%~P" set "AHK_EXE=%%~P"
)

if not defined AHK_EXE (
  echo [hotkeys] AutoHotkey v2 not found. Install it to enable global F8/F9 while unfocused.
  exit /b 1
)

start "" "%AHK_EXE%" "%HOTKEY_SCRIPT%"
echo [hotkeys] Started AutoHotkey with hotkeys.ahk.
exit /b 0