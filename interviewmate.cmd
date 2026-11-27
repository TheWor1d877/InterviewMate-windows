@echo off
rem InterviewMate 启动器（Windows）
rem
rem 用法：interviewmate.cmd [子命令]
rem 子命令见《2-骨架与配置》命令行入口一节：
rem start / warmup / stop / status / test / devices / audio-test / hotkey

setlocal

rem 从脚本自身位置推导仓库根，不依赖当前工作目录
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

rem 优先用仓库内的 venv 解释器，没有则退回 PATH 里的 python
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem 让 -m app.cli 在任意工作目录下都能找到 app 包
if defined PYTHONPATH (
  set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%REPO_ROOT%"
)

rem 退出码原样透传
"%PYTHON%" -m app.cli %*
exit /b %ERRORLEVEL%