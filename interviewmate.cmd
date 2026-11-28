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

rem 仅在 start 子命令时先拉起全局热键助手（原 scripts\windows\start-hotkeys.cmd
rem 已并入本脚本，一条 interviewmate.cmd start 完成）；失败只提示，不阻断启动
if /i "%~1"=="start" goto :start-hotkeys
goto :run-app

:start-hotkeys
set "HOTKEY_SCRIPT=%REPO_ROOT%\scripts\windows\hotkeys.ahk"
if not exist "%HOTKEY_SCRIPT%" (
  echo [hotkeys] Script not found: "%HOTKEY_SCRIPT%"
  goto :run-app
)

rem ---- AHK v2 探测：默认安装路径 -> 注册表 InstallDir -> PATH ----
rem 全程用「for 变量 + if not defined」的结构：if 块内的 %VAR% 在整块解析期
rem 就展开，拿不到同块内刚 set 的值，所以 set 之后的使用必须另起语句。
set "AHK_EXE="
for %%P in (
  "%ProgramFiles%\AutoHotkey\v2\AutoHotkey64.exe"
  "%ProgramFiles%\AutoHotkey\v2\AutoHotkey32.exe"
  "%ProgramFiles%\AutoHotkey\v2\AutoHotkey.exe"
  "%ProgramFiles%\AutoHotkey\AutoHotkey.exe"
  "%LocalAppData%\Programs\AutoHotkey\v2\AutoHotkey64.exe"
  "%LocalAppData%\Programs\AutoHotkey\v2\AutoHotkey32.exe"
  "%LocalAppData%\Programs\AutoHotkey\v2\AutoHotkey.exe"
) do if not defined AHK_EXE if exist "%%~P" set "AHK_EXE=%%~P"

rem 注册表兜底：per-user 安装器可能装在别处，HKCU 优先、HKLM 次之。
rem 读注册表与使用 AHK_INSTALL 是两条独立语句；探测动作抽成子例程，
rem 与默认路径共用同一份候选列表。
if defined AHK_EXE goto :hotkeys-launch
set "AHK_INSTALL="
for /f "tokens=2,*" %%R in ('reg query "HKCU\Software\AutoHotkey" /v InstallDir 2^>nul') do set "AHK_INSTALL=%%S"
if not defined AHK_INSTALL for /f "tokens=2,*" %%R in ('reg query "HKLM\SOFTWARE\AutoHotkey" /v InstallDir 2^>nul') do set "AHK_INSTALL=%%S"
if defined AHK_INSTALL call :probe-ahk-dir "%AHK_INSTALL%"

rem PATH 兜底：where.exe 系统自带，无额外依赖
if defined AHK_EXE goto :hotkeys-launch
for %%N in (AutoHotkey64.exe AutoHotkey32.exe AutoHotkey.exe) do (
  if not defined AHK_EXE (
    for /f "delims=" %%W in ('where %%N 2^>nul') do if not defined AHK_EXE set "AHK_EXE=%%W"
  )
)

:hotkeys-launch
if not defined AHK_EXE (
  echo [hotkeys] AutoHotkey v2 not found. Install it to enable global F8/F9 while unfocused.
  goto :run-app
)

rem hotkeys.ahk 顶部是 #SingleInstance Force：重复 start 会替换旧实例，
rem 热键脚本更新后重跑 interviewmate.cmd start 即完成重载，
rem 因此这里无条件启动，不做「是否已运行」的幂等检测。
start "" "%AHK_EXE%" "%HOTKEY_SCRIPT%"
echo [hotkeys] Started AutoHotkey with hotkeys.ahk.

:run-app
rem 退出码原样透传
"%PYTHON%" -m app.cli %*
exit /b %ERRORLEVEL%

rem 子例程：在给定安装目录（%1）下探测 AutoHotkey v2 可执行文件，
rem 找到时写入 AHK_EXE（call 不复制变量环境，调用方直接可见）。
:probe-ahk-dir
set "AHK_EXE="
for %%P in (
  "%~1\v2\AutoHotkey64.exe"
  "%~1\v2\AutoHotkey32.exe"
  "%~1\v2\AutoHotkey.exe"
  "%~1\AutoHotkey.exe"
) do if not defined AHK_EXE if exist "%%~P" set "AHK_EXE=%%~P"
exit /b 0