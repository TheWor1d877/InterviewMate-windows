; InterviewMate 全局热键（AutoHotkey v2）
;
; F8 触发语音开始/停止，F9 截图并立即提交。两个键都通过
;   python -m app.cli hotkey voice-toggle
;   python -m app.cli hotkey screenshot
; 通知正在运行的主进程，主进程不在时 CLI 自己返回 E_SOCKET，不影响输入。
;
; 用法：安装 AutoHotkey v2 后双击本脚本即可常驻。
; 终端窗口处于前台时热键不触发，避免与终端内原有的 F8/F9 绑定重复处理。

#Requires AutoHotkey v2.0
#SingleInstance Force

; 脚本位于 <仓库根>\scripts\windows，向上两级得到仓库根
RepoRoot := A_ScriptDir "\..\.."

; 优先仓库内的 venv 解释器，没有则用 PATH 里的 python
VenvPython := RepoRoot "\.venv\Scripts\python.exe"
PythonExe := FileExist(VenvPython) ? VenvPython : "python"

; 允许用环境变量覆盖解释器路径，便于换机器时不改脚本
if EnvGet("INTERVIEWMATE_PYTHON") != ""
    PythonExe := EnvGet("INTERVIEWMATE_PYTHON")

RunHotkey(action) {
    global PythonExe, RepoRoot
    ; 与 interviewmate.cmd 保持一致：从仓库根执行，PYTHONPATH 指向仓库根
    EnvSet "PYTHONPATH", RepoRoot
    try {
        RunWait('"' PythonExe '" -m app.cli hotkey ' action, RepoRoot, "Hide")
    } catch as err {
        MsgBox "InterviewMate 热键调用失败：" err.Message, "InterviewMate", "Icon!"
    }
}

; 终端窗口里不接管 F8/F9，避免与终端内的按键各触发一次
#HotIf !WinActive("ahk_exe WindowsTerminal.exe")
F8::RunHotkey("voice-toggle")
F9::RunHotkey("screenshot")
#HotIf