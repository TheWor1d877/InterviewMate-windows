; InterviewMate 全局热键（AutoHotkey v2）
;
; F8 触发语音开始/停止切换，F9 截图并立即提交。两个键都通过
;   python -m app.cli hotkey voice-toggle
;   python -m app.cli hotkey screenshot
; 通知正在运行的主进程，主进程不在时 CLI 自己返回 E_SOCKET，不影响输入。
;
; 用法：安装 AutoHotkey v2 后双击本脚本即可常驻；interviewmate.cmd start
; 会自动尝试拉起。热键全局无条件注册：AHK 在跑时界面会短路自己的 F8/F9
; 绑定，同一次按键只走 socket 一条路；终端在前台也由本脚本处理，
; 否则「界面短路 + AHK 前台不接管」会变成两边都不响应。

#Requires AutoHotkey v2.0
#SingleInstance Force
; 连按同一个热键时不重入，避免叠加两次 CLI 调用
#MaxThreadsPerHotkey 1

; 脚本位于 <仓库根>\scripts\windows，向上两级得到仓库根
RepoRoot := A_ScriptDir "\..\.."

; 只用仓库内的 venv 解释器，不回退 PATH 里的裸 python
PythonExe := RepoRoot "\.venv\Scripts\python.exe"
if !FileExist(PythonExe) {
    TrayTip("Python executable not found: " . PythonExe, "InterviewMate")
    ExitApp(1)
}

RunHotkey(action) {
    global PythonExe, RepoRoot
    ; 与 interviewmate.cmd 保持一致：从仓库根执行，PYTHONPATH 指向仓库根
    EnvSet("PYTHONPATH", RepoRoot)
    try {
        ExitCode := RunWait('"' . PythonExe . '" -m app.cli hotkey ' . action, RepoRoot, "Hide")
    } catch as err {
        TrayTip("InterviewMate hotkey failed: " . err.Message, "InterviewMate")
        return
    }
    if (ExitCode != 0)
        TrayTip("InterviewMate hotkey failed with exit code " . ExitCode, "InterviewMate")
}

; 全局无条件注册。界面在 AHK 可用时（mark_global_hotkeys_active）会短路
; 自己的 F8/F9 绑定，因此不会重复处理；终端在前台时也必须由本脚本接管。
F8::RunHotkey("voice-toggle")
F9::RunHotkey("screenshot")