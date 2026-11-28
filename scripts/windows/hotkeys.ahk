; InterviewMate 全局热键（AutoHotkey v2）
;
; F7 截图并立即提交；F8 按住说话：按下开始录音，松开停止并提交最终文本；
; F9 项目模式开关；F10 上下文视图显示/切回；F11 清空上下文、重开一轮。
; 六个动作都通过
;   python -m app.cli hotkey voice-start
;   python -m app.cli hotkey voice-stop
;   python -m app.cli hotkey screenshot
;   python -m app.cli hotkey project-mode-toggle
;   python -m app.cli hotkey context-view-toggle
;   python -m app.cli hotkey context-clear
; 通知正在运行的主进程，主进程不在时 CLI 自己返回 E_SOCKET，不影响输入。
;
; 用法：安装 AutoHotkey v2 后双击本脚本即可常驻；interviewmate.cmd start
; 会自动尝试拉起。热键全局无条件注册：AHK 在跑时界面会短路自己的截图
; 绑定（F8/F9/F10/F11 不在界面绑定），同一次按键只走 socket 一条路；
; 终端在前台也由本脚本处理，否则「界面短路 + AHK 前台不接管」会变成
; 两边都不响应。

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

; 上一次已经提示过的失败。同一失败只提示一次，成功一次就清空；
; 这样 CLI 持续不可用时不会反复弹气泡，恢复后再失败会重新提示一次。
LastNotifiedFailure := ""

RunHotkey(action) {
    global PythonExe, RepoRoot, LastNotifiedFailure
    ; 与 interviewmate.cmd 保持一致：从仓库根执行，PYTHONPATH 指向仓库根
    EnvSet("PYTHONPATH", RepoRoot)
    try {
        ExitCode := RunWait('"' . PythonExe . '" -m app.cli hotkey ' . action, RepoRoot, "Hide")
    } catch as err {
        NotifyFailureOnce("hotkey failed: " . err.Message)
        return
    }
    if (ExitCode != 0) {
        NotifyFailureOnce("hotkey failed with exit code " . ExitCode)
        return
    }
    ; CLI 成功一次就把去重状态清空：下次再失败仍然提示一次
    LastNotifiedFailure := ""
}

; 同一个失败只提示一次。启动期的「Python 可执行文件缺失」不走这里，
; 仍然每次启动都会提示，见脚本顶部。
NotifyFailureOnce(message) {
    global LastNotifiedFailure
    if (message = LastNotifiedFailure)
        return
    LastNotifiedFailure := message
    TrayTip("InterviewMate " . message, "InterviewMate")
}

; 全局无条件注册。界面在 AHK 可用时（mark_global_hotkeys_active）会短路
; 自己的截图绑定，因此不会重复处理；终端在前台时也必须由本脚本接管。
;
; F8 按住说话：按下发 voice-start，松开发 voice-stop。
; 按下分支先发一次 start 再用 KeyWait 等到松开，按住期间键盘自动重复
; 触发的 down 事件因 #MaxThreadsPerHotkey 1 被丢弃，不会连发 voice-start；
; 松开由独立的 F8 up 热键接住（down 与 up 是两个热键，各有自己的线程槽），
; 发出 voice-stop。这对动作 AHK 不感知模式：普通模式主进程按采集解释，
; 项目模式主进程把同一对动作解释成「截取窗口开/关」。
; F7 截图、F9 项目模式开关、F10 上下文视图切换、F11 清空上下文，
; 都是按一下触发一次。
F8::
{
    RunHotkey("voice-start")
    KeyWait "F8"
}
F8 up::RunHotkey("voice-stop")
F7::RunHotkey("screenshot")
F9::RunHotkey("project-mode-toggle")
F10::RunHotkey("context-view-toggle")
F11::RunHotkey("context-clear")