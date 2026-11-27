# 平台差异与 Windows 适配

本文件是 Windows 侧唯一入库的文档。Linux 侧的实现、配置与契约写在《1-总纲》《2-骨架与配置》《3-文本输入与流式回答》《4-截图输入与多模态》《5-语音输入与流式ASR》《6-Dify检索与项目知识》《7-代码问题模式》《8-集成与验收》《接口契约》里。

迁移、安装与排错的具体命令写在本地 `docs/9.1-Windows迁移操作指南.md`。那份文件是临时操作指南，不进 Git；本文件只写目标形态、契约与验收。

两边的说法冲突时，以本文件为准；操作步骤以 9.1 为准。

## Summary

Windows 侧是同一套 InterviewMate 在 Windows 上的运行形态，不是第二个产品。协议、配置结构、状态取值、错误码、事件流全部共用，只有平台相关的能力分成两套实现。

本文件回答三个问题：

- 从 Ubuntu 这一版到 Windows 版要改哪些地方。
- 两套怎么并存，怎么防止跑偏。
- Windows 侧的目标机器、依赖、验收怎么定。

当前状态：Linux 侧 M1 到 M9 已完成，平台门面与五个后端可用。Windows 侧五个后端仍是接口占位，调用抛 `NotImplementedError`，实现顺序见「实施顺序」一节。迁移与安装按本地 9.1 操作指南执行。

## Windows 侧运行形态

Windows 侧分两层：桌面层与后台服务层。

| 层 | 跑什么 | 跑在哪 |
|---|---|---|
| 桌面层 | InterviewMate TUI、ASR、音频采集、截图、编译 | Windows 原生 Python |
| 服务层 | Dify（api、worker、web、nginx、db、redis、qdrant、plugin_daemon）与 Ollama | WSL2 里的 Ubuntu 24.04 与 Docker Engine |

这样分的理由：

- Dify 只有 Docker 形态，Windows 上跑 Linux 容器必须有 WSL2。
- Ollama 与 Dify 在同一台 WSL 里，Dify 访问 Ollama 的地址仍是 `http://host.docker.internal:11434`，与 Ubuntu 侧一致，compose 与 `.env` 不用改。
- Ubuntu 上用的 bash 命令、`systemctl --user ollama` 原样可用；`scripts/linux/stack.sh` 已删除（脚本已删除，项目转 Windows-only），启停改用根目录 `stack.cmd up/down`。
- 音频、截图、ASR 与界面都要碰 Windows 桌面，留在原生 Python 最省事。

Windows 侧不装 Docker Desktop，Docker Engine 装在 WSL 的 Ubuntu 里；服务操作在 WSL 的 bash 里做，应用操作在 Git Bash 里做。

数据放在 WSL 的 ext4 里，不放在 `/mnt/c`：跨文件系统的权限与性能都不合适。

## 两套装置如何并存

### 为什么是单仓库

平台相关的代码只有 8 个调用点，其余代码与平台无关：

| 位置 | 平台相关的内容 |
|---|---|
| `app/main.py` 的截图接口探测 | D-Bus 调用静默截图扩展，兜底 XDG Portal |
| `app/main.py` 的音频设备枚举 | `sounddevice` |
| `app/cli.py` 的 `devices` 命令 | `sounddevice` |
| `app/runtime/socket_server.py` 的监听 | `AF_UNIX` socket |
| `app/runtime/socket_server.py` 的权限收紧 | `os.chmod` |
| `app/runtime/process_manager.py` 的存活探测 | `os.kill(pid, 0)` |
| `app/runtime/process_manager.py` 的残留连接检查 | `socket.AF_UNIX` |
| `app/runtime/socket_client.py` 的两处连接 | `socket.AF_UNIX` |

这 8 个调用点现在统一走 `app/platform/` 门面，平台相关实现落在 `app/platform/linux/` 与 `app/platform/windows/` 下。

分成两个仓库会带来三个具体后果。`socket_protocol.py` 的请求与响应格式会漂移。DeepSeek 与 Dify 客户端要各写一遍。M1 到 M9 每个里程碑的实现与验收标准都要写两遍。这个项目是用户自己一个人用的玩具工具，多出来的维护量换不来东西。

### 目录结构

平台能力放在 `app/platform/` 下：

```text
app/platform/
├── __init__.py        唯一选择点，get_platform()
├── types.py           平台能力接口定义
├── linux/
│   ├── screenshot.py  静默截图扩展，兜底 XDG Portal
│   ├── audio.py       PipeWire 与 PulseAudio 的回环采集
│   ├── transport.py   AF_UNIX socket 传输
│   ├── process.py     进程存活探测
│   └── compiler.py    g++
└── windows/
    ├── screenshot.py  mss 截屏
    ├── audio.py       WASAPI 回环采集
    ├── transport.py   TCP loopback 传输
    ├── process.py     进程存活探测
    └── compiler.py    cl.exe 与 MinGW-w64 g++
```

规则有两条：

- 平台在进程启动时选定一次，上层任何模块都不判断 `sys.platform`。
- 门面返回的结果两平台同构。上层拿到的截图是内存里的字节与宽高，音频设备是一张带名称、索引、采样率、声道数的表，IPC 是同一个请求与响应格式。

现状：`app/platform/` 目录已经建好。Linux 侧五个后端都已实现并被上层调用；Windows 侧五个后端目前是接口占位，调用会抛 `NotImplementedError`，实现顺序见本文件「实施顺序」一节。

### 界面契约

| 能力 | 接口返回 |
|---|---|
| 截图 | 图片字节、宽、高 |
| 音频设备枚举 | 名称、索引、是否输入、是否回环、默认采样率、声道数 |
| 音频采集 | 按设备原生采样率与声道数开流，下混与重采样由平台后端完成 |
| IPC | 一行 JSON 请求、一行 JSON 响应；Windows 侧在传输层多一次 token 握手，字段见《2-骨架与配置》进程通信一节 |
| 进程存活 | 布尔值 |

### 不跑偏的保障

当前仓库没有 CI 配置。两平台并存要靠这个才不烂掉，需要补：

- GitHub Actions 跑 `ubuntu-latest` 与 `windows-latest` 两个 runner，执行同一套 pytest。
- 单 `main` 分支承载两平台，不开长期分叉分支。
- `app/runtime/socket_protocol.py` 只有一份，任何平台都不复制。

## 平台能力矩阵

下表描述两边的目标形态，不是当前进度。Windows 侧的真后端实现按本文件「实施顺序」一节做。

| 能力 | Linux | Windows |
|---|---|---|
| 截屏 | GNOME Shell 静默截图扩展，兜底 XDG Portal | `mss` |
| 截屏是否需要用户确认 | 装上扩展就不需要，兜底 Portal 时需要 | 不需要，静默抓取 |
| 截屏范围 | 按配置：整屏、主屏或指定屏 | 按配置，mss 支持整屏与单屏 |
| 截屏落盘 | 扩展与 Portal 都是临时文件，读完删除 | 不落盘，直接在内存 |
| 麦克风采集 | `sounddevice` | `sounddevice` |
| 系统回环音频 | PulseAudio 与 PipeWire 的 monitor source | WASAPI loopback |
| 音频设备枚举 | `sounddevice` | `sounddevice` 与 `soundcard` |
| 进程通信 | AF_UNIX socket 文件 | TCP loopback 加端口文件加随机 token |
| 进程存活探测 | `os.kill(pid, 0)` | `psutil` |
| 退出触发 | 信号与 IPC `stop` 动作 | IPC `stop` 动作，Ctrl+C 兜底 |
| 启动入口 | 仓库根的 `interviewmate` 脚本 | 新增 `interviewmate.cmd` |
| 统一入口 | `python -m app.cli` | `python -m app.cli` |
| 代码编译器 | `g++` | `cl.exe` 或 MinGW-w64 的 `g++` |
| 终端 | 任意现代终端 | Windows Terminal |
| 后台服务 | 本机 systemd 与 Docker | WSL2 Ubuntu 里的 systemd 与 Docker Engine |
| 全局热键监听 | 进程内监听，需在 input 组；失焦可用，F8 按住说话 | 不支持全局监听，F8、F9 仅终端内可用，F8 为切换语义 |
| 全局快捷键绑定 | 进程内监听优先，也可用 GNOME 自定义快捷键调 CLI | AutoHotkey v2 静默调 CLI 的 `hotkey` 子命令 |
| ASR 加速 | NVIDIA GPU，CUDA | 默认 CPU，给 Ollama 留显存；显存够时可切 CUDA |

## Linux 截图后端

Linux 侧走 Wayland，截图有两条路：主路是本仓库自带的 GNOME Shell 扩展，兜底是 XDG Portal。

扩展在 GNOME Shell 进程内跑，用 `Shell.Screenshot` 抓配置指定的那块画面，屏幕上不出现任何提示。扩展源码在 `extensions/interviewmate-screenshot/`，装法见《4-截图输入与多模态》「静默截图扩展」一节，调用形态见《接口契约》「GNOME 静默截图扩展」一节。

兜底的 XDG Portal 是标准接口，GNOME 下必定弹一次确认框。

| 项 | 静默截图扩展 | XDG Portal |
|---|---|---|
| 是否弹界面 | 不弹 | 弹一次确认框 |
| 抓哪一块 | 按配置：整屏（含多显示器拼接）、主屏或指定屏 | 只能整屏 |
| 落盘 | 不落盘，PNG 直接写进内存流 | Portal 临时文件，读完删除 |
| 依赖 | GNOME Shell 扩展已启用 | `dbus-next` 与桌面环境的 portal 服务 |

## Windows 截图后端

### 调用形态

用 `mss` 抓整屏。`mss` 在 Windows 上支持整屏与多屏抓取，纯 Python，没有系统级依赖。

`mss` 返回的是内存里的位图，转成 PNG 字节后交给上层。上层拿到的形态与 Linux 侧一致：图片字节、宽、高。

### 与 Linux 侧的差异

| 项 | Linux | Windows |
|---|---|---|
| 是否弹界面 | 装了静默截图扩展就不弹，兜底 Portal 时弹 | 不弹 |
| 用户操作 | 装了扩展无操作，兜底时点一次确认 | 无 |
| 是否落盘 | 扩展与 Portal 都是临时文件，读完删除 | 不落盘 |
| 失败形态 | 扩展没装或调用失败、Portal 报错、用户拒绝、超时 | 抓取失败、设备不可用 |

Linux 侧的确认框来自 XDG Portal。`interactive=false` 只表示不让用户先调参数，跳不过 Portal 自己的确认界面，GNOME 下的策略是一定会弹。

装上本仓库自带的 GNOME Shell 扩展后就没有确认框了，扩展的装法见《4-截图输入与多模态》「静默截图扩展」一节。

### 失败处理

抓取失败、显示器不可用、权限不足，都返回 `E_SCREENSHOT`，状态栏提示截图失败，程序继续运行。处理方式与 Linux 侧一致，错误码定义见《2-骨架与配置》。

### 依赖

`mss`。Windows 侧不装 `dbus-next`。

## Windows 音频采集

### 设备枚举

麦克风枚举用 `sounddevice`，与 Linux 侧同一个包。Windows 上的设备名、索引顺序与 Linux 不同，`devices` 命令的输出格式要统一，格式定义见《5-语音输入与流式ASR》。

### 系统回环音频

Linux 侧的 `system_monitor` 落在 PulseAudio 与 PipeWire 的 monitor source 上。Windows 没有 monitor source 这个概念，对应物是 WASAPI loopback：把默认扬声器当作输入端打开，采到的是系统播放的混合流。

用 `soundcard` 实现，`get_microphone(id=..., include_loopback=True)`。PortAudio 不暴露 loopback 设备枚举，所以系统回环这一路不能只靠 `sounddevice`。

### 来源枚举的对应关系

配置项 `audio.source` 的取值两平台一致，落地机制不同：

| 取值 | Linux | Windows |
|---|---|---|
| `microphone` | 麦克风输入设备 | 麦克风输入设备 |
| `system_monitor` | PulseAudio 与 PipeWire 的 monitor source | 默认扬声器的 WASAPI loopback |
| `mixed` | 麦克风与回环各采一路，带来源标签 | 同左，loopback 一路走 `soundcard` |
| `auto` | 默认麦克风与默认回环都有时混成一路，只有一路时用那一路 | 同左，默认设备取 WASAPI 的默认端点 |

`auto` 两个都不通时报 `E_AUDIO_DEVICE`。两路都在时结果是 `mixed`，不是回环优先。默认设备由后端标记 `is_default`，配置不绑定设备名或索引。

### 采样率与声道

两个平台走同一套流程，下混与重采样都在平台后端内完成：

1. 按设备原生采样率与原生声道数开流。
2. 下混成单声道。
3. 重采样到 16000 Hz。
4. 按 9600 点拼成 600 毫秒一块送进模型，块长关系见《接口契约》的「FunASR 音频块参数」。

Windows 侧的回环一路必是立体声，原生采样率多为 48000 Hz。Linux 侧的默认设备可能报出上百个输入声道，开流要按两声道封顶，否则 PortAudio 会崩。上层拿到的永远是 16000 Hz 单声道，重采样与下混不是可选项。

### 依赖

`soundcard`、`numpy`。

## Windows 进程通信

### 传输层

Windows 侧用 TCP loopback：

- 监听 `127.0.0.1` 上的随机端口。
- 端口号写进 `runtime/interviewmate.port`。
- 随机 token 写进 `runtime/interviewmate.token`。
- 客户端读端口文件与 token，连上后先发一行 token，再发一行 JSON，读一行 JSON，断开。

token 的握手在传输层完成：服务端先读一行比对 token，通过后才交给协议层，不匹配直接断开。协议层不感知 token，`app/runtime/socket_protocol.py` 一行不改。

socket 文件在 Linux 侧承担了文件权限隔离的作用。Windows 没有 POSIX 权限位，这层隔离由 token 承担：token 存在只有当前用户可读的位置，端口只监听回环地址。

热键与命令行都走 `python -m app.cli`，由 CLI 经传输层自动带 token，外部脚本不需要自己拼握手。

### 协议层

协议层两平台共用，`app/runtime/socket_protocol.py` 一行不改。请求是单行 JSON，字段 `action` 与可选的 `request_id`；响应是单行 JSON，字段 `ok`、`error`、`payload`。`action` 的取值与处理见《2-骨架与配置》进程通信一节。

### 与 Linux 侧的逐项对照

| 项 | Linux | Windows |
|---|---|---|
| 地址 | socket 文件路径 | `127.0.0.1` 加端口 |
| 地址存放位置 | `runtime/interviewmate.sock` | `runtime/interviewmate.port` |
| 隔离手段 | 文件权限 `600` | 随机 token |
| 残留判定 | socket 文件存在但连不上 | 端口文件存在但连不上 |
| 清理 | 收尾时删 socket 文件 | 收尾时删端口文件与 token 文件 |

### 失败细分

Windows 侧把 `E_SOCKET` 分成三种情况，日志与 `status` 输出要能区分：

- 端口文件不存在：主进程没在运行。
- 端口文件存在但连不上：主进程异常退出，留了残留文件，删掉重建。
- 连上了但 token 不匹配：端口被别的程序占了，换端口重试。

## Windows 进程存活探测与退出

### 不能用 os.kill

`app/runtime/process_manager.py` 现在用 `os.kill(pid, 0)` 判断进程是否存活。CPython 官方文档对 Windows 的说明是：`os.kill` 只对 `CTRL_C_EVENT` 与 `CTRL_BREAK_EVENT` 做特殊处理，其它任何 sig 值一律调 `TerminateProcess` 把目标进程杀掉，退出码设为该 sig。

也就是说 `os.kill(pid, 0)` 在 Windows 上会把正在运行的主进程直接终止。这行代码在 Linux 上是对的，Windows 侧必须换实现。

### Windows 侧做法

用 `psutil.Process(pid).is_running()`。`psutil` 在两个平台都能用，Windows 侧新增这一个依赖。

判定链两平台一致：

1. 先探 IPC 通道。通道通，主进程在。
2. 通道不通，读 PID 文件。
3. PID 文件里的 PID 仍然存活，说明主进程活着但 IPC 坏了，按异常处理。
4. PID 文件不存在或进程已死，判定主进程没在运行。

Windows 上 PID 会被复用，所以第 3 步不能只看 PID 数字，要看进程本身。

### 退出

Windows 上 `SIGTERM` 不存在，`loop.add_signal_handler` 不可用。所以停止流程统一走 IPC 的 `stop` 动作，Ctrl+C 只作兜底。

`stop` 的收尾顺序与 Linux 侧一致：先停音频采集，取消模型任务，结束 ASR 流，关闭 HTTP 客户端，删端口文件与 token 文件，再删 PID 文件。

Windows 上 Ctrl+C 不会自动杀掉子进程，收尾要自己收干净。

## Windows 终端与全局快捷键

### 终端要求

必须用 Windows Terminal，理由有三条：

- conhost 下 Shift+回车与回车不可区分，底部输入框没法换行。
- 控制台默认代码页不是 UTF-8，中文与制表符会乱码。
- F8、F9 在部分笔记本上被 Fn 锁占用，需要在 BIOS 或厂商设置里关掉 Fn 锁。

字体要能显示中文与制表符，终端宽度小于配置的 `min_width` 时布局行为与 Linux 侧一致。

### 界面形态

界面就是终端 TUI，不做 GUI 重写，也不打包成单文件 exe。

- 现有面板、状态栏与按键绑定都按 Textual 写，改 GUI 等于重写交互。
- `torch` 与 `funasr` 打进 PyInstaller 是 GB 级体积，动态导入还容易失败。启动器用 `interviewmate.cmd` 调 `.venv` 里的解释器，升级只换代码。
- 应用层接管 Ctrl+V：Wayland 用 `wl-paste`、X11 与 XWayland 用 `xclip` 读系统剪贴板，读不到时提示终端原生粘贴键（Ctrl+Shift+V 或右键）；鼠标中键粘 PRIMARY 选区。终端自己的粘贴键照常可用，Windows Terminal 原生支持 Ctrl+V。

### 全局快捷键

Linux 侧由进程内全局热键监听接收 F8、F9，终端失焦也能用；F8 按住说话，松开停止并自动提交，F9 按一次截一张图。监听要求用户在 input 组（`sudo usermod -aG input $USER` 后重新登录），不可用时回退到终端内按键，F8 为切换语义，只在终端有焦点时有效。GNOME 自定义快捷键与进程内监听同时绑 F8、F9 会各执行一次，配好监听后移除这些绑定。

Windows 没有等价的系统级命令绑定机制，用 AutoHotkey v2 绑 F8 与 F9，动作分别是 `voice-toggle` 与 `screenshot`。Windows 不支持进程内全局监听，应用内的 F8、F9 绑定只在终端有焦点时生效，F8 是按一下开始、再按一下停止的切换语义；终端失焦时的触发继续用 AutoHotkey 调 CLI。

AutoHotkey 不在终端里开窗口跑命令，而是用 `python.exe -m app.cli hotkey <动作>` 静默调用 CLI，CLI 再经 IPC 通知主进程。调用用隐藏窗口启动，不闪黑框，也不改变当前焦点。不要用 `pythonw.exe`：没有控制台时 stderr 是 `None`，日志写到 stderr 会出问题。PowerToys Keyboard Manager 只能重映射按键，不能执行命令，不满足需求。

Windows 侧不注册全局热键、不做键盘钩子，终端获得焦点时 F8、F9 由应用内的按键绑定直接处理。AutoHotkey 脚本见本地 9.1。

## Windows 依赖与安装

### 依赖分组

| 分组 | 内容 |
|---|---|
| 公共依赖 | `textual`、`rich`、`httpx`、`pydantic`、`PyYAML`、`python-dotenv`、`numpy`、`sounddevice`、`pillow`、`funasr`、`modelscope`、`torch`、`kaldi-native-fbank` |
| Linux 专有 | `dbus-next`，Wayland 截屏必需 |
| Windows 专有 | `mss`、`soundcard`、`psutil` |

`pyproject.toml` 按这三组拆开，Linux 专有与 Windows 专有不互相安装。

### 截图后端配置取值

`providers.screenshot` 的键名两平台相同，取值不同：Windows 机器上取 `mss`，Linux 机器上取 `gnome`，`mock` 两平台都能用。取值不是 `mock` 时必须与当前平台的截图后端名一致，对不上时启动以 `E_CONFIG` 结束并说明不一致。

### torch 与 CUDA

不要依赖 PyPI 默认 wheel 带不带 CUDA。两个平台都按 pytorch.org 安装选择器给出的命令装，显式指定索引源，再装 `funasr` 等其余包。

GPU 只有一块，Ollama 常驻会占掉大部分显存，所以 Windows 侧 ASR 默认 `device: cpu`，把 8 GB 显存留给 Ollama 与 Dify 的嵌入模型。改成 CUDA 也能用，显存不足时按原有逻辑回退 CPU，指标口径两边相同。

### 已确认不用折腾的包

`kaldi-native-fbank` 有官方 Windows wheel，1.23.x 提供 `cp312-cp312-win_amd64.whl`，不需要源码编译。`funasr` 与 `modelscope` 在 Windows 上正常，`modelscope` 的缓存目录落在用户目录下，路径写法与 Linux 不同。

## Windows 目标机器

硬件与 Linux 侧一致：NVIDIA RTX 4060，显存 8 GB。

软件前提：

- Windows 10 或 Windows 11，WSL2 可用。
- WSL2 里的 Ubuntu 24.04，`systemd` 开启。
- WSL 内装 Docker Engine 与 compose 插件，不装 Docker Desktop。
- WSL 内装 Ollama，用户服务监听 `0.0.0.0:11434`，与 Ubuntu 侧同一份 unit。
- Windows 原生 Python 3.11 以上。
- Windows Terminal。
- AutoHotkey v2。
- Git for Windows 可选，用来在 Windows 侧用 bash 操作仓库。
- 依赖安装顺序：先装 CUDA 版 `torch`，再装 `funasr` 等其余包。

后台服务的安装与数据搬迁见本地 9.1。

本地模型目录放在用户目录下的 `models` 文件夹里，与 Linux 侧的 `~/models/<模型名>` 是同一个位置，路径写法按 Windows 的用户目录展开。ASR 用 `paraformer-zh-streaming` 与 `fsmn-vad` 两个目录，目录名不变。

## Windows 验收清单

### 启动与命令

- `interviewmate.cmd start` 能启动，缺 `DEEPSEEK_API_KEY` 时以退出码 3 结束并指出变量名。
- `interviewmate.cmd status` 返回六项状态，主进程不在运行时明确说明进程没在运行。
- `interviewmate.cmd hotkey screenshot` 在主进程运行时返回 `ok: true`，不在时返回 `ok: false` 与 `E_SOCKET`。
- `interviewmate.cmd devices` 能列出音频输入设备。
- 端口文件与 token 文件存在，token 不匹配时连接被拒。

### 进程

- 重复启动不产生第二个主进程。
- `stop` 能让主进程退出并清理端口文件、token 文件与 PID 文件。
- 端口文件存在但连不上时能识别为残留并重建。
- 主进程被强杀后，`status` 能判定进程没在运行，且不误杀任何进程。

### 输入

- 按 F9 能拿到截图，直接进入回答流程，不需要按回车。
- 截图失败时状态栏出现 `E_SCREENSHOT`，程序不退出。
- 截图内容不落盘，采集后 `runtime/` 与 `logs/` 里没有图片文件。
- 按 F8 能采到 16000 Hz 单声道音频，loopback 能采到系统播放的声音。
- 连续两段录音之间没有互相干扰。
- ASR 默认走 CPU 能加载；改成 CUDA 后显存不足时回退 CPU 并在日志写原因。

### 界面与快捷键

- Windows Terminal 下 Shift+回车能换行，回车能提交。
- Ctrl+V 能把系统剪贴板内容粘进输入框。
- AutoHotkey 按 F8、F9 时静默调用 CLI，不弹命令行窗口，不改变当前焦点。
- 终端缩到 80 列时仍是两栏，右侧不被挤没。

### 安全

- 日志文件里搜不到 API Key 值与 Authorization 头的值。
- 截图与录音不落盘。

## 已知差异与降级表

| 项 | 表现 | 原因 | 用户侧应对 |
|---|---|---|---|
| 截图确认 | 装上静默截图扩展后没有确认框，兜底 Portal 时要点一次 | GNOME 下的 Portal 策略一定会弹确认界面 | 装扩展，或接受这一次点击 |
| 截屏范围 | 整屏、主屏或指定屏 | 需求收敛，不做窗口与区域 | 需要窗口截图时另说 |
| 音频采样率 | 两端都按设备原生采样率开流再重采样到 16000 Hz | 设备原生采样率不统一，Windows 的 loopback 必为立体声 | 无，程序内部处理 |
| 回环音频设备 | Windows 侧采的是默认扬声器的 loopback | WASAPI 没有 monitor source 概念 | 需要指定设备时走配置 |
| 设备名 | 两边设备名不同 | 底层音频框架不同 | 用 `devices` 命令查 |
| 快捷键绑定 | Windows 要装 AutoHotkey v2 | Windows 没有系统级命令绑定 | 装 AutoHotkey，登录时自启 |
| 终端 | Windows 侧必须用 Windows Terminal | conhost 的 Shift+回车与编码限制 | 不用 conhost |
| 子进程回收 | Windows 上 Ctrl+C 不自动杀子进程 | 进程组语义不同 | 收尾流程自己处理 |
| 计时精度 | Windows 侧单调时钟分辨率约 15.6 毫秒 | 系统时钟实现 | 超时断言留余量，不按墙钟对齐音频块 |
| 后台服务 | Dify 与 Ollama 跑在 WSL2 里 | Linux 容器只能在 WSL2 里跑 | 用根目录 `stack.cmd up/down` 起停，空闲时 WSL 会自动关（原 `scripts/linux/stack.sh` 已删除，项目转 Windows-only） |
| 显存 | Ollama 常驻占显存，ASR 默认走 CPU | 单卡 8 GB | 停 Ollama 或 `wsl --shutdown` 后再切 CUDA |

## 实施顺序

第 1 步与第 2 步已在 Linux 侧 M1 完成，其余按下面的顺序做。

1. **已完成：立门面与接口定义**。`app/platform/` 的门面与接口定义立起来，Linux 侧按原有行为实现一遍，Windows 侧的五个后端先立成接口占位。
2. **已完成：收拢平台调用点**。把 Linux 侧的实现改成走门面，上层不再判断 `sys.platform`。行为不变，跑一遍现有测试。
3. **待做：传输与进程**。先做 `transport.py` 与 `process.py`：主进程启动就要监听 IPC，判重与收尾要读 PID，这两块不落地，Windows 上连 TUI 都起不来。token 握手在传输层做，协议层不动。
4. **待做：截图**。`mss`，支持整屏、主屏与指定屏，验收含 F9 直接出图与失败提示。
5. **待做：音频**。麦克风走 `sounddevice`，回环走 `soundcard` 的 WASAPI loopback，`auto` 与默认设备标记照 Linux 侧规则。
6. **待做：编译器与启动器**。`cl.exe` 或 MinGW 的 `g++`，补 `interviewmate.cmd`。
7. **待做：依赖与 CI**。`pyproject.toml` 按平台拆依赖，加 `ubuntu-latest` 与 `windows-latest` 两个 runner。
8. **待做：逐项验收**。每做完一项就补「Windows 验收清单」里对应的条目，两个平台各跑一遍。

Windows 侧不单独排里程碑。它是 Linux 侧做完之后的延伸，两边共用同一份验收标准，只有平台差异项分开列。
