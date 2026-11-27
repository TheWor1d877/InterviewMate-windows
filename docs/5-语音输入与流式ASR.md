# 语音输入与流式 ASR

按一次 F8 开始采集面试官的语音，FunASR 边说边出字，再按一次 F8 停止并把最终文本自动提交。

依赖：《3-文本输入与流式回答》。

本文件描述 Windows 侧实现。麦克风走 `sounddevice`，系统回环走 `soundcard` 的 WASAPI loopback。

## 交付物

```text
app/
├── audio/
│   ├── audio_manager.py         采集启停与状态
│   ├── audio_capture.py         sounddevice 回调
│   ├── audio_devices.py         设备枚举与来源选择
│   ├── audio_source.py          麦克风与系统回环的分离采集，系统回环走 WASAPI loopback
│   └── funasr_engine.py         FunASR 模型加载与流式推理
└── inputs/
    └── hotkey_controller.py     F8 触发
```

临时识别文字显示在状态栏下方的一行，这行由 M2 的 `app/ui/layout.py` 预留槽位渲染，不改 M1 已有的 UI 文件。

## F8 交互

```text
AUDIO OFF
    |
  按一次 F8
    |
AUDIO ON
    |
持续采集和 ASR
    |
再按一次 F8
    |
停止采集
    |
ASR FINAL
    |
自动提交题目
```

1. 程序启动后不持续监听麦克风。
2. 按一次 F8 启动语音采集，状态栏显示 AUDIO 为 on，ASR 为 ready。采集期间一直识别。
3. FunASR 处理音频流。
4. 识别过程中的部分结果只作临时显示，不提交。
5. 再按一次 F8 停止采集。
6. ASR 完成最后一次解码并合并文本。
7. 最终文字自动提交给问题处理模块，不需要再按回车。

Windows Terminal 前台时由应用内 F8 绑定处理，AutoHotkey 脚本在终端前台时不接管，同一次按键不会处理两遍；400 毫秒内的重复触发只真正切换一次。

前置条件：Windows 下由 AutoHotkey v2 全局绑定 F8，终端失焦也能用；脚本静默调用 CLI，经 IPC 通知主进程，见《3-文本输入与流式回答》。终端有焦点时由应用内按键处理，两条路都是按一下开始、再按一下停止的切换语义。

F8 有三条触发路径，最后都落到同一个控制器：

- AutoHotkey 全局绑定：终端失焦时也能触发 F8，脚本调用 `interviewmate.cmd hotkey voice-toggle`，经 IPC 送给主进程，语义是切换。
- 界面快捷键：键名取 `hotkeys.voice_toggle`，默认 F8。终端有焦点时由 Textual 接收，界面层把按键绑到 `InterviewMateApp.action_voice_toggle` 上，装配层用 `attach_voice_toggle` 把采集启停的回调挂进来，语义是切换。
- socket 动作：`action` 取 `voice-toggle`，主进程转给 `HotkeyController.handle_voice_toggle`，语义是切换。

三条路径调的是同一个 `HotkeyController`，正在采集就停止并提交最终文本，没在采集就开始采集。socket 这条路返回的字段见《2-骨架与配置》进程通信一节。

停止时的处理顺序：

```python
async def toggle_voice():
    if audio_manager.is_running:
        await audio_manager.stop()

        transcript = await asr_engine.flush()

        if transcript.strip():
            await question_manager.add_text(
                text=transcript,
                source="voice",
                auto_submit=True,
            )
        else:
            status_bar.set_asr_error("没有识别到内容")
    else:
        await asr_engine.ensure_loaded()
        await audio_manager.start()
```

最终文本为空时不发起模型请求，状态栏显示识别失败。

## 音频来源

要支持 Windows 下的多种输入：

- 麦克风。
- 系统输出回环音频。
- 用户指定的设备。

优先支持系统回环，用来识别会议软件播放的远端声音。

设备枚举：麦克风用 `sounddevice` 枚举输入设备，系统回环用 `soundcard` 的 `all_microphones(include_loopback=True)` 枚举扬声器端点的回环。`interviewmate.cmd devices` 列出可用输入源和它们的编号。回环设备的编号从 10000 起，与 PortAudio 的麦克风编号分开。

来源取值是 `microphone`、`system_monitor`、`mixed`、`auto`。`microphone`、`system_monitor`、`mixed` 按来源挑设备，同类有多台时优先系统默认设备，没有默认标记时取列表第一个。`auto` 在默认麦克风与默认回环都可用时混成一路，来源标签是 `mixed`；只有回环时用回环，只有麦克风时用麦克风；两个都不通时报错码 `E_AUDIO_DEVICE`。

默认设备在设备枚举时动态解析：麦克风按 `sounddevice` 的默认输入索引标记；回环按 `soundcard` 的默认扬声器端点 ID 标记，ID 相同的回环设备就是默认回环。解析失败时所有设备按非默认处理，不影响枚举，选设备时退回列表第一个。配置里不写具体设备名或索引，换设备后重新枚举即生效。

麦克风与系统音频分开采集时，各自保留来源标签，避免把用户自己说的话当成面试官的问题。混成一路的音频允许用 VAD 切分，也可以接说话人区分，但说话人识别不作为正常工作的前提。

## 采集参数

采样率 16000 Hz，单声道，块长 200 毫秒。

Windows 设备原生采样率多为 48000 Hz，WASAPI loopback 端点通常是立体声，按原生格式采集后再下混并重采样到 16000 Hz。回环一路走 WASAPI 共享模式，`soundcard` 在库内部把端点混音格式转换到请求采样率，端点原生采样率拿不到时按 48000 Hz 开流。

开流声道数上限为 2，超过按 2 开流后下混单声道。部分虚拟声卡驱动会报出大量输入声道，PortAudio 按这个数开流会崩溃，所以设备原生声道数再大都只按 2 声道开流。

200 毫秒是采集回调每次读取的量，不是送进模型的块长。送到模型之前要先按 9600 点拼成 600 毫秒一块，块长关系见《接口契约》的「FunASR 音频块参数」。

音频块进入一个有界异步队列，队列满时丢弃最旧的块，丢弃计数写日志。音频回调线程只做读取和入队，不发网络请求。

## FunASR 模型

模型用 `AutoModel` 加载，参数见《接口契约》的「FunASR 流式推理调用」。

ASR 与 VAD 用两个独立的 `AutoModel` 实例，ASR 实例不传 `vad_model`：

```python
from pathlib import Path

MODEL_DIR = Path.home() / "models"

asr_model = AutoModel(
    model="paraformer-zh-streaming",
    model_path=str(MODEL_DIR / "paraformer-zh-streaming"),
    device="cuda",
)

vad_model = AutoModel(
    model="fsmn-vad",
    model_path=str(MODEL_DIR / "fsmn-vad"),
    device="cuda",
)
```

本地模型目录放在 `~/models/<模型名>`，用 `model_path` 传，不能用 `model_dir`：funasr 只在 kwargs 里存在 `model_path` 时才跳过 hub 下载，`model_dir` 不参与该判断，权重会被重新拉进 modelscope 缓存。

Windows 上装 `torch` 要指定 CUDA 索引源：`pip install torch --index-url https://download.pytorch.org/whl/cu124`。PyPI 默认 wheel 是 CPU 版。

两个实例都只初始化一次，整个进程生命周期内复用同一个实例。VAD 实例只在配置项 `vad_enabled` 为真时创建。

设备选择 `device` 取 `auto`、`cuda`、`cpu`。`auto` 先试 CUDA，显存不足或初始化失败时回退 CPU。GPU 初始化失败不能挡住程序启动，回退后 ASR 状态标成 ready 并在日志里写回退原因。两边的 GPU 型号一致，回退行为与指标口径相同。

## 流式推理

每条语音流的做法：

1. 新建一个空字典作为 `cache`。
2. 把采集到的 200 毫秒块攒够 600 毫秒，取 9600 点的一维 float32 波形。
3. 调用 `asr_model.generate`，传入同一个 `cache` 字典，`is_final` 置 false。
4. 返回文本按顺序追加到当前句子的缓冲。
5. 停止采集时，用 `is_final` 为 true 的那次调用冲刷剩余缓存，把最后的片段追加进缓冲。
6. 冲刷后丢弃这个 `cache` 字典，下一条语音流重新建一个。

推理调用的参数组合：

```python
result = asr_model.generate(
    input=chunk,
    cache=cache,
    is_final=is_final,
    chunk_size=[0, 10, 5],
    encoder_chunk_look_back=4,
    decoder_chunk_look_back=1,
    batch_size=1,
    disable_pbar=True,
)
```

几条容易踩的点：

- `cache` 在同一条语音流的所有块之间共用，语音流之间不共用。跨流共用会让上一条语音的状态串到下一条。
- `batch_size` 固定为 1，一次调用只传一段波形。传多段会报错。
- Paraformer 流式在 finalize 之后会重置内部缓存，所以冲刷完要把字典丢掉。
- `disable_pbar=True` 关掉 funasr 的进度条，不关会往终端刷行。
- 返回的文本只覆盖当前调用解码的片段，不保证是完整句子。整句由多次结果按顺序拼接得到。

## 语音端点检测

VAD 用 `fsmn-vad`，配置项 `vad_enabled` 控制开关。

VAD 输出的语音区间用来判断一句话是否结束。区间边界由模型给出，可能不完整，客户端按收到的区间处理，不自行修正边界。

语音段落在模型输出里已经带有标点时，不再额外接标点模型。

## 部分结果与最终结果

| 结果 | 来源 | 动作 |
|---|---|---|
| 部分结果 | `generate` 返回但本段音频还没结束 | 只更新状态栏的临时显示 |
| 最终结果 | 停止采集后 `is_final` 为 true 的那次调用 | 合并成整句并自动提交 |

一段短句识别完不能触发模型请求。只有最终结果才提交。

## 状态显示

状态栏的 ASR 取值见《2-骨架与配置》。加载模型时是 loading，就绪后是 ready，加载失败是 error。

语音采集开着时 AUDIO 显示 on，临时识别文字显示在顶部状态栏下方的一行，不占用左右两栏。

## 性能

记录 ASR 相关耗时：识别处理耗时与停止到最终文本的耗时，指标名见《2-骨架与配置》。

减少耗时的做法：

- 启动时预加载模型。
- 复用同一个模型实例。
- 音频块走异步队列。
- 用 VAD 切分语音段。
- 限制音频缓存长度，避免长会议把内存吃满。
- 每块音频只调一次模型，不做重复推理。

## 实测结论

- FunASR 的流式解码比离线整段识别少一个尾字。这是 funasr 流式解码本身的差异，不是本实现的偏差。
- 模型在 CUDA 上首次加载约 6 秒。加载在工作线程里做，不占住事件循环；模型就绪前按 F8，会等到加载完才开始采集。

## 验收

- 按一次 F8 开始采集，状态栏 AUDIO 变成 on，再按一次停止。
- 停止后最终文本自动提交，不需要按回车。
- 部分结果不触发模型请求，最终结果只触发一次。
- 说话时状态栏持续更新临时文字，停止后临时文字变成正式题目。
- 停止采集时没有识别到内容，状态栏提示识别失败，不发起模型请求。
- 连续两段录音之间没有互相干扰，第二段的识别结果里没有第一段的残留文字。
- 设备不可用时 `interviewmate.cmd devices` 列出可用设备，报错码 `E_AUDIO_DEVICE`。
- ASR 模型加载失败时程序照常启动，状态栏 ASR 显示 error。
