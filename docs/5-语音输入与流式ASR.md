# 语音输入与流式 ASR

按住 F8 采集面试官的语音，FunASR 边说边出字；松开 F8 停止后，整段音频由 SenseVoiceSmall 离线重识别、再经拼音级热词纠正产出最终文本并自动提交。流式部分结果只作实时预览。进入项目模式后 F8 的语义变成截取面试官问题，见下面的项目模式一节。

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
│   ├── funasr_engine.py         FunASR 模型加载与流式推理，缓存整段录音供重识别
│   ├── sensevoice_engine.py     SenseVoiceSmall 整段重识别 + 热词纠正
│   └── hotword_corrector.py     拼音级热词纠正，纯逻辑无模型依赖
└── inputs/
    └── hotkey_controller.py     F8 触发
```

临时识别文字显示在状态栏下方的一行，这行由 M2 的 `app/ui/layout.py` 预留槽位渲染，不改 M1 已有的 UI 文件。

## F8 交互

```text
AUDIO OFF
    |
  按下 F8（AHK 发 voice-start）
    |
AUDIO ON
    |
持续采集和 ASR
    |
  松开 F8（AHK 发 voice-stop）
    |
停止采集
    |
ASR FINAL
    |
自动提交题目
```

1. 程序启动后不持续监听麦克风。
2. 按下 F8 开始语音采集，状态栏显示 AUDIO 为 on，ASR 为 ready。采集期间一直识别。
3. FunASR 处理音频流。
4. 识别过程中的部分结果只作临时显示，不提交。
5. 松开 F8 停止采集。
6. ASR 完成最后一次解码并合并文本。
7. 最终文字自动提交给问题处理模块，不需要再按回车。

Windows 下 F8 由 AutoHotkey v2 全局接管，脚本常驻运行、无条件生效，终端失焦也能用；按下发 `voice-start`，松开发 `voice-stop`，脚本静默调用 CLI，经 IPC 通知主进程，见《3-文本输入与流式回答》。界面被 AHK 接管后不再绑定 F8，同一次按键不会处理两遍。按住路径不去抖，重复触发由采集器的 `is_running` 防重入：已在采集时 `voice-start` 忽略，没在采集时 `voice-stop` 空操作。

socket 的动作落到主进程两条路径：`voice-start` 调 `MainState.start_voice_capture`，`voice-stop` 调 `MainState.stop_voice_capture`，都转给 `HotkeyController`；向后兼容的 `voice-toggle` 走 `MainState.toggle_voice_capture`（400ms 去抖），CLI 手动调用仍可用。三个动作共用同一段结果处理。触发类动作先回 `accepted` 回执再后台执行，返回字段见《2-骨架与配置》进程通信一节。

F8 是双语义键，按主进程状态分派：普通模式走上面这条语音链路；项目模式双流转写运行中，按下/松开变成面试官问题截取窗口的开与关，不走普通采集链路，见下面的项目模式一节。分派在 `MainState.start_voice_capture` / `stop_voice_capture` 入口处判断，全局热键与 socket 两条触发路共用同一份实现。

松开后的处理顺序：

```python
async def stop_voice():
    if audio_manager.is_running:
        await audio_manager.stop()
        # stop 内部：冲刷流式引擎 → 取整段录音 → SenseVoice 重识别
        # → 热词纠正 → 最终文本；重识别失败退回流式文本

        if transcript.strip():
            await question_manager.add_text(
                text=transcript,
                source="voice",
                auto_submit=True,
            )
        else:
            status_bar.set_asr_error("没有识别到内容")
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

回环采集涉及 COM：`soundcard` 只在导入它的线程上初始化过 COM（单线程套间 STA），而枚举与采集都在独立线程里跑，线程里必须先按 STA 初始化 COM、离开时配对释放；不初始化时回环设备一枚都枚不到。

开流声道数上限为 2，超过按 2 开流后下混单声道。部分虚拟声卡驱动会报出大量输入声道，PortAudio 按这个数开流会崩溃，所以设备原生声道数再大都只按 2 声道开流。

采集回调每次读 200 毫秒，送进模型的块长也是 200 毫秒：块攒满 3200 点就送进模型，不足一块时先攒着。块长关系见《接口契约》的「FunASR 音频块参数」。

音频块进入一个有界异步队列，队列满时丢弃最旧的块，丢弃计数写日志。音频回调线程只做读取和入队，不发网络请求。

## FunASR 模型

模型用 `AutoModel` 加载，参数见《接口契约》的「FunASR 流式推理调用」。

ASR 与 VAD 用两个独立的 `AutoModel` 实例，ASR 实例不传 `vad_model`：

```python
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("INTERVIEWMATE_MODELS") or REPO_ROOT / "models")

asr_model = AutoModel(
    model="ParaformerStreaming",
    model_path=str(MODEL_DIR / "paraformer-zh-streaming"),
    device="cuda",
)

vad_model = AutoModel(
    model="FsmnVADStreaming",
    model_path=str(MODEL_DIR / "fsmn-vad"),
    device="cuda",
)
```

本地模型目录默认在仓库根 `models/<目录名>`，可用环境变量 `INTERVIEWMATE_MODELS` 覆盖。用 `model_path` 传，不能用 `model_dir`：funasr 只在 kwargs 里存在 `model_path` 时才跳过 hub 下载，`model_dir` 不参与该判断，权重会被重新拉进 modelscope 缓存。

`model` 参数填的是 FunASR 注册表名，不是本地目录名：目录名是 ModelScope 仓库名，注册表名才是 funasr 注册表里登记的类名，把目录名原样传给 `AutoModel(model=...)` 会报 `model '...' is not registered`。两组名字的对应关系：

| 本地目录名 | 注册表名 |
|---|---|
| `paraformer-zh-streaming` | `ParaformerStreaming` |
| `fsmn-vad` | `FsmnVADStreaming` |

配置 `asr.model` 两种写法都认：写目录名是旧默认值，兼容保留；写注册表名直接用。都不认识时按自定义目录原样处理。

Windows 上装 `torch` 要指定 CUDA 索引源：`pip install torch --index-url https://download.pytorch.org/whl/cu124`。PyPI 默认 wheel 是 CPU 版。

两个实例都只初始化一次，整个进程生命周期内复用同一个实例。VAD 实例只在配置项 `vad_enabled` 为真时创建。

设备选择 `device` 取 `auto`、`cuda`、`cpu`。`auto` 先试 CUDA，显存不足或初始化失败时回退 CPU。GPU 初始化失败不能挡住程序启动，回退后 ASR 状态标成 ready 并在日志里写回退原因。两边的 GPU 型号一致，回退行为与指标口径相同。

## 流式推理

每条语音流的做法：

1. 新建一个空字典作为 `cache`。
2. 把采集到的块攒满 3200 点（200 毫秒），取一维 float32 波形，不足一块时先攒着。
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
| 最终结果 | 停止采集后由整段音频重识别 + 热词纠正产出 | 合并成整句并自动提交 |

一段短句识别完不能触发模型请求。只有最终结果才提交。流式 partial 预览保留不动（UI 体验不变），但提交的最终文本不再来自流式解码，见下面的重识别与热词纠正一节。

## 重识别与热词纠正

最终文本的产出流程：流式识别照常跑（保留实时预览与按句预览），同时录音期间在 `funasr_engine` 层为当前录音会话缓存整段 PCM（16 kHz，上限与语音流缓存同源 300 秒，超限丢最早）；松开 F8 后把整段录音交给 SenseVoiceSmall 重新识别一遍，输出文本清掉 rich text 标签（`<|zh|>`、`<|NEUTRAL|>`、`<|Speech|>` 等，正则 `<\|[^|]*\|>`），再过拼音级热词纠正，才是提交用的最终文本。

### SenseVoiceSmall 重识别引擎

`app/audio/sensevoice_engine.py`，`funasr AutoModel` 加载本地 `models/SenseVoiceSmall`（`use_itn=True`、`language=zh`，VAD 用 `fsmn-vad`，`max_single_segment_time=30000`）。`model` 直接传本地目录路径，funasr 读目录内 `config.yaml` 确定注册名；模型目录沿用 `models/` 约定，`INTERVIEWMATE_MODELS` 可覆盖。

- 整段识别无内部状态（每次推理新建 `cache`），一个进程共享一个实例，语音链路与项目模式共用，识别调用在锁内串行执行；
- 设备取 `auto` 时先试 CUDA，失败回退 CPU 并记日志；与流式引擎同时驻留显存约 3.5 GB，8 GB 显存可容纳；
- 主进程启动预加载：流式引擎加载成功后顺带预热重识别模型，失败只记 WARNING；
- 重识别失败或结果为空时退回流式文本（同样过热词纠正），不阻断提交；引擎加载失败只告警降级，不阻断主进程；
- 识别异常抛 `E_ASR_DECODE`、加载异常抛 `E_ASR_LOAD`，都不吞异常。

### 拼音级热词纠正

`app/audio/hotword_corrector.py`，纯逻辑无模型依赖。算法：pypinyin 把识别文本与热词各自转成拼音序列，在识别文本的连续汉字段里滑窗匹配，编辑距离不超过阈值就替换成热词写法。阈值 `thr = int(热词拼音长度 × 0.35)`：2 字词仅拼音完全相同（距离 0）才替换，3 字词允许距离 1，越短越严，避免近音短词误伤。保护区：片段拼音与文字都与热词完全相同的不参与匹配，防止正确热词的子串被「补字」误替换；同音错字（文字不同）不受保护。候选按距离小、位置靠前、窗口长排序后贪心取不重叠片段。该算法经实验验证：纯汉字 CER 0.0023、零误伤。

热词表 `config/hotwords.txt`，每行一词，`#` 开头是注释行，空行忽略；文件不存在或为空时热词表为空，纠正直接原样跳过。路径可用环境变量 `INTERVIEWMATE_HOTWORDS` 覆盖。表里的纯英文热词（HPACK、QUIC 等）不参与拼音匹配，英文写法靠模型自己识别。用户自行增删维护，保存为 UTF-8 编码。

### asr-replay

`asr-replay` 子命令的输出与产品链路一致：读 wav、下混、重采样到 16 kHz 后整段交给 SenseVoice 重识别加热词纠正，打印最终文本。基线音频 `tests/assets/asr-test-question.wav`（「解释一下什么是快速排序」）识别逐字正确。`--block-ms` 对整段识别无效，保留参数只为命令行兼容。

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

## 项目模式

面试的另一种用法：面试官在另一个窗口（视频会议、共享的题面页面）说话，考生自己也在讲话。项目模式开两条独立的转写流，把双方的话分角色记下来，面试官提问时截取一段直接触发回答。

### 双流与角色分工

- 麦克风一路记「我」（考生自己），固定角色标注是 `[我]`。
- 系统回环一路记「面试官」，固定角色标注是 `[面试官]`。回环优先取默认扬声器的回环端点；用户耳机是默认扬声器时端点编号会漂移，设环境变量 `INTERVIEWMATE_LOOPBACK_DEVICE` 为确定的回环端点编号（`interviewmate.cmd devices` 输出里标注「系统回环」的那个）可覆盖默认选择。
- 两路各自一个独立的 FunASR 引擎实例：流式模型内部有 cache 状态，两路混喂同一实例会把识别状态串流。模型目录与主链路一致（仓库根 `models/`，可用 `INTERVIEWMATE_MODELS` 覆盖）。
- 角色只按物理来源区分，不做说话人分离、不做回声对冲：自己说话被回环拾进去串了角色，由人工按 F11 重开一轮处理。
- 一路故障不拖垮另一路：引擎加载失败、开流失败、解码失败都记日志后该路跳过或继续。两路都起不来时启动失败，界面不切换模式。

### 按句落库与对话日志

- 按句落库：句末标点（。！？；… 等）、单句字数上限（60 字）、单句时长上限（20 秒）任一触发，就把当前句写进对话日志；部分结果只经事件总线发 `dialogue.partial` 给界面的双路临时区，完整句子发 `dialogue.sentence`。
- 对话日志（`app/dialogue/dialogue_log.py`）每个会话一个 md 文件，写在 `logs/dialogues/{session_id}.md`，一行一条，会话 ID 是本地时间 `YYYYMMDD-HHMMSS`，同一秒内重开时加 `-2`、`-3` 序号。行格式：

```text
{HH:MM:SS} [我] 这句话是我说的
{HH:MM:SS} [面试官] 这句话是面试官说的
{HH:MM:SS} **[问题]** 被截取窗口标记的问题句
```

- `clear` 只封存不删除：当前会话文件原样保留，另开一个新会话文件，返回新会话 ID。内存里留一份按写入顺序排的行缓存：全文给界面的上下文视图（F10），`recent_text()` 按时间序从尾部取最近 8000 字符喂给请求拼装，在行边界截断、不重排。

### 键位

项目模式下键位整体切换（状态栏 MODE/SESSION、提示栏键位、双路临时区显隐一起变）：

| 键 | 普通模式 | 项目模式 |
|---|---|---|
| F7 | 截图（AHK 接管后由 socket 处理） | 截图 |
| F8 | 按住说话 | 按住截取面试官问题：按下开窗口，松开把窗口内的面试官语句标成问题并提交 |
| F9 | — | 进入/退出项目模式（`project-mode-toggle`，AHK 发起） |
| F10 | 重新分析 | 上下文视图（右栏临时切成对话日志全文，再按一次切回） |
| F11 | — | 清空上下文：封存当前对话日志并重开一轮（`context-clear`） |

进入项目模式时把 F10 的绑定切到上下文视图，退出时恢复成重新分析。F11 的清空是同步动作：主进程先封存日志重开会话，再让界面刷新双路临时区与上下文视图。

### 截取窗口与问题提交

按下 F8 开始截取窗口，窗口期间落库的面试官句子先收集，松开时逐句在日志里标记成 `**[问题]**` 行并拼接成问题文本。问题文本非空时放后台走问题提交链，语义等同用户手动输入了这个问题：题面显示到右栏顶部，问题管理器发一次流式请求。窗口里没有面试官语句时不提交，返回的 `text` 是空串。普通模式的语音采集状态字段（AUDIO/ASR）不因截取窗口变化。

### 请求拼装

项目模式提交问题时，发给模型的 user 消息按这个顺序拼装（system 不变）：

1. 对话日志上下文（`recent_text()` 的最近一段，含角色标注，追加在 user 消息靠前位置）；
2. 本轮面试官问题；
3. 本轮 RAG 检索内容（题型判定的 `project_scope` 是 `mixed` 或 `full` 时检索，见《3-文本输入与流式回答》题型判定一节）；
4. 两句附加指令：检索到内容时跟在检索内容之后，没检索到内容（含 Dify 离线降级）时单独成段跟在问题后面。

```text
问题：对话上下文：
15:30:01 [我] 先介绍一下我做的项目
15:30:05 [面试官] 那讲讲你的缓存设计
15:30:10 **[问题]** 那讲讲你的缓存设计

问题：那讲讲你的缓存设计

背景上下文：
…（Dify 检索片段）

检索出来的内容不一定对问题有用，请围绕问题进行回答。面试官的问题来自语音转写，可能不精准，先理解他的意思，再以精准的表述回答。
```

普通模式（非项目模式）不注入日志、不附加指令，请求拼装与从前完全一致。Dify 没配置或检索失败时按《6-Dify检索与项目知识》故障降级一节处理：检索超时（`rag.timeout=8` 秒）或失败后按没取到资料继续，请求照常发出、回答照常生成，项目模式的提交不会被检索阻塞。

## 验收

- 按住 F8 开始采集，状态栏 AUDIO 变成 on，松开停止。
- 停止后最终文本自动提交，不需要按回车。
- 部分结果不触发模型请求，最终结果只触发一次。
- 说话时状态栏持续更新临时文字，停止后临时文字变成正式题目。
- 停止采集时没有识别到内容，状态栏提示识别失败，不发起模型请求。
- 连续两段录音之间没有互相干扰，第二段的识别结果里没有第一段的残留文字。
- 设备不可用时 `interviewmate.cmd devices` 列出可用设备，报错码 `E_AUDIO_DEVICE`。
- ASR 模型加载失败时程序照常启动，状态栏 ASR 显示 error。
