"""音频采集层。

设备枚举与音频块采集放在这里，见《2-骨架与配置》依赖一节。

分工：

- :mod:`app.audio.audio_devices`：设备枚举与来源选择；
- :mod:`app.audio.audio_capture`：单路的流式采集与有界块队列；
- :mod:`app.audio.audio_source`：麦克风与系统回环的分离采集与混合采集；
- :mod:`app.audio.funasr_engine`：FunASR 模型加载与流式推理；
- :mod:`app.audio.audio_manager`：采集启停、状态与部分结果、最终文本的分送。

采集参数见《5-语音输入与流式ASR》「采集参数」一节。
"""

from app.audio.audio_capture import (
    AudioBlock,
    AudioCapture,
    BoundedAudioQueue,
)
from app.audio.audio_devices import (
    AUDIO_SOURCES,
    AudioSource,
    SourceSelection,
    list_audio_devices,
    resolve_source,
    select_source,
    split_devices,
)
from app.audio.audio_manager import (
    AUDIO_STATE_OFF,
    AUDIO_STATE_ON,
    RESULT_EMPTY,
    RESULT_ERROR,
    RESULT_STARTED,
    RESULT_STOPPED,
    AsrEngine,
    AsrEngineFactory,
    AudioManager,
    Capture,
    CaptureFactory,
    TextCallback,
)
from app.audio.audio_source import (
    MixedCapture,
    SeparatedCapture,
    create_capture,
)
from app.audio.funasr_engine import (
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEFAULT_ASR_MODEL_NAME,
    DEFAULT_VAD_MODEL_NAME,
    MODEL_CHUNK_SAMPLES,
    STATE_ERROR,
    STATE_LOADING,
    STATE_READY,
    FunASREngine,
)

__all__ = [
    "AudioBlock",
    "AudioCapture",
    "BoundedAudioQueue",
    "AUDIO_SOURCES",
    "AudioSource",
    "SourceSelection",
    "list_audio_devices",
    "resolve_source",
    "select_source",
    "split_devices",
    "MixedCapture",
    "SeparatedCapture",
    "create_capture",
    "AUDIO_STATE_ON",
    "AUDIO_STATE_OFF",
    "RESULT_STARTED",
    "RESULT_STOPPED",
    "RESULT_EMPTY",
    "RESULT_ERROR",
    "AsrEngine",
    "AsrEngineFactory",
    "AudioManager",
    "Capture",
    "CaptureFactory",
    "TextCallback",
    "FunASREngine",
    "DEFAULT_ASR_MODEL_NAME",
    "DEFAULT_VAD_MODEL_NAME",
    "MODEL_CHUNK_SAMPLES",
    "DEVICE_AUTO",
    "DEVICE_CUDA",
    "DEVICE_CPU",
    "STATE_LOADING",
    "STATE_READY",
    "STATE_ERROR",
]
