"""Windows 侧的音频后端：sounddevice 麦克风 + soundcard 的 WASAPI loopback。

麦克风枚举与采集用 ``sounddevice``，因为 PortAudio 能直接给出输入设备的索引、
原生采样率与输入声道数。系统回环用 ``soundcard`` 的 WASAPI loopback：
PortAudio 不暴露 loopback 端点，``soundcard`` 的 ``all_microphones``
传 ``include_loopback=True`` 时会把每个扬声器端点包成一个可录音的输入设备。

两路的设备编号分开，避免与 PortAudio 的索引撞车：

- 麦克风沿用 ``sounddevice`` 给出的真实索引；
- 回环设备的索引从 :data:`LOOPBACK_INDEX_BASE` 起按顺序加，真实设备 ID 记在
  :attr:`WasapiAudioBackend._loopback_ids` 里，开流时按 ID 找回去。

采集链路与《5-语音输入与流式ASR》「采集参数」一节一致：按设备原生采样率与
原生声道数开流，回调里下混成单声道，再重采样到 16000 Hz，按
``block_duration_ms`` 出块。开流声道数最多两声道，更多声道只下混不用：
部分虚拟声卡驱动会报出大量输入声道，PortAudio 按这个数开流会崩溃。

WASAPI loopback 走共享模式，``soundcard`` 以共享模式打开端点并在库内部完成
到请求采样率的转换，端点混音格式的原生采样率在 ``soundcard`` 里拿不到，
回环一路按 :data:`SHARED_MODE_RATE`（48000 Hz）开流。

设备枚举与开流都是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
超时报 ``E_AUDIO_DEVICE``，见《8-集成与验收》「异步任务」与「超时管理」两节。

``soundcard`` 只在方法体内按需导入，模块顶层不 import 它；``sounddevice``
同样是按需导入，这样没有声卡的环境也能导入本模块。
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from app.config import current_config
from app.core.exceptions import AudioDeviceError
from app.platform.types import AudioBlockCallback, AudioDevice, AudioStream
from app.utils.logger import log_event

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "WasapiAudioBackend",
    "DEFAULT_INIT_TIMEOUT",
    "LOOPBACK_INDEX_BASE",
    "SHARED_MODE_RATE",
]


# 回环设备编号的起点：与 PortAudio 的麦克风索引分开，避免同一个数字指两台设备
LOOPBACK_INDEX_BASE: int = 10000

# WASAPI 共享模式的常见混音采样率：soundcard 拿不到端点原生采样率时按它开流
SHARED_MODE_RATE: float = 48000.0

# 回环设备在设备表里的显示后缀，界面上一眼分得清是扬声器的回环
_LOOPBACK_NAME_SUFFIX: str = "（回环）"

# 开流声道数上限：设备报再多声道也只按两声道开，下混成单声道
_MAX_STREAM_CHANNELS: int = 2

# 音频设备初始化超时的默认秒数：进程里没加载过配置时用它，
# 见《8-集成与验收》「超时管理」一节
DEFAULT_INIT_TIMEOUT: float = 10.0

_T = TypeVar("_T")


def _run_with_timeout(func: Callable[[], _T], timeout: float, what: str) -> _T:
    """在超时时间内跑一个阻塞调用。

    阻塞调用放到线程里跑，调用方只等结果，见《8-集成与验收》「异步任务」一节。
    超时时线程停不下来（Python 杀不掉线程），这里把它设成守护线程，
    进程退出时不等它。

    :param func: 要跑的阻塞调用。
    :param timeout: 超时秒数。
    :param what: 这一步的中文说法，进错误文案。
    :returns: 阻塞调用的返回值。
    :raises AudioDeviceError: 超时；调用本身的异常原样抛出去。
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        """在线程里跑阻塞调用，结果放进 box。"""
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - 原样抛回调用方
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True, name="im-audio-init")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AudioDeviceError(f"{what}超时（{timeout:g} 秒）")
    error = box.get("error")
    if error is not None:
        raise error
    return box["value"]  # type: ignore[return-value]


def _downmix_to_mono(indata: "np.ndarray") -> "np.ndarray":
    """把一段采集数据下混成单声道。

    多声道按各声道求平均，单声道直接用。

    :param indata: 回调拿到的数据，形状是 ``(采样点, 声道数)``。
    :returns: 新的一维 float32 数组，入参不被修改。
    """
    import numpy as np

    samples = np.asarray(indata, dtype="float32")
    if samples.ndim == 1:
        return np.array(samples, dtype="float32", copy=True)
    if samples.ndim != 2:
        return np.array(samples.reshape(-1), dtype="float32", copy=True)
    if samples.shape[1] == 0:
        return np.zeros(0, dtype="float32")
    if samples.shape[1] == 1:
        return np.array(samples[:, 0], dtype="float32", copy=True)
    return np.asarray(samples.mean(axis=1), dtype="float32")


def _resample_to(samples: "np.ndarray", src_rate: float, dst_rate: int) -> "np.ndarray":
    """把一段波形按线性插值重采样到目标采样率。

    :param samples: 一维波形。
    :param src_rate: 原采样率。
    :param dst_rate: 目标采样率。
    :returns: 新的一维 float32 数组，入参不被修改。
    """
    import numpy as np

    source = np.asarray(samples, dtype="float32").reshape(-1)
    if source.size == 0:
        return np.zeros(0, dtype="float32")
    if float(src_rate) == float(dst_rate):
        # 采样率一样也要复制一份：PortAudio 会复用回调里的缓冲区
        return np.array(source, dtype="float32", copy=True)
    out_len = int(round(source.size * float(dst_rate) / float(src_rate)))
    if out_len <= 0:
        return np.zeros(0, dtype="float32")
    if out_len == 1:
        return np.full(1, float(source[0]), dtype="float32")
    # 输入下标按比例铺开，两端对齐到首尾采样点
    positions = np.arange(out_len, dtype="float64") * (source.size - 1) / (out_len - 1)
    resampled = np.interp(positions, np.arange(source.size, dtype="float64"), source)
    return np.asarray(resampled, dtype="float32")


def _as_device_index(value: Any) -> int | None:
    """把默认设备取值转成设备索引。

    ``sounddevice.default.device`` 里 -1 表示未知，None 或转换失败的
    取值也按未知处理。

    :param value: 默认设备取值。
    :returns: 非负索引；未知时返回 None。
    """
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _device_channels(device: Any) -> int:
    """取 soundcard 设备报出的声道数，取不到时按单声道处理。

    :param device: soundcard 的麦克风设备。
    :returns: 声道数，至少是 1。
    """
    try:
        channels = int(getattr(device, "channels", 1) or 1)
    except Exception:
        return 1
    return max(channels, 1)


def _loopback_native_rate(device: Any) -> float:
    """取回环设备开流用的采样率。

    ``soundcard`` 的设备对象没有原生采样率属性，WASAPI 共享模式由库内部
    按端点混音格式转换到请求采样率，所以拿不到时按
    :data:`SHARED_MODE_RATE` 开流。

    :param device: soundcard 的麦克风设备。
    :returns: 采样率，正好是正数。
    """
    try:
        rate = float(getattr(device, "samplerate", 0.0) or 0.0)
    except Exception:
        return SHARED_MODE_RATE
    return rate if rate > 0 else SHARED_MODE_RATE


def _loopback_index(position: int) -> int:
    """回环设备在设备表里的编号。

    :param position: 回环设备在本次枚举里的顺序，0 起数。
    :returns: 编号。
    """
    return LOOPBACK_INDEX_BASE + position


def _is_loopback_index(index: int) -> bool:
    """这个编号是不是回环设备。

    :param index: 设备编号。
    :returns: 是回环设备返回 True。
    """
    return index >= LOOPBACK_INDEX_BASE


class _SoundDeviceStream:
    """已经打开的 sounddevice 输入流。

    :meth:`stop` 先停再关，关过的流再停一次不报错。
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._active = True

    @property
    def is_active(self) -> bool:
        """流是不是还开着。"""
        return self._active

    def stop(self) -> None:
        """关掉流。"""
        if not self._active:
            return
        self._active = False
        try:
            self._stream.stop()
        except Exception as exc:
            log_event(
                "audio.stream.stop_failed",
                f"停止音频流失败：{exc}",
                level="WARNING",
            )
        try:
            self._stream.close()
        except Exception as exc:
            log_event(
                "audio.stream.close_failed",
                f"关闭音频流失败：{exc}",
                level="WARNING",
            )


class _WasapiLoopbackStream:
    """一路已经打开的 WASAPI loopback 采集流。

    ``soundcard`` 的录音是阻塞式读取，采集放在独立线程里循环读块：
    线程内先打开 recorder，之后每次读 ``block_frames`` 帧，下混并重采样后
    交给回调。回调线程只做读取、下混、重采样与入队，不发网络请求。

    :param recorder_factory: 打开 recorder 的工厂，实际调用的是
        ``microphone.recorder(...)``，它是上下文管理器。
    :param block_frames: 每次读取的帧数，按回环开流采样率算。
    :param native_rate: 回环开流采样率。
    :param sample_rate: 目标采样率。
    :param on_block: 收到一块音频时调的回调。
    """

    def __init__(
        self,
        *,
        recorder_factory: Callable[[], Any],
        block_frames: int,
        native_rate: float,
        sample_rate: int,
        on_block: AudioBlockCallback,
    ) -> None:
        self._recorder_factory = recorder_factory
        self._block_frames = block_frames
        self._native_rate = native_rate
        self._sample_rate = sample_rate
        self._on_block = on_block
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._recorder: Any = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="im-audio-loopback",
        )

    @property
    def is_active(self) -> bool:
        """流是不是还开着。"""
        return self._thread.is_alive() and not self._stop.is_set()

    def start(self, timeout: float) -> None:
        """启动采集线程并等 recorder 打开。

        :param timeout: 等 recorder 打开的超时秒数。
        :raises AudioDeviceError: 打开超时，或 recorder 打开失败。
        """
        self._thread.start()
        if not self._ready.wait(timeout):
            self._stop.set()
            raise AudioDeviceError(f"开启音频流超时（{timeout:g} 秒）")
        if self._error is not None:
            raise AudioDeviceError(f"开启音频流失败：{self._error}")

    def stop(self) -> None:
        """停掉采集线程并关掉 recorder。已经关掉的流再停一次不报错。"""
        if self._stop.is_set() and not self._thread.is_alive():
            return
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(2.0)

    def _run(self) -> None:
        """采集线程：打开 recorder，循环读块，下混重采样后交给回调。"""
        try:
            with self._recorder_factory() as recorder:
                self._recorder = recorder
                self._ready.set()
                while not self._stop.is_set():
                    data = recorder.record(numframes=self._block_frames)
                    samples = _downmix_to_mono(data)
                    samples = _resample_to(samples, self._native_rate, self._sample_rate)
                    try:
                        self._on_block(samples)
                    except Exception as exc:
                        log_event(
                            "audio.stream.callback_failed",
                            f"音频块回调失败：{exc}",
                            level="WARNING",
                            error=str(exc),
                        )
        except BaseException as exc:  # noqa: BLE001 - 打开失败要回传给调用方
            self._error = exc
            self._ready.set()
        finally:
            self._recorder = None


class WasapiAudioBackend:
    """基于 sounddevice 与 WASAPI loopback 的音频设备枚举与采集后端。

    :param init_timeout: 音频设备初始化超时秒数，None 时取进程里加载好的
        配置里的 ``audio.init_timeout``，见《8-集成与验收》「超时管理」一节。
    """

    # 后端名
    name: str = "wasapi"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("sounddevice", "soundcard", "numpy")

    def __init__(self, init_timeout: float | None = None) -> None:
        self._init_timeout = init_timeout
        self._loopback_lock = threading.Lock()
        self._loopback_ids: dict[int, str] = {}

    @property
    def init_timeout(self) -> float:
        """音频设备初始化超时秒数。

        构造时没给就从进程里加载好的配置里取 ``audio.init_timeout``，
        配置还没加载时用 :data:`DEFAULT_INIT_TIMEOUT`，
        见《8-集成与验收》「超时管理」一节。
        """
        if self._init_timeout is not None:
            return float(self._init_timeout)
        config = current_config()
        if config is not None:
            return float(config.audio.init_timeout)
        return DEFAULT_INIT_TIMEOUT

    def _enumerate_loopbacks(self) -> list[tuple[int, Any]]:
        """枚举 WASAPI loopback 设备，并记住编号到设备 ID 的对应关系。

        ``soundcard`` 不可导入或枚举失败时返回空表，麦克风一路照常可用。
        顺序按设备名与 ID 排序，保证同一台机器上多次枚举的编号一致。

        :returns: 二元组表，第一个是设备编号，第二个是 soundcard 设备。
        """
        try:
            import soundcard as sc
        except Exception as exc:
            log_event(
                "audio.loopback.import_failed",
                f"soundcard 不可导入，系统回环不可用：{exc}",
                level="WARNING",
            )
            return []
        try:
            candidates = list(sc.all_microphones(include_loopback=True))
        except Exception as exc:
            log_event(
                "audio.loopback.enumerate_failed",
                f"枚举系统回环失败：{exc}",
                level="WARNING",
            )
            return []
        loopbacks: list[Any] = []
        for device in candidates:
            try:
                if not bool(getattr(device, "isloopback", False)):
                    continue
            except Exception:
                continue
            loopbacks.append(device)
        loopbacks.sort(
            key=lambda device: (
                str(getattr(device, "name", "")),
                str(getattr(device, "id", "")),
            )
        )
        entries = [(_loopback_index(position), device) for position, device in enumerate(loopbacks)]
        with self._loopback_lock:
            self._loopback_ids = {
                index: str(getattr(device, "id", ""))
                for index, device in entries
                if str(getattr(device, "id", ""))
            }
        return entries

    def _default_loopback_id(self) -> str | None:
        """取系统默认扬声器端点的 ID，回环设备按这个 ID 标默认。

        :returns: 设备 ID；soundcard 不可用或取不到时返回 None。
        """
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            device_id = str(getattr(speaker, "id", ""))
        except Exception:
            return None
        return device_id or None

    def _resolve_loopback(self, device: int) -> tuple[Any, float, int]:
        """按编号找回环设备，并取出开流采样率与声道数。

        优先用上次枚举记下的设备 ID，设备拔插后 ID 与编号对不上时按当前
        枚举结果重新匹配。

        :param device: 回环设备编号。
        :returns: 三元组，第一个是 soundcard 设备，第二个是开流采样率，
            第三个是开流声道数（不超过两声道）。
        :raises AudioDeviceError: 编号找不到对应的回环设备。
        """
        import soundcard as sc

        with self._loopback_lock:
            cached_id = self._loopback_ids.get(device)
        resolved: Any = None
        if cached_id:
            try:
                candidate = sc.get_microphone(cached_id, include_loopback=True)
                if bool(getattr(candidate, "isloopback", False)):
                    resolved = candidate
            except Exception:
                resolved = None
        if resolved is None:
            for index, candidate in self._enumerate_loopbacks():
                if index == device:
                    resolved = candidate
                    break
        if resolved is None:
            raise AudioDeviceError(f"系统回环设备不可用：编号 {device}")
        channels = min(_device_channels(resolved), _MAX_STREAM_CHANNELS)
        return resolved, _loopback_native_rate(resolved), channels

    def list_devices(self) -> list[AudioDevice]:
        """列出本机的音频输入设备。

        麦克风来自 ``sounddevice``，只保留输入声道数大于 0 的设备，索引是
        ``sounddevice`` 给的真实索引。系统回环来自 ``soundcard`` 的
        ``all_microphones(include_loopback=True)``，只保留 loopback 设备，
        索引从 :data:`LOOPBACK_INDEX_BASE` 起。

        默认设备在这里标出来：默认输入索引对应的麦克风 ``is_default`` 置真；
        与 ``soundcard`` 默认扬声器端点 ID 相同的回环设备同样置真。默认探测
        失败时全部按非默认处理，不影响枚举。

        枚举是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
        见《8-集成与验收》「异步任务」与「超时管理」两节。

        :returns: 设备表，字段见 :class:`~app.platform.types.AudioDevice`。
        :raises AudioDeviceError: sounddevice 不可导入，枚举失败，或枚举超时。
        """
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioDeviceError(f"sounddevice 不可导入：{exc}") from exc

        def _probe() -> tuple[list[Any], int | None, list[tuple[int, Any]], str | None]:
            """在线程里做完三轮枚举与默认设备探测。"""
            raw = list(sd.query_devices())
            try:
                pair = sd.default.device
                default_input = _as_device_index(pair[0])
            except Exception:
                default_input = None
            loopbacks = self._enumerate_loopbacks()
            return raw, default_input, loopbacks, self._default_loopback_id()

        try:
            raw_devices, default_input, loopbacks, default_loopback_id = (
                _run_with_timeout(_probe, self.init_timeout, "枚举音频设备")
            )
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"枚举音频设备失败：{exc}") from exc

        result: list[AudioDevice] = []
        for position, device in enumerate(raw_devices):
            item = dict(device)
            channels = int(item.get("max_input_channels", 0) or 0)
            if channels <= 0:
                continue
            index = int(item.get("index", position))
            result.append(
                AudioDevice(
                    name=str(item.get("name", "")),
                    index=index,
                    is_input=True,
                    is_loopback=False,
                    default_samplerate=float(
                        item.get("default_samplerate", 0.0) or 0.0
                    ),
                    channels=channels,
                    is_default=default_input is not None and index == default_input,
                )
            )
        for index, device in loopbacks:
            name = str(getattr(device, "name", ""))
            device_id = str(getattr(device, "id", ""))
            result.append(
                AudioDevice(
                    name=f"{name}{_LOOPBACK_NAME_SUFFIX}",
                    index=index,
                    is_input=True,
                    is_loopback=True,
                    default_samplerate=_loopback_native_rate(device),
                    channels=min(_device_channels(device), _MAX_STREAM_CHANNELS),
                    is_default=bool(default_loopback_id)
                    and device_id == default_loopback_id,
                )
            )
        return result

    def record(
        self,
        *,
        sample_rate: int,
        channels: int,
        seconds: float,
    ) -> "np.ndarray":
        """采一段麦克风音频。

        这个方法只采默认麦克风，系统回环走
        :meth:`start_stream` 的回环分支。

        :param sample_rate: 采样率。
        :param channels: 声道数。
        :param seconds: 采集时长。
        :returns: float32 数组，形状是 ``(采样点, 声道数)``。
        :raises AudioDeviceError: sounddevice 不可导入，或采集失败。
        """
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioDeviceError(f"sounddevice 不可导入：{exc}") from exc
        try:
            recording = sd.rec(
                int(sample_rate * seconds),
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            raise AudioDeviceError(f"采集失败：{exc}") from exc
        return recording

    def _open_microphone_stream(
        self,
        *,
        device: int | None,
        sample_rate: int,
        block_duration_ms: int,
        on_block: AudioBlockCallback,
    ) -> AudioStream:
        """打开麦克风输入流。

        :param device: 麦克风索引，None 表示默认输入设备。
        :param sample_rate: 目标采样率。
        :param block_duration_ms: 每次回调读的毫秒数。
        :param on_block: 收到一块音频时调的回调。
        :returns: 这一路流的句柄。
        :raises AudioDeviceError: 设备不可用，开流失败，或开流超时。
        """
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioDeviceError(f"sounddevice 不可导入：{exc}") from exc
        timeout = self.init_timeout
        try:
            info = _run_with_timeout(
                lambda: sd.query_devices(device, "input"),
                timeout,
                "查询音频设备",
            )
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"查询音频设备失败：{exc}") from exc
        native_rate = float(info.get("default_samplerate", 0.0) or 0.0)
        if native_rate <= 0:
            native_rate = float(sample_rate)
        native_channels = int(info.get("max_input_channels", 0) or 0)
        if native_channels <= 0:
            native_channels = 1
        # 采集最终下混成单声道，开流最多按立体声：
        # 部分虚拟声卡驱动报出上百个输入声道，PortAudio 按这个数开流会崩溃
        if native_channels > _MAX_STREAM_CHANNELS:
            native_channels = _MAX_STREAM_CHANNELS
        # 每次回调读的量按设备原生采样率算
        block_frames = max(1, int(round(native_rate * block_duration_ms / 1000.0)))

        def _callback(
            indata: "np.ndarray",
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            """采集回调：读取、下混、重采样，然后交给上层入队。"""
            if status:
                log_event(
                    "audio.stream.status",
                    "采集流状态有告警",
                    level="DEBUG",
                    status=str(status),
                )
            samples = _downmix_to_mono(indata)
            samples = _resample_to(samples, native_rate, sample_rate)
            on_block(samples)

        def _open() -> Any:
            """建好输入流并启动它。"""
            stream = sd.InputStream(
                device=device,
                samplerate=native_rate,
                blocksize=block_frames,
                channels=native_channels,
                dtype="float32",
                callback=_callback,
            )
            stream.start()
            return stream

        try:
            stream = _run_with_timeout(_open, timeout, "开启音频流")
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"开启音频流失败：{exc}") from exc
        return _SoundDeviceStream(stream)

    def _open_loopback_stream(
        self,
        *,
        device: int,
        sample_rate: int,
        block_duration_ms: int,
        on_block: AudioBlockCallback,
    ) -> AudioStream:
        """打开 WASAPI loopback 采集流。

        :param device: 回环设备编号。
        :param sample_rate: 目标采样率。
        :param block_duration_ms: 每次读取的毫秒数。
        :param on_block: 收到一块音频时调的回调。
        :returns: 这一路流的句柄。
        :raises AudioDeviceError: 设备不可用，开流失败，或开流超时。
        """
        timeout = self.init_timeout
        try:
            loopback, native_rate, native_channels = _run_with_timeout(
                lambda: self._resolve_loopback(device),
                timeout,
                "查询音频设备",
            )
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"查询音频设备失败：{exc}") from exc
        block_frames = max(1, int(round(native_rate * block_duration_ms / 1000.0)))

        def _recorder_factory() -> Any:
            """打开 soundcard 的 recorder，由采集线程进入上下文。"""
            return loopback.recorder(
                samplerate=native_rate,
                channels=native_channels,
                blocksize=block_frames,
            )

        stream = _WasapiLoopbackStream(
            recorder_factory=_recorder_factory,
            block_frames=block_frames,
            native_rate=native_rate,
            sample_rate=sample_rate,
            on_block=on_block,
        )
        try:
            stream.start(timeout)
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"开启音频流失败：{exc}") from exc
        return stream

    def start_stream(
        self,
        *,
        device: int | None,
        sample_rate: int,
        channels: int,
        block_duration_ms: int,
        on_block: AudioBlockCallback,
    ) -> AudioStream:
        """开始回调式采集。

        编号大于等于 :data:`LOOPBACK_INDEX_BASE` 的是系统回环，走 WASAPI
        loopback；其余编号与 None 是麦克风，走 ``sounddevice``。

        两路都按设备原生采样率与原生声道数开流，回调里下混成单声道并重采样
        到 ``sample_rate``，每次出一块 ``block_duration_ms`` 毫秒的音频。
        回调线程只做读取、下混、重采样与入队，不发网络请求，也不做推理。

        开流是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
        见《8-集成与验收》「异步任务」与「超时管理」两节。

        :param device: 设备编号，None 表示默认输入设备。
        :param sample_rate: 目标采样率。
        :param channels: 目标声道数，当前只支持 1。
        :param block_duration_ms: 每次回调读的毫秒数。
        :param on_block: 收到一块音频时调的回调。
        :returns: 这一路流的句柄。
        :raises AudioDeviceError: 设备不可用，开流失败，或开流超时。
        """
        if device is not None and _is_loopback_index(device):
            return self._open_loopback_stream(
                device=device,
                sample_rate=sample_rate,
                block_duration_ms=block_duration_ms,
                on_block=on_block,
            )
        return self._open_microphone_stream(
            device=device,
            sample_rate=sample_rate,
            block_duration_ms=block_duration_ms,
            on_block=on_block,
        )