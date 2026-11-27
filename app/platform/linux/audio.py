"""Linux 侧的音频后端：sounddevice。

见《9-平台差异与Windows适配》「平台能力矩阵」一节：Linux 侧的麦克风采集与
设备枚举用 ``sounddevice``；系统回环音频落在 PulseAudio 与 PipeWire 的
monitor source 上，那一路的设备名带 ``.monitor`` 后缀，枚举时按名字认出来。

本模块把 :func:`app.main.list_input_devices` 的枚举逻辑与
:func:`app.cli.cmd_audio_test` 的采集逻辑搬过来，错误文案照抄原文。

采集时 ``sounddevice`` 必须按模块属性调用（``sd.rec``、``sd.wait``）：
既有测试会替换 ``sounddevice`` 模块上的这两个属性，绑成局部变量就替换不掉。

流式采集走 ``sd.InputStream`` 的回调方式：每次回调读 200 毫秒，
按《5-语音输入与流式ASR》「采集参数」一节下混成单声道并重采样到 16000 Hz，
回调线程只做读取与入队，不发网络请求，也不做推理。

设备枚举与开流都是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
超时报 ``E_AUDIO_DEVICE``，见《8-集成与验收》「异步任务」与「超时管理」两节。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from app.config import current_config
from app.core.exceptions import AudioDeviceError
from app.platform.types import AudioBlockCallback, AudioDevice, AudioStream
from app.utils.logger import log_event

if TYPE_CHECKING:
    import numpy as np

__all__ = ["SoundDeviceAudioBackend", "DEFAULT_INIT_TIMEOUT"]


# 系统回环设备的设备名标记：PulseAudio 与 PipeWire 的 monitor source
# 在设备名里带这个后缀，见《9-平台差异与Windows适配》「平台能力矩阵」一节。
_LOOPBACK_NAME_MARKER: str = ".monitor"

# 向 PipeWire 问默认设备的命令行工具与它认的两个目标
_WPCTL_COMMAND: str = "wpctl"
_DEFAULT_SINK_TARGET: str = "@DEFAULT_AUDIO_SINK@"
_DEFAULT_SOURCE_TARGET: str = "@DEFAULT_AUDIO_SOURCE@"

# 问一次默认设备的超时秒数：wpctl 只是查状态，超时就当拿不到
_WPCTL_TIMEOUT: float = 2.0

# wpctl inspect 输出里节点名的行，形如 ``  * node.name = "alsa_output.xxx"``，
# 行首可能带 ``* `` 标记当前节点，值在引号里
_WPCTL_NODE_NAME_PATTERN = re.compile(r'^\s*(?:\*\s*)?node\.name\s*=\s*"([^"]*)"\s*$')

# 音频设备初始化超时的默认秒数：进程里没加载过配置时用它，
# 见《8-集成与验收》「超时管理」一节
DEFAULT_INIT_TIMEOUT: float = 10.0

_T = TypeVar("_T")


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

    设备原生采样率是 44100 或 48000，采集目标是 16000，重采样自己做，
    不引新依赖，见《5-语音输入与流式ASR》「采集参数」一节。

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


def _open_input_stream(
    sd: Any,
    *,
    device: int | None,
    native_rate: float,
    native_channels: int,
    block_frames: int,
    callback: Callable[["np.ndarray", int, Any, Any], None],
) -> Any:
    """建好输入流并启动它。

    PortAudio 开流是阻塞调用，这一步整个放到线程里跑，
    见《8-集成与验收》「异步任务」一节。

    :param sd: sounddevice 模块。
    :param device: 设备索引。
    :param native_rate: 设备原生采样率。
    :param native_channels: 设备原生声道数。
    :param block_frames: 每次回调读的帧数。
    :param callback: 采集回调。
    :returns: 已经启动的输入流。
    """
    stream = sd.InputStream(
        device=device,
        samplerate=native_rate,
        blocksize=block_frames,
        channels=native_channels,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    return stream


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


def _parse_wpctl_node_name(stdout: str) -> str | None:
    """从 ``wpctl inspect`` 的输出里取节点名。

    输出里 ``node.name = "<名字>"`` 那一行就是要的名字，行首可能带 ``* ``。

    :param stdout: 命令的 stdout。
    :returns: 节点名；输出里没有这一行时返回 None。
    """
    for line in stdout.splitlines():
        match = _WPCTL_NODE_NAME_PATTERN.match(line)
        if match is not None:
            name = match.group(1).strip()
            return name or None
    return None


def _query_wpctl_node_name(target: str) -> str | None:
    """问 PipeWire 某个默认设备（sink 或 source）的节点名。

    wpctl 不可用、超时、退出码非零、输出里没有节点名，一律返回 None，
    由调用方退回 ``sounddevice`` 的默认设备方案。

    :param target: ``@DEFAULT_AUDIO_SINK@`` 或 ``@DEFAULT_AUDIO_SOURCE@``。
    :returns: 节点名；拿不到时返回 None。
    """
    executable = shutil.which(_WPCTL_COMMAND)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "inspect", target],
            capture_output=True,
            text=True,
            timeout=_WPCTL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _parse_wpctl_node_name(completed.stdout or "")


def _find_input_index_by_name(devices: list[Any], name: str) -> int | None:
    """在设备表里按设备名找一个能采集的输入设备，返回它的索引。

    :param devices: 同一次 ``sd.query_devices()`` 的全量结果。
    :param name: 要匹配的设备名。
    :returns: 设备索引；没有同名输入设备时返回 None。
    """
    for position, device in enumerate(devices):
        try:
            item = dict(device)
            if str(item.get("name", "")) != name:
                continue
            if int(item.get("max_input_channels", 0)) <= 0:
                continue
            return int(item.get("index", position))
        except Exception:
            continue
    return None


def _has_device_named(devices: list[Any], name: str) -> bool:
    """设备表里有没有叫这个名字的设备。

    :param devices: 同一次 ``sd.query_devices()`` 的全量结果。
    :param name: 要匹配的设备名。
    :returns: 有同名设备返回 True。
    """
    for device in devices:
        try:
            if str(dict(device).get("name", "")) == name:
                return True
        except Exception:
            continue
    return False


def _detect_defaults_via_wpctl(
    devices: list[Any],
) -> tuple[int | None, str | None]:
    """按 PipeWire 的当前默认设备解析默认输入索引与默认回环设备名。

    默认 source 的节点名就是默认输入设备的设备名；默认 sink 的节点名加上
    ``.monitor`` 就是它的回环设备名。两个都要在设备表里匹配到才算成功，
    少一个就整体当作没解析出来，由调用方退回 ``sounddevice`` 的方案。

    :param devices: 同一次 ``sd.query_devices()`` 的全量结果。
    :returns: 二元组，第一个是默认输入索引，第二个是默认回环设备名。
    """
    source_name = _query_wpctl_node_name(_DEFAULT_SOURCE_TARGET)
    sink_name = _query_wpctl_node_name(_DEFAULT_SINK_TARGET)
    if source_name is None or sink_name is None:
        return None, None
    input_index = _find_input_index_by_name(devices, source_name)
    loopback_name: str | None = sink_name + _LOOPBACK_NAME_MARKER
    if not _has_device_named(devices, loopback_name):
        return None, None
    if input_index is None:
        return None, None
    return input_index, loopback_name


def _detect_defaults_via_sounddevice(
    sd: Any, devices: list[Any]
) -> tuple[int | None, str | None]:
    """按 ``sounddevice`` 给出的默认设备解析默认输入索引与默认回环设备名。

    默认输入按 ``sd.default.device`` 的输入索引标；默认回环取默认输出设备名
    加 ``.monitor``。探测失败（拿不到 ``sd.default``、取值异常）一律返回
    ``(None, None)``，不能让设备枚举跟着失败。

    :param sd: sounddevice 模块。
    :param devices: 同一次 ``sd.query_devices()`` 的全量结果。
    :returns: 二元组，第一个是默认输入索引，第二个是默认回环设备名。
    """
    try:
        pair = sd.default.device
        input_index = _as_device_index(pair[0])
        output_index = _as_device_index(pair[1])
    except Exception:
        return None, None
    name_out: str | None = None
    if output_index is not None:
        for position, device in enumerate(devices):
            try:
                item = dict(device)
                index = int(item.get("index", position))
            except Exception:
                continue
            if index == output_index:
                name_out = str(item.get("name", ""))
                break
    if name_out is None:
        return input_index, None
    return input_index, name_out + _LOOPBACK_NAME_MARKER


def _detect_default_devices(
    sd: Any, devices: list[Any]
) -> tuple[int | None, str | None]:
    """解析默认输入索引与默认回环设备名，三级回退。

    先用 ``wpctl`` 问 PipeWire 的默认 sink 与 source，这是设备当前真实生效的
    默认设备；wpctl 不可用或匹配不到时退回 ``sounddevice`` 的默认设备方案；
    两者都失败时返回 ``(None, None)``，设备表照常返回，只是都不标默认。

    :param sd: sounddevice 模块。
    :param devices: 同一次 ``sd.query_devices()`` 的全量结果。
    :returns: 二元组，第一个是默认输入索引，第二个是默认回环设备名。
    """
    input_index, loopback_name = _detect_defaults_via_wpctl(devices)
    if input_index is not None and loopback_name is not None:
        return input_index, loopback_name
    return _detect_defaults_via_sounddevice(sd, devices)


class SoundDeviceAudioBackend:
    """基于 sounddevice 的音频设备枚举与采集后端。

    :param init_timeout: 音频设备初始化超时秒数，None 时取进程里加载好的
        配置里的 ``audio.init_timeout``，见《8-集成与验收》「超时管理」一节。
    """

    # 后端名
    name: str = "sounddevice"

    # 该能力需要的第三方包名
    required_packages: tuple[str, ...] = ("sounddevice",)

    def __init__(self, init_timeout: float | None = None) -> None:
        self._init_timeout = init_timeout

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

    def list_devices(self) -> list[AudioDevice]:
        """列出本机的音频输入设备。

        只保留输入声道数大于 0 的设备，设备索引取 ``sounddevice`` 给的真实索引。
        设备名里带 ``.monitor`` 的是系统回环设备，``is_loopback`` 置真。

        默认设备也在这里标出来：默认输入索引对应的设备 ``is_default`` 置真；
        默认输出 sink 的回环设备（设备名等于默认输出节点名加 ``.monitor``）
        同样置真。默认设备先问 PipeWire（``wpctl``），问不到再按
        ``sounddevice`` 的默认设备取值，两者都失败时全部按非默认处理，
        不影响枚举本身。

        枚举是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
        见《8-集成与验收》「异步任务」与「超时管理」两节。

        :returns: 设备表，字段见 :class:`~app.platform.types.AudioDevice`。
        :raises AudioDeviceError: sounddevice 不可导入，枚举失败，或枚举超时。
        """
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioDeviceError(f"sounddevice 不可导入：{exc}") from exc
        try:
            devices = _run_with_timeout(
                lambda: sd.query_devices(),
                self.init_timeout,
                "枚举音频设备",
            )
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"枚举音频设备失败：{exc}") from exc
        result: list[AudioDevice] = []
        # 默认探测放在结果组装之前，失败时两个值都是 None，不影响枚举
        default_input_index, default_loopback_name = _detect_default_devices(
            sd, devices
        )
        for position, device in enumerate(devices):
            item = dict(device)
            if int(item.get("max_input_channels", 0)) <= 0:
                continue
            name = str(item.get("name", ""))
            index = int(item.get("index", position))
            is_default = default_input_index is not None and index == default_input_index
            if default_loopback_name is not None and name == default_loopback_name:
                is_default = True
            result.append(
                AudioDevice(
                    name=name,
                    index=index,
                    is_input=True,
                    is_loopback=_LOOPBACK_NAME_MARKER in name,
                    default_samplerate=float(
                        item.get("default_samplerate", 0.0) or 0.0
                    ),
                    channels=int(item.get("max_input_channels", 0)),
                    is_default=is_default,
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
        """采一段音频。

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

        流按设备原生采样率与原生声道数打开，每次回调读 ``block_duration_ms``
        毫秒，下混成单声道并重采样到 ``sample_rate`` 后交给 ``on_block``。
        回调线程只做读取与入队，不发网络请求，也不做推理。

        开流是阻塞调用，放到线程里跑并按 ``audio.init_timeout`` 限时，
        见《8-集成与验收》「异步任务」与「超时管理」两节。

        :param device: 设备索引，None 表示用默认输入设备。
        :param sample_rate: 目标采样率。
        :param channels: 目标声道数，当前只支持 1。
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
        # 采集最终下混成单声道，开流最多按立体声；PipeWire 的默认设备会
        # 报出上百个输入声道，PortAudio 按这个数开流会崩溃
        if native_channels > 2:
            native_channels = 2
        # 每次回调读的量按设备原生采样率算，200 毫秒是回调读取量
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

        try:
            stream = _run_with_timeout(
                lambda: _open_input_stream(
                    sd,
                    device=device,
                    native_rate=native_rate,
                    native_channels=native_channels,
                    block_frames=block_frames,
                    callback=_callback,
                ),
                timeout,
                "开启音频流",
            )
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"开启音频流失败：{exc}") from exc
        return _SoundDeviceStream(stream)
