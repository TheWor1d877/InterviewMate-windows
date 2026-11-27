"""音频设备枚举与来源选择。

见《5-语音输入与流式ASR》「音频来源」一节：

- 来源取值是 ``microphone``、``system_monitor``、``mixed``、``auto``；
- 同一类设备里优先选 ``is_default`` 为真的系统默认设备，没有默认标记时
  取列表第一个；
- ``auto`` 在默认麦克风与默认回环都可用时混成一路，只有回环时用回环，
  只有麦克风时用麦克风，两个都不通时报错误码 ``E_AUDIO_DEVICE``；
- 麦克风与系统回环在设备表里分得开，来源选择就靠这个区分。

设备表来自平台音频后端，字段见 :class:`~app.platform.types.AudioDevice`：
``is_loopback`` 为真的是系统回环设备（Linux 的 monitor source、
Windows 的 WASAPI loopback），其余输入设备是麦克风；``is_default`` 为真的
是系统当前的默认设备，配置不绑定具体设备名或索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.exceptions import AudioDeviceError
from app.platform.types import AudioBackend, AudioDevice

__all__ = [
    "AudioSource",
    "AUDIO_SOURCES",
    "SOURCE_MICROPHONE",
    "SOURCE_SYSTEM_MONITOR",
    "SOURCE_MIXED",
    "SOURCE_AUTO",
    "SourceSelection",
    "list_audio_devices",
    "split_devices",
    "select_source",
    "resolve_source",
]


# 来源的合法取值，与配置项 ``audio.source`` 一致，
# 见《5-语音输入与流式ASR》「音频来源」一节
AudioSource = Literal["microphone", "system_monitor", "mixed", "auto"]

AUDIO_SOURCES: tuple[str, ...] = ("microphone", "system_monitor", "mixed", "auto")

# 麦克风一路的来源标签
SOURCE_MICROPHONE: str = "microphone"

# 系统回环一路的来源标签
SOURCE_SYSTEM_MONITOR: str = "system_monitor"

# 两路混成一路时的来源标签
SOURCE_MIXED: str = "mixed"

# 自动选择时的来源取值，选中之后会解析成上面三个里的一个
SOURCE_AUTO: str = "auto"


@dataclass(frozen=True)
class SourceSelection:
    """一次来源选择的结果。

    :param source: 实际生效的来源。``auto`` 会解析成 ``microphone``、
        ``system_monitor`` 或 ``mixed``，不会停在 ``auto`` 上。
    :param devices: 这一路要开的设备。``mixed`` 是两个，
        第一个是麦克风，第二个是系统回环。
    """

    source: str
    devices: tuple[AudioDevice, ...]

    @property
    def device(self) -> AudioDevice:
        """第一个设备，单路来源时就是它。"""
        return self.devices[0]


def list_audio_devices(backend: AudioBackend) -> list[AudioDevice]:
    """列出本机的音频输入设备。

    :param backend: 平台音频后端。
    :returns: 设备表，字段见 :class:`~app.platform.types.AudioDevice`。
    :raises AudioDeviceError: 底层音频库不可导入，或枚举失败。
    """
    return backend.list_devices()


def split_devices(
    devices: list[AudioDevice],
) -> tuple[list[AudioDevice], list[AudioDevice]]:
    """把设备表分成麦克风与系统回环两堆。

    :param devices: 设备表。
    :returns: 二元组，第一个是麦克风列表，第二个是系统回环列表。
    """
    microphones = [
        device for device in devices if device.is_input and not device.is_loopback
    ]
    loopbacks = [device for device in devices if device.is_input and device.is_loopback]
    return microphones, loopbacks


def _pick(devices: list[AudioDevice], source: str) -> AudioDevice:
    """取系统默认设备，没有默认标记时取第一个，空列表报错。

    :param devices: 候选设备。
    :param source: 来源名，进了错误文案。
    :returns: 选中的设备。
    :raises AudioDeviceError: 候选设备为空。
    """
    if not devices:
        raise AudioDeviceError(f"没有可用的{source}设备")
    for device in devices:
        if device.is_default:
            return device
    return devices[0]


def select_source(source: str, devices: list[AudioDevice]) -> SourceSelection:
    """按来源选设备。

    同类设备里优先系统默认设备，没有默认标记时取列表第一个。
    ``auto`` 在默认麦克风与默认回环都可用时混成一路（来源标签是
    ``mixed``），只有回环时用回环，只有麦克风时用麦克风；两个都不通抛
    :class:`~app.core.exceptions.AudioDeviceError`，错误码是 ``E_AUDIO_DEVICE``，
    见《5-语音输入与流式ASR》「音频来源」一节。

    :param source: 来源名，取值见 :data:`AUDIO_SOURCES`。
    :param devices: 设备表。
    :returns: 选择结果。
    :raises AudioDeviceError: 来源名不合法，或这一路没有可用设备。
    """
    if source not in AUDIO_SOURCES:
        legal = "、".join(AUDIO_SOURCES)
        raise AudioDeviceError(f"未知的音频来源：{source!r}，合法取值：{legal}")
    microphones, loopbacks = split_devices(devices)
    if source == SOURCE_MICROPHONE:
        return SourceSelection(
            source=SOURCE_MICROPHONE,
            devices=(_pick(microphones, "麦克风"),),
        )
    if source == SOURCE_SYSTEM_MONITOR:
        return SourceSelection(
            source=SOURCE_SYSTEM_MONITOR,
            devices=(_pick(loopbacks, "系统回环"),),
        )
    if source == SOURCE_MIXED:
        # mixed 两路都要有，缺一路就报错
        return SourceSelection(
            source=SOURCE_MIXED,
            devices=(
                _pick(microphones, "麦克风"),
                _pick(loopbacks, "系统回环"),
            ),
        )
    # auto：默认麦克风与默认回环都有时混成一路，只有一路时用那一路
    if microphones and loopbacks:
        return SourceSelection(
            source=SOURCE_MIXED,
            devices=(
                _pick(microphones, "麦克风"),
                _pick(loopbacks, "系统回环"),
            ),
        )
    if loopbacks:
        return SourceSelection(
            source=SOURCE_SYSTEM_MONITOR,
            devices=(_pick(loopbacks, "系统回环"),),
        )
    if microphones:
        return SourceSelection(
            source=SOURCE_MICROPHONE,
            devices=(_pick(microphones, "麦克风"),),
        )
    raise AudioDeviceError("没有可用的音频设备：系统回环与麦克风都不通")


def resolve_source(backend: AudioBackend, source: str) -> SourceSelection:
    """枚举设备并按来源选设备。

    :param backend: 平台音频后端。
    :param source: 来源名，取值见 :data:`AUDIO_SOURCES`。
    :returns: 选择结果。
    :raises AudioDeviceError: 枚举失败，来源名不合法，或这一路没有可用设备。
    """
    return select_source(source, list_audio_devices(backend))
