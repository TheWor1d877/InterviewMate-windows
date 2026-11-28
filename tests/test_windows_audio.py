"""Windows 音频后端的 COM 初始化路径测试。

``soundcard`` 只在导入它的线程上初始化 COM（它的 ``mediafoundation`` 模块
在 import 时对当前线程调一次 ``CoInitialize``），而项目的枚举与采集都在
独立线程里跑。线程没有初始化 COM 时 ``CoCreateInstance`` 返回
``CO_E_NOTINITIALIZED``（0x800401f0），回环设备一枚都枚不到，主进程日志里
反复出现 ``audio.loopback.enumerate_failed``。

这里用假 soundcard 模块与替身验证修复后的路径：

- 枚举回环前当前线程的 COM 被初始化，离开时配对释放；
- 线程已按别的套间模型初始化（``RPC_E_CHANGED_MODE``）时不做配对释放，
  枚举照常进行；
- 默认扬声器探测与按 ID 找回环设备同样走初始化与配对释放；
- 回环采集线程整个生命周期内初始化一次、退出时释放一次；
- ``_com_apartment`` 中途抛错也配对释放；
- 真实 ``ole32`` 路径（Windows）下初始化与释放可配对反复使用。
"""

from __future__ import annotations

import sys
import threading
import wave as wave_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import app.platform.windows.audio as audio_module
from app.audio.loopback_test import SoundDevicePlayer, run_loopback_test
from app.audio.asr_replay import load_wav_int16_mono, pcm_int16_to_float32
from app.core.exceptions import AudioDeviceError
from app.platform.windows.audio import (
    LOOPBACK_INDEX_BASE,
    WasapiAudioBackend,
    _WasapiLoopbackStream,
)

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
class _FakeLoopbackDevice:
    """假 soundcard 设备：够 ``_enumerate_loopbacks`` 取字段用。"""

    def __init__(self, device_id: str, name: str, *, isloopback: bool = True) -> None:
        self.id = device_id
        self.name = name
        self.isloopback = isloopback


def _install_fake_soundcard(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loopbacks: list[_FakeLoopbackDevice] | None = None,
    speaker_id: str | None = "spk-default",
) -> ModuleType:
    """把 soundcard 换成假模块，返回这个假模块。"""
    fake = ModuleType("soundcard")
    fake.loopbacks = list(loopbacks or [])  # type: ignore[attr-defined]
    fake.speaker_id = speaker_id  # type: ignore[attr-defined]

    def all_microphones(include_loopback: bool = False) -> list[Any]:
        """按 include_loopback 返回回环设备。"""
        if not include_loopback:
            return []
        return list(fake.loopbacks)  # type: ignore[attr-defined]

    def default_speaker() -> SimpleNamespace:
        """返回默认扬声器端点。"""
        return SimpleNamespace(id=fake.speaker_id)  # type: ignore[attr-defined]

    def get_microphone(device_id: str, include_loopback: bool = False) -> Any:
        """按 WASAPI ID 找设备，找不到就报错。"""
        for device in fake.loopbacks:  # type: ignore[attr-defined]
            if device.id == device_id:
                return device
        raise LookupError(f"设备不存在：{device_id}")

    fake.all_microphones = all_microphones  # type: ignore[attr-defined]
    fake.default_speaker = default_speaker  # type: ignore[attr-defined]
    fake.get_microphone = get_microphone  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soundcard", fake)
    return fake


class _ComRecorder:
    """替掉 _com_initialize/_com_uninitialize，记下初始化与释放次数。"""

    def __init__(self) -> None:
        self.initialized = 0
        self.uninitialized = 0

    def install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        changed_mode: bool = False,
    ) -> None:
        """装上计数替身；``changed_mode`` 模拟 RPC_E_CHANGED_MODE。"""

        def fake_initialize() -> bool:
            self.initialized += 1
            return not changed_mode

        def fake_uninitialize() -> None:
            self.uninitialized += 1

        monkeypatch.setattr(audio_module, "_com_initialize", fake_initialize)
        monkeypatch.setattr(audio_module, "_com_uninitialize", fake_uninitialize)


class _FakeRecorder:
    """假 soundcard recorder：上下文管理器，record 返回全零块。"""

    def __init__(self, *, channels: int = 2) -> None:
        self.channels = channels
        self.record_calls = 0

    def __enter__(self) -> "_FakeRecorder":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def record(self, numframes: int) -> Any:
        """返回一块全零的 ``(帧数, 声道数)`` float32 数据。"""
        self.record_calls += 1
        return np.zeros((numframes, self.channels), dtype="float32")


# --------------------------------------------------------------------------
# COM 初始化路径
# --------------------------------------------------------------------------
def test_enumerate_loopbacks_initializes_and_releases_com(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """枚举回环前初始化 COM，离开时配对释放，回环设备能枚举到。"""
    recorder = _ComRecorder()
    recorder.install(monkeypatch)
    _install_fake_soundcard(
        monkeypatch, loopbacks=[_FakeLoopbackDevice("spk-default", "扬声器")]
    )

    entries = WasapiAudioBackend()._enumerate_loopbacks()

    assert [index for index, _device in entries] == [LOOPBACK_INDEX_BASE]
    assert recorder.initialized == 1
    assert recorder.uninitialized == 1


def test_enumerate_loopbacks_keeps_com_when_thread_in_other_apartment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """线程已按别的套间模型初始化（RPC_E_CHANGED_MODE）时不做配对释放。

    这次调用没有加引用计数，释放反而会把别人的初始化注销掉。
    """
    recorder = _ComRecorder()
    recorder.install(monkeypatch, changed_mode=True)
    _install_fake_soundcard(
        monkeypatch, loopbacks=[_FakeLoopbackDevice("spk-default", "扬声器")]
    )

    entries = WasapiAudioBackend()._enumerate_loopbacks()

    assert len(entries) == 1
    assert recorder.initialized == 1
    assert recorder.uninitialized == 0


def test_default_loopback_id_initializes_com(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认扬声器探测同样走初始化与配对释放。"""
    recorder = _ComRecorder()
    recorder.install(monkeypatch)
    _install_fake_soundcard(monkeypatch, speaker_id="spk-default")

    backend = WasapiAudioBackend()

    assert backend._default_loopback_id() == "spk-default"
    assert recorder.initialized == 1
    assert recorder.uninitialized == 1


def test_resolve_loopback_initializes_com_and_pairs_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按编号找回环设备时初始化 COM，释放次数与初始化一致。"""
    recorder = _ComRecorder()
    recorder.install(monkeypatch)
    _install_fake_soundcard(
        monkeypatch, loopbacks=[_FakeLoopbackDevice("spk-default", "扬声器")]
    )

    device, rate, channels = WasapiAudioBackend()._resolve_loopback(LOOPBACK_INDEX_BASE)

    assert device.id == "spk-default"
    assert rate > 0
    assert channels >= 1
    assert recorder.initialized >= 1
    assert recorder.uninitialized == recorder.initialized


def test_loopback_stream_thread_initializes_com(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回环采集线程整个生命周期内初始化一次 COM，退出时释放一次。"""
    recorder = _ComRecorder()
    recorder.install(monkeypatch)
    fake_recorder = _FakeRecorder()
    got_block = threading.Event()

    def on_block(samples: Any) -> None:
        got_block.set()

    stream = _WasapiLoopbackStream(
        recorder_factory=lambda: fake_recorder,
        block_frames=9600,
        native_rate=48000.0,
        sample_rate=16000,
        on_block=on_block,
    )
    stream.start(timeout=2.0)
    try:
        assert got_block.wait(2.0), "采集线程没有出块"
    finally:
        stream.stop()

    assert fake_recorder.record_calls >= 1
    assert recorder.initialized == 1
    assert recorder.uninitialized == 1


def test_com_apartment_releases_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上下文体中途抛错也配对释放。"""
    recorder = _ComRecorder()
    recorder.install(monkeypatch)

    with pytest.raises(RuntimeError):
        with audio_module._com_apartment():
            raise RuntimeError("中途炸了")

    assert recorder.initialized == 1
    assert recorder.uninitialized == 1


@pytest.mark.skipif(sys.platform != "win32", reason="只覆盖 Windows 的 ole32 路径")
def test_com_initialize_pairs_with_uninitialize_on_windows() -> None:
    """真实 ole32：全新线程里初始化返回 True，配对释放后还能再次初始化。"""
    results: dict[str, Any] = {}

    def target() -> None:
        results["first"] = audio_module._com_initialize()
        audio_module._com_uninitialize()
        results["second"] = audio_module._com_initialize()
        audio_module._com_uninitialize()

    thread = threading.Thread(target=target, name="im-test-com")
    thread.start()
    thread.join(5.0)
    assert not thread.is_alive(), "COM 初始化线程卡住了"
    assert results["first"] is True
    assert results["second"] is True

# --------------------------------------------------------------------------
# loopback-test 编排
# --------------------------------------------------------------------------
def _write_loopback_wav(
    path: Path,
    *,
    rate: int = 16000,
    seconds: float = 0.3,
    amplitude: float = 0.5,
) -> None:
    """生成一段 16k/16bit/单声道的 wav，内容是正弦，供编排测试读用。"""
    frames = int(rate * seconds)
    t = np.arange(frames, dtype="float32") / rate
    wave = (amplitude * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    pcm = np.clip(np.rint(wave * 32768.0), -32768, 32767).astype("<i2")
    with wave_module.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


class _FakeLoopbackEngine:
    """假 ASR 引擎：把调用顺序写进共享事件表，记录喂进去的块。"""

    def __init__(
        self,
        events: list[str],
        *,
        feed_text: str = "增量",
        final_text: str = "最终文本",
    ) -> None:
        self._events = events
        self._feed_text = feed_text
        self._final_text = final_text
        self.fed: list[Any] = []

    async def ensure_loaded(self) -> None:
        """记录引擎加载完成。"""
        self._events.append("engine_loaded")

    async def feed(self, block: Any) -> str:
        """记录一次喂块，返回预置的增量文本。"""
        self._events.append("feed")
        self.fed.append(np.asarray(block, dtype="float32").reshape(-1))
        return self._feed_text

    async def flush(self) -> str:
        """记录冲刷，返回预置的最终文本。"""
        self._events.append("flush")
        return self._final_text


class _FakeLoopbackPlayer:
    """假播放器：记录播放的波形与采样率。"""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.played: Any = None
        self.rate: int | None = None

    def play(self, samples: Any, sample_rate: int) -> None:
        """记录一次播放，立即返回。"""
        self._events.append("play")
        self.played = np.asarray(samples, dtype="float32").reshape(-1)
        self.rate = sample_rate


class _FakeLoopbackRecorder:
    """假采集器：start 记录顺序，stop 返回预置的采集波形。"""

    def __init__(self, events: list[str], captured: Any) -> None:
        self._events = events
        self._captured = captured
        self.start_error: BaseException | None = None

    def start(self) -> None:
        """记录启动；预置了 start_error 时抛出去。"""
        if self.start_error is not None:
            raise self.start_error
        self._events.append("record_start")

    def stop(self) -> Any:
        """记录停流，返回预置波形。"""
        self._events.append("record_stop")
        return self._captured


async def test_run_loopback_test_orchestrates_play_capture_and_asr(
    tmp_path: Path,
) -> None:
    """编排顺序与数据流：加载引擎→启动采集→播放→停流→喂引擎→冲刷。"""
    wav_path = tmp_path / "case.wav"
    _write_loopback_wav(wav_path, seconds=0.3)  # 4800 点 → 两块
    events: list[str] = []
    engine = _FakeLoopbackEngine(events)
    player = _FakeLoopbackPlayer(events)
    t = np.arange(4800, dtype="float32") / 16000
    captured = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    recorder = _FakeLoopbackRecorder(events, captured)

    result = await run_loopback_test(wav_path, engine, player, recorder)

    assert events == [
        "engine_loaded",
        "record_start",
        "play",
        "record_stop",
        "feed",
        "feed",
        "flush",
    ]
    assert player.rate == 16000
    # 播放的波形是 wav 的内容（int16 往返量化误差放宽）
    expected = pcm_int16_to_float32(load_wav_int16_mono(wav_path))
    assert np.allclose(player.played, expected, atol=1e-3)
    # 引擎收到的是采集到的波形（不是源文件），按 200 毫秒切成两块
    assert len(engine.fed) == 2
    assert np.allclose(np.concatenate(engine.fed), captured, atol=1e-6)
    # 结果聚合：识别文本拼接、时长与电平都对得上
    assert result.text == "增量增量最终文本"
    assert result.ok is True
    assert result.peak == pytest.approx(0.4, abs=1e-3)
    assert result.played_seconds == pytest.approx(0.3)
    assert result.captured_seconds == pytest.approx(0.3)


async def test_run_loopback_test_passes_with_loud_capture_and_no_text(
    tmp_path: Path,
) -> None:
    """识别为空但录到明显电平：链路算通，结论提示没识别出文本。"""
    wav_path = tmp_path / "loud.wav"
    _write_loopback_wav(wav_path, seconds=0.2, amplitude=0.5)
    events: list[str] = []
    engine = _FakeLoopbackEngine(events, feed_text="", final_text="")
    player = _FakeLoopbackPlayer(events)
    t = np.arange(3200, dtype="float32") / 16000
    captured = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    recorder = _FakeLoopbackRecorder(events, captured)

    result = await run_loopback_test(wav_path, engine, player, recorder)

    assert result.text == ""
    assert result.peak == pytest.approx(0.5, abs=1e-3)
    assert result.ok is True
    assert "没有识别出文本" in result.conclusion


async def test_run_loopback_test_fails_when_capture_is_silent(tmp_path: Path) -> None:
    """识别为空且全程静音：链路不算通，结论提示检查音量与默认输出设备。"""
    wav_path = tmp_path / "silent.wav"
    _write_loopback_wav(wav_path, seconds=0.2)
    events: list[str] = []
    engine = _FakeLoopbackEngine(events, feed_text="", final_text="")
    player = _FakeLoopbackPlayer(events)
    recorder = _FakeLoopbackRecorder(events, np.zeros(3200, dtype="float32"))

    result = await run_loopback_test(wav_path, engine, player, recorder)

    assert result.text == ""
    assert result.peak == pytest.approx(0.0, abs=1e-6)
    assert result.ok is False
    assert "音量" in result.conclusion
    assert "默认输出设备" in result.conclusion


async def test_run_loopback_test_expected_is_reference_only(tmp_path: Path) -> None:
    """期望文案只做参考对比：不一致不影响链路是否算通。"""
    wav_path = tmp_path / "expected.wav"
    _write_loopback_wav(wav_path, seconds=0.2)
    events: list[str] = []
    engine = _FakeLoopbackEngine(events, feed_text="解释一下", final_text="什么是GDS")
    player = _FakeLoopbackPlayer(events)
    recorder = _FakeLoopbackRecorder(events, np.zeros(3200, dtype="float32"))

    result = await run_loopback_test(
        wav_path, engine, player, recorder, expected="解释一下什么是GDS"
    )

    assert result.text == "解释一下什么是GDS"
    assert result.expected == "解释一下什么是GDS"
    assert result.expected_match is True
    assert result.ok is True

    # 换成不匹配的期望：expected_match 变假，但识别文本非空，链路照样算通
    engine = _FakeLoopbackEngine(events, feed_text="解释一下", final_text="什么是GDS")
    recorder = _FakeLoopbackRecorder(events, np.zeros(3200, dtype="float32"))
    result = await run_loopback_test(
        wav_path, engine, player, recorder, expected="别的文案"
    )
    assert result.expected_match is False
    assert result.ok is True


async def test_run_loopback_test_propagates_recorder_failure(tmp_path: Path) -> None:
    """采集器启动失败时异常原样抛出，播放与喂引擎都不发生。"""
    wav_path = tmp_path / "fail.wav"
    _write_loopback_wav(wav_path, seconds=0.2)
    events: list[str] = []
    engine = _FakeLoopbackEngine(events)
    player = _FakeLoopbackPlayer(events)
    recorder = _FakeLoopbackRecorder(events, np.zeros(0, dtype="float32"))
    recorder.start_error = AudioDeviceError("回环设备不可用")

    with pytest.raises(AudioDeviceError):
        await run_loopback_test(wav_path, engine, player, recorder)

    assert "play" not in events
    assert "feed" not in events
    assert "engine_loaded" in events

def test_sound_device_player_resamples_to_device_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """输出设备默认采样率与波形不一致时，先重采样再播，避免设备拒绝开流。"""
    import sounddevice as sd

    captured: dict[str, Any] = {}

    def fake_query(device: Any, kind: Any) -> dict[str, Any]:
        assert device is None and kind == "output"
        return {"default_samplerate": 48000.0}

    def fake_play(data: Any, samplerate: float, **kwargs: Any) -> None:
        captured["data"] = np.asarray(data, dtype="float32").reshape(-1)
        captured["rate"] = int(samplerate)

    monkeypatch.setattr(sd, "query_devices", fake_query)
    monkeypatch.setattr(sd, "play", fake_play)
    monkeypatch.setattr(sd, "wait", lambda: None)

    samples = np.zeros(1600, dtype="float32")  # 0.1 秒 @16k
    SoundDevicePlayer().play(samples, 16000)

    assert captured["rate"] == 48000
    assert captured["data"].size == 4800  # 0.1 秒 @48k


def test_sound_device_player_keeps_rate_when_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查不到设备默认采样率时按原采样率播，行为与不加兜底一致。"""
    import sounddevice as sd

    captured: dict[str, Any] = {}

    def fake_query(device: Any, kind: Any) -> dict[str, Any]:
        raise RuntimeError("枚举失败")

    def fake_play(data: Any, samplerate: float, **kwargs: Any) -> None:
        captured["data"] = np.asarray(data, dtype="float32").reshape(-1)
        captured["rate"] = int(samplerate)

    monkeypatch.setattr(sd, "query_devices", fake_query)
    monkeypatch.setattr(sd, "play", fake_play)
    monkeypatch.setattr(sd, "wait", lambda: None)

    samples = np.zeros(1600, dtype="float32")
    SoundDevicePlayer().play(samples, 16000)

    assert captured["rate"] == 16000
    assert captured["data"].size == 1600