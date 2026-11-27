"""Linux 侧的全局按键后端：直接读 ``/dev/input``。

Wayland 下拿不到系统级的按键，Textual 的按键绑定又只在本终端有焦点时
生效，也拿不到松键。这里绕开界面层，直接读内核的事件设备
``/dev/input/event*``：每台键盘一个后台线程，``os.read`` 阻塞读
``input_event`` 结构体，把 EV_KEY 上的按下与松开交给
:class:`~app.inputs.global_hotkeys.HotkeyListener`。

只依赖标准库，不用 evdev/pynput：

- 结构体格式 ``llHHi``（timeval 两个 long、type/code 两个 u16、value 一个 s32），
  64 位 Linux 上是 24 字节；
- 键盘识别靠 ioctl ``EVIOCGNAME`` 取名字、``EVIOCGBIT`` 取能力位：
  事件类型含 EV_KEY 且键集含 KEY_A(30) 才算键盘；
- 没有权限、设备消失、能力位读不出来都只跳过该设备并记 WARNING，
  不抛到调用方；
- 只支持 f1..f12，其余键不入表。

读 ``/dev/input`` 需要当前用户在 ``input`` 组里，见本仓的部署说明。
"""

from __future__ import annotations

import fcntl
import glob
import os
import struct
import threading
from collections.abc import Callable

from app.utils.logger import log_event

__all__ = [
    "EvdevHotkeyProvider",
    "KEY_NAME_TO_CODE",
    "INPUT_EVENT_SIZE",
    "is_keyboard_capability",
]

# input_event 结构体：timeval 两个 long，type/code 两个 u16，value 一个 s32
INPUT_EVENT_FORMAT: str = "llHHi"
INPUT_EVENT_SIZE: int = struct.calcsize(INPUT_EVENT_FORMAT)

# 事件类型：EV_SYN=0，EV_KEY=1
EV_SYN: int = 0
EV_KEY: int = 1

# 按键值：0 松开，1 按下，2 长按重复（忽略）
KEY_RELEASE: int = 0
KEY_PRESS: int = 1
KEY_REPEAT: int = 2

# 键盘识别用的键码
KEY_A: int = 30

# 支持表：F1..F10 是 59..68，F11 是 87，F12 是 88，其余键不入表
KEY_NAME_TO_CODE: dict[str, int] = {
    **{f"f{index}": 58 + index for index in range(1, 11)},
    "f11": 87,
    "f12": 88,
}

# 键码到键名，事件解析按这张表过滤
KEY_CODE_TO_NAME: dict[int, str] = {
    code: name for name, code in KEY_NAME_TO_CODE.items()
}

# ioctl 编号的拼装参数，见 Linux 内核 <asm-generic/ioctl.h>。
# _IOC(dir, type, nr, size) = dir << 30 | size << 16 | type << 8 | nr
_IOC_READ: int = 2
_IOC_DIRSHIFT: int = 30
_IOC_SIZESHIFT: int = 16
_IOC_TYPESHIFT: int = 8
_EVDEV_IOC_TYPE: int = 0x45  # 'E'

# EVIOCGNAME 的设备名缓冲区大小
DEVICE_NAME_SIZE: int = 256

# EVIOCGBIT(EV_SYN) 拿事件类型位集，EV_MAX 是 0x1f，4 字节够用，留 8 字节
EVENT_BITS_SIZE: int = 8

# EVIOCGBIT(EV_KEY) 拿键位集，KEY_CNT 是 0x300
KEY_BITS_SIZE: int = 0x300 // 8

# stop 时 join 线程的秒数上限，阻塞读可能退不出来，不能无限等
STOP_JOIN_TIMEOUT: float = 1.0


def _ioc_read(number: int, size: int) -> int:
    """拼一个 ``_IOC(_IOC_READ, 'E', number, size)`` 编号。"""
    return (
        (_IOC_READ << _IOC_DIRSHIFT)
        | (_EVDEV_IOC_TYPE << _IOC_TYPESHIFT)
        | number
        | (size << _IOC_SIZESHIFT)
    )


# EVIOCGNAME(len)：取设备名
EVIOCGNAME: int = _ioc_read(0x06, DEVICE_NAME_SIZE)


def _eviocgbit(event_type: int, size: int) -> int:
    """``EVIOCGBIT(event_type, size)`` 的 ioctl 编号。"""
    return _ioc_read(0x20 + event_type, size)


def _bit_is_set(bits: bytes | bytearray, bit: int) -> bool:
    """位集里某一位是不是 1。"""
    index = bit // 8
    if index >= len(bits):
        return False
    return bool(bits[index] & (1 << (bit % 8)))


def is_keyboard_capability(event_bits: bytes, key_bits: bytes) -> bool:
    """事件类型含 EV_KEY 且键集含 KEY_A 才算键盘。

    只含 KEY_POWER 之类的设备（电源键、多媒体键）不算键盘，
    不把它们当成办公键盘去读。
    """
    return _bit_is_set(event_bits, EV_KEY) and _bit_is_set(key_bits, KEY_A)


def _parse_key_event(data: bytes) -> tuple[str, bool] | None:
    """解析一条 ``input_event``，返回 ``(键名, 是否按下)`` 或 None。

    只认 EV_KEY 上的按下与松开；长按重复（value=2）、EV_SYN、长度不对的
    记录、不在支持表里的键一律返回 None。
    """
    if len(data) != INPUT_EVENT_SIZE:
        return None
    _sec, _usec, event_type, code, value = struct.unpack(INPUT_EVENT_FORMAT, data)
    if event_type != EV_KEY:
        return None
    name = KEY_CODE_TO_NAME.get(code)
    if name is None:
        return None
    if value == KEY_PRESS:
        return name, True
    if value == KEY_RELEASE:
        return name, False
    return None


def _close_quietly(fd: int) -> None:
    """关一个 fd，失败也不报错。"""
    try:
        os.close(fd)
    except OSError:
        pass


def _ioctl_read_buffer(fd: int, request: int, size: int) -> bytes:
    """发一个读取型 ioctl，返回被原地写入的缓冲区内容。

    ``fcntl.ioctl`` 的参数是可变缓冲区时，内核把结果写进这个缓冲区，
    返回值是系统调用的整数返回（``EVIOCGNAME`` 是名字长度），
    不是读到的字节；所以这里只把调用当副作用，内容从同一个缓冲区取。
    """
    buf = bytearray(size)
    fcntl.ioctl(fd, request, buf)
    return bytes(buf)


def _device_name(fd: int) -> str:
    """取输入设备的名字，读不到时返回空串。"""
    try:
        raw = _ioctl_read_buffer(fd, EVIOCGNAME, DEVICE_NAME_SIZE)
    except OSError:
        return ""
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _device_capabilities(fd: int) -> tuple[bytes, bytes]:
    """取事件类型位集与键位集，ioctl 失败时抛 OSError。"""
    event_bits = _ioctl_read_buffer(
        fd, _eviocgbit(EV_SYN, EVENT_BITS_SIZE), EVENT_BITS_SIZE
    )
    key_bits = _ioctl_read_buffer(
        fd, _eviocgbit(EV_KEY, KEY_BITS_SIZE), KEY_BITS_SIZE
    )
    return event_bits, key_bits


class EvdevHotkeyProvider:
    """读 ``/dev/input/event*`` 的全局按键提供者。

    :param root: 输入设备目录，测试时注入临时目录，默认 ``/dev/input``。
    """

    def __init__(self, root: str = "/dev/input") -> None:
        self._root = root
        # 停止信号，读取线程每轮检查一次
        self._stop_event = threading.Event()
        # 保护 _fds 与 _threads，stop 与读取线程的收尾都在它下面改
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._fds: set[int] = set()

    def available(self) -> bool:
        """至少有一台能打开、能识别的键盘时返回 True，不抛异常。"""
        try:
            for path in self._event_paths():
                fd = self._open_keyboard(path)
                if fd is None:
                    continue
                _close_quietly(fd)
                return True
        except Exception as exc:
            log_event(
                "input.hotkey.scan_failed",
                "扫描输入设备时出错，当作全局热键不可用",
                level="WARNING",
                root=self._root,
                error=type(exc).__name__,
            )
        return False

    def start(self, emit: Callable[[str, bool], None]) -> None:
        """给每台键盘起一个 daemon 线程读按键。

        没有可用键盘时记一条 WARNING 就返回，不抛异常；上一次没 ``stop``
        就再调 ``start`` 直接返回。
        """
        if self._threads:
            return
        self._stop_event.clear()
        opened = 0
        for path in self._event_paths():
            fd = self._open_keyboard(path)
            if fd is None:
                continue
            thread = threading.Thread(
                target=self._read_loop,
                args=(path, fd, emit),
                name=f"hotkey-{os.path.basename(path)}",
                daemon=True,
            )
            with self._lock:
                self._fds.add(fd)
            try:
                thread.start()
            except RuntimeError as exc:
                self._close_tracked_fd(fd)
                log_event(
                    "input.hotkey.thread_failed",
                    "输入设备读取线程启动失败，跳过",
                    level="WARNING",
                    device=path,
                    error=type(exc).__name__,
                )
                continue
            with self._lock:
                self._threads.append(thread)
            opened += 1
        if opened == 0:
            log_event(
                "input.hotkey.no_device",
                "没有可读取的键盘，全局热键不可用",
                level="WARNING",
                root=self._root,
            )

    def stop(self) -> None:
        """停止所有读取线程并释放设备。重复调用安全。"""
        self._stop_event.set()
        with self._lock:
            fds = list(self._fds)
            threads = list(self._threads)
            self._threads.clear()
        for fd in fds:
            self._close_tracked_fd(fd)
        for thread in threads:
            thread.join(timeout=STOP_JOIN_TIMEOUT)

    def _event_paths(self) -> list[str]:
        """root 下的全部事件设备路径，排序保证结果稳定。"""
        return sorted(glob.glob(os.path.join(self._root, "event*")))

    def _open_keyboard(self, path: str) -> int | None:
        """打开一台键盘，返回 fd；不是键盘或打不开时返回 None 并记日志。"""
        try:
            fd = os.open(path, os.O_RDONLY)
        except PermissionError:
            log_event(
                "input.hotkey.permission_denied",
                "没有权限读取输入设备，跳过",
                level="WARNING",
                device=path,
            )
            return None
        except OSError as exc:
            log_event(
                "input.hotkey.device_unavailable",
                "打不开输入设备，跳过",
                level="WARNING",
                device=path,
                error=type(exc).__name__,
            )
            return None
        name = _device_name(fd)
        try:
            event_bits, key_bits = _device_capabilities(fd)
        except OSError as exc:
            log_event(
                "input.hotkey.capability_failed",
                "读不到输入设备的能力位，跳过",
                level="WARNING",
                device=path,
                error=type(exc).__name__,
            )
            _close_quietly(fd)
            return None
        if not is_keyboard_capability(event_bits, key_bits):
            log_event(
                "input.hotkey.not_keyboard",
                "输入设备不是键盘，跳过",
                level="DEBUG",
                device=path,
                name=name or "-",
            )
            _close_quietly(fd)
            return None
        log_event(
            "input.hotkey.device_opened",
            "读到键盘",
            device=path,
            name=name or "-",
        )
        return fd

    def _close_tracked_fd(self, fd: int) -> None:
        """关闭一个登记过的 fd，只关一次。"""
        with self._lock:
            if fd not in self._fds:
                return
            self._fds.discard(fd)
        _close_quietly(fd)

    def _read_loop(
        self,
        path: str,
        fd: int,
        emit: Callable[[str, bool], None],
    ) -> None:
        """一个设备的读取线程：阻塞读、解析、上报，直到 stop 或设备消失。"""
        try:
            while not self._stop_event.is_set():
                try:
                    data = os.read(fd, INPUT_EVENT_SIZE)
                except OSError as exc:
                    if self._stop_event.is_set():
                        # stop 关掉了 fd，正常退出
                        break
                    log_event(
                        "input.hotkey.read_failed",
                        "读输入设备失败，这台设备不再上报",
                        level="WARNING",
                        device=path,
                        error=type(exc).__name__,
                    )
                    break
                if not data:
                    # 设备被拔掉，read 返回空
                    break
                parsed = _parse_key_event(data)
                if parsed is None:
                    continue
                name, pressed = parsed
                try:
                    emit(name, pressed)
                except Exception as exc:
                    log_event(
                        "input.hotkey.emit_failed",
                        "上报按键时出错，已忽略",
                        level="WARNING",
                        key=name,
                        error=type(exc).__name__,
                    )
        finally:
            self._close_tracked_fd(fd)