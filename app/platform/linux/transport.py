"""Linux 侧的传输层：AF_UNIX socket。

见《9-平台差异与Windows适配》「Windows 进程通信」一节：Linux 侧用 socket 文件
做进程通信，文件权限收紧到 ``600`` 做隔离；Windows 侧换 TCP loopback 加
token，那是后续里程碑的事。

本模块把 :mod:`app.runtime.socket_server` 的监听部分与
:mod:`app.runtime.process_manager` 的 socket 文件管理搬过来，行为与错误文案
保持一致：

- 端点文件默认在 ``runtime/interviewmate.sock``；
- 端点文件存在时先连一次确认有没有活进程占用，有活进程就不能重建；
- 监听创建失败时把半成品的 socket 文件清掉；
- socket 文件创建后立刻把权限收紧到 ``600``。

端点文件的生命周期归本模块管：``start`` 建监听并收紧权限，``stop`` 关监听并删
端点文件。``socket.start`` 与 ``socket.stop`` 这两条日志由
:mod:`app.runtime.socket_server` 记，传输层不重复记。

日志走项目日志管道：统一用 :func:`app.utils.logger.log_event` 写，事件名是点分
小写的 ``socket.*``。不用标准库的 ``logging.getLogger(__name__)``，那条路走不到
项目记录器 ``interviewmate``，日志进不了日志文件也不过脱敏。
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.utils.logger import log_event

__all__ = ["UnixSocketTransport"]


# socket 文件权限：只有属主可读写
_SOCKET_FILE_MODE: int = 0o600

# 探测端点有没有被活进程占用时的连接超时，秒
_PROBE_TIMEOUT: float = 1.0


class UnixSocketTransport:
    """AF_UNIX socket 传输层。

    :param socket_path: socket 文件路径。
    """

    # 后端名
    name: str = "unix"

    # 该能力不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = Path(socket_path)
        self._server: asyncio.AbstractServer | None = None

    @property
    def endpoint_path(self) -> Path:
        """当前监听的 socket 文件路径。"""
        return self._socket_path

    async def start(
        self,
        on_connect: Callable[
            [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]
        ],
    ) -> None:
        """开始监听。

        端点文件被活进程占用的情况由 :meth:`prepare_endpoint` 判断，
        调用方要先拿到 ``True`` 再启动。

        启动失败时把已经建出来的 socket 文件清掉，不留半成品。

        :param on_connect: 有连接进来时调用的回调。
        :raises OSError: 监听创建失败。
        """
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._server = await asyncio.start_unix_server(
                on_connect,
                path=str(self._socket_path),
            )
        except OSError:
            # 启动失败时把已经建出来的 socket 文件清掉，不留半成品。
            # 错误日志由 socket_server 记，传输层不重复记同一条。
            self._remove_socket_file()
            raise
        except BaseException:
            self._remove_socket_file()
            raise
        self.restrict_permissions()

    async def stop(self) -> None:
        """停止监听并删掉 socket 文件。

        重复调用安全。
        """
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception as exc:
                # 关闭监听失败不影响后续删文件
                log_event(
                    "socket.error",
                    "关闭 socket 监听失败",
                    level="ERROR",
                    path=str(self._socket_path),
                    error=str(exc),
                )
        self._remove_socket_file()

    def connect(self, *, timeout: float) -> socket.socket:
        """连到服务端，返回已连接的 socket 对象。

        返回同步 socket，调用方直接用它 ``sendall`` 与 ``recv``。

        :param timeout: 连接超时秒数。
        :raises OSError: 连不上。
        """
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(timeout)
            client.connect(str(self._socket_path))
        except OSError as exc:
            client.close()
            log_event(
                "socket.error",
                "连接 socket 失败",
                level="ERROR",
                path=str(self._socket_path),
                error=str(exc),
            )
            raise
        return client

    def is_endpoint_in_use(self) -> bool:
        """端点是不是被活进程占用。

        文件不存在返回 False；连得上说明有活进程在监听，返回 True；
        连不上（文件不存在、连接被拒、不是 socket 等）说明是残留文件，返回 False。

        :returns: 被活进程占用返回 True。
        """
        if not self._socket_path.exists():
            return False
        return self._probe_connect()

    def prepare_endpoint(self) -> bool:
        """把端点准备好，返回能不能继续启动。

        - 文件不存在：直接返回 True；
        - 文件存在且连得上：说明有活进程占用，返回 False；
        - 文件存在但连不上：是上次退出留下的残留文件，删掉后返回 True。

        :returns: 可以重建 socket 文件时返回 True，被活进程占用时返回 False。
        """
        if not self._socket_path.exists():
            return True
        if self._probe_connect():
            return False
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            # 残留文件删不掉（例如路径是个目录），当作不能重建
            return False
        return True

    def cleanup_endpoint(self) -> None:
        """删掉 socket 文件，不存在也不报错。"""
        self._remove_socket_file()

    def restrict_permissions(self) -> None:
        """把 socket 文件权限收紧到只有属主可读写。

        socket 文件由监听创建，权限按 umask 生成，默认过宽（本机其它用户能连上来）。
        这里显式 chmod 一次；失败只记日志，不影响监听。
        """
        try:
            os.chmod(self._socket_path, _SOCKET_FILE_MODE)
        except OSError as exc:
            log_event(
                "socket.error",
                "设置 socket 文件权限失败",
                level="WARNING",
                path=str(self._socket_path),
                error=str(exc),
            )

    def _probe_connect(self) -> bool:
        """连一次 socket 文件，判断有没有活进程在监听。

        连得上说明有活进程；连不上（文件不存在、连接被拒、不是 socket 等）
        说明是残留文件。
        """
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(_PROBE_TIMEOUT)
            client.connect(str(self._socket_path))
        except OSError:
            return False
        finally:
            client.close()
        return True

    def _remove_socket_file(self) -> None:
        """删掉 socket 文件，不存在也不报错。"""
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log_event(
                "socket.error",
                "删除 socket 文件失败",
                level="WARNING",
                path=str(self._socket_path),
                error=str(exc),
            )
