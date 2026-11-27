"""Windows 侧的传输层：TCP loopback + token。

见《9-平台差异与Windows适配》「Windows 进程通信」一节：Windows 侧监听
``127.0.0.1`` 上的随机端口，端口号写进端点文件（``runtime/interviewmate.port``），
随机 token 写进同目录的 ``interviewmate.token``，客户端每次连接都要先发一行
token。socket 文件在 Linux 侧承担的文件权限隔离，在 Windows 侧由 token 承担。

本模块把 :mod:`app.runtime.socket_server` 的监听部分与
:mod:`app.runtime.process_manager` 的端点文件管理搬过来，行为与错误文案
与 Linux 侧（:mod:`app.platform.linux.transport`）保持一致，差别只在端点
形态与隔离手段：

- 端点文件里写的是端口号，token 写进同目录的 ``.token`` 文件；
- token 握手在传输层做：监听侧先把 ``on_connect`` 包一层，先读一行比对
  token，不匹配（含占用探测那种连上就断的连接）直接断开，协议层无感知；
- 客户端侧 ``connect`` 连上后先发一行 token 再返回 socket，调用方照旧
  发 JSON；
- Windows 没有 POSIX 权限位，:meth:`restrict_permissions` 是空操作，隔离
  完全由 token 承担。

端点文件与 token 文件的生命周期归本模块管：``start`` 建监听并写两个文件，
``stop`` 关监听并删两个文件。``socket.start`` 与 ``socket.stop`` 这两条日志
由 :mod:`app.runtime.socket_server` 记，传输层不重复记。

日志走项目日志管道：统一用 :func:`app.utils.logger.log_event` 写，事件名是
点分小写的 ``socket.*``。不用标准库的 ``logging.getLogger(__name__)``，那条
路走不到项目记录器 ``interviewmate``，日志进不了日志文件也不过脱敏。
"""

from __future__ import annotations

import asyncio
import secrets
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.utils.logger import log_event

__all__ = ["TcpLoopbackTransport"]


# 监听与连接都固定在本机回环地址
_HOST: str = "127.0.0.1"

# token 文件的扩展名：端点文件去掉后缀换这个
_TOKEN_FILE_SUFFIX: str = ".token"

# 探测端点有没有被活进程占用时的连接超时，秒
_PROBE_TIMEOUT: float = 1.0


class TcpLoopbackTransport:
    """TCP loopback 传输层。

    :param endpoint_path: 端口文件的路径，token 文件在同目录、后缀换 ``.token``。
    """

    # 后端名
    name: str = "tcp-loopback"

    # 该能力不需要额外第三方包
    required_packages: tuple[str, ...] = ()

    def __init__(self, endpoint_path: Path) -> None:
        self._endpoint_path = Path(endpoint_path)
        self._token_path = self._endpoint_path.with_suffix(_TOKEN_FILE_SUFFIX)
        self._server: asyncio.AbstractServer | None = None
        self._token: str = ""

    @property
    def endpoint_path(self) -> Path:
        """当前监听的端口文件路径。"""
        return self._endpoint_path

    @property
    def token_path(self) -> Path:
        """当前使用的 token 文件路径。"""
        return self._token_path

    async def start(
        self,
        on_connect: Callable[
            [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]
        ],
    ) -> None:
        """开始监听，并把端口号与 token 写进文件。

        端点文件被活进程占用的情况由 :meth:`prepare_endpoint` 判断，
        调用方要先拿到 ``True`` 再启动。

        启动失败时把已经建出来的监听与文件清掉，不留半成品。

        :param on_connect: 有连接进来时调用的回调；本层会先做 token 握手，
            通过后才调用它。
        :raises OSError: 监听创建失败或端点文件写入失败。
        """
        self._endpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._token = secrets.token_hex(32)
        try:
            self._server = await asyncio.start_server(
                self._wrap_on_connect(on_connect),
                host=_HOST,
                port=0,
            )
        except BaseException:
            self._remove_files()
            raise
        try:
            port = self._server.sockets[0].getsockname()[1]
            self._endpoint_path.write_text(f"{port}\n", encoding="utf-8")
            self._token_path.write_text(f"{self._token}\n", encoding="utf-8")
        except BaseException:
            server, self._server = self._server, None
            server.close()
            self._remove_files()
            raise
        self.restrict_permissions()

    async def stop(self) -> None:
        """停止监听并删掉端口文件与 token 文件。

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
                    "关闭通信监听失败",
                    level="ERROR",
                    path=str(self._endpoint_path),
                    error=str(exc),
                )
        self._remove_files()

    def connect(self, *, timeout: float) -> socket.socket:
        """连到服务端，发过 token 后返回已连接的 socket 对象。

        返回同步 socket，token 握手在这里做掉，调用方照旧直接发 JSON。

        :param timeout: 连接超时秒数。
        :raises OSError: 端口文件不存在或不可读、连不上、发送 token 失败。
        """
        port = self._read_port()
        token = self._read_token()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.settimeout(timeout)
            client.connect((_HOST, port))
            client.sendall((token + "\n").encode("utf-8"))
        except OSError as exc:
            client.close()
            log_event(
                "socket.error",
                "连接通信端口失败",
                level="ERROR",
                path=str(self._endpoint_path),
                error=str(exc),
            )
            raise
        return client

    def is_endpoint_in_use(self) -> bool:
        """端点是不是被活进程占用。

        端口文件不存在返回 False；连得上说明有活进程在监听，返回 True；
        连不上（文件不存在、连接被拒、内容不是端口号等）说明是残留文件，
        返回 False。

        :returns: 被活进程占用返回 True。
        """
        if not self._endpoint_path.exists():
            return False
        return self._probe_connect()

    def prepare_endpoint(self) -> bool:
        """把端点准备好，返回能不能继续启动。

        - 端口文件不存在：直接返回 True；
        - 端口文件存在且连得上：说明有活进程占用，返回 False；
        - 端口文件存在但连不上：是上次退出留下的残留文件，删掉端口文件
          与 token 文件后返回 True。

        :returns: 可以重建端点时返回 True，被活进程占用时返回 False。
        """
        if not self._endpoint_path.exists():
            return True
        if self._probe_connect():
            return False
        try:
            self._endpoint_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # 残留文件删不掉（例如路径是个目录），当作不能重建
            return False
        self._remove_token_file()
        return True

    def cleanup_endpoint(self) -> None:
        """删掉端口文件与 token 文件，不存在也不报错。"""
        self._remove_files()

    def restrict_permissions(self) -> None:
        """收紧端点文件的权限。

        Windows 没有 POSIX 权限位，这里是空操作：监听只绑 ``127.0.0.1``，
        本机其它用户的隔离由 token 承担。
        """

    def _wrap_on_connect(
        self,
        on_connect: Callable[
            [asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]
        ],
    ) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
        """给 ``on_connect`` 包一层 token 握手。

        先读一行比对 token：空行（连上就断的占用探测）或不匹配都直接断开，
        协议层无感知；匹配才把连接交给 ``on_connect``。
        """

        async def handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            try:
                line = await reader.readline()
            except (ConnectionResetError, BrokenPipeError, OSError):
                # 对端连上就断（占用探测）或握手时网络出错，直接收尾
                await self._close_writer(writer)
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not line or text != self._token:
                log_event(
                    "socket.error",
                    "token 校验未通过，断开连接",
                    level="WARNING",
                    path=str(self._endpoint_path),
                )
                await self._close_writer(writer)
                return
            await on_connect(reader, writer)

        return handler

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter) -> None:
        """关闭连接，写不出去时的错误忽略。"""
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass

    def _read_port(self) -> int:
        """读端口文件里的端口号。

        :raises OSError: 文件不存在、读不了或内容不是合法端口号。
        """
        try:
            text = self._endpoint_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"端口文件不存在或不可读：{self._endpoint_path}（{exc}）") from exc
        try:
            port = int(text)
        except ValueError as exc:
            raise OSError(f"端口文件内容不是合法端口号：{self._endpoint_path}") from exc
        if port <= 0 or port > 65535:
            raise OSError(f"端口文件内容不是合法端口号：{self._endpoint_path}")
        return port

    def _read_token(self) -> str:
        """读 token 文件里的 token。

        :raises OSError: 文件不存在或读不了。
        """
        try:
            return self._token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(f"token 文件不存在或不可读：{self._token_path}（{exc}）") from exc

    def _probe_connect(self) -> bool:
        """连一次端口文件里的端口，判断有没有活进程在监听。

        连得上说明有活进程；连不上（文件不存在、连接被拒、内容不是端口号等）
        说明是残留文件。
        """
        try:
            port = self._read_port()
        except OSError:
            return False
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.settimeout(_PROBE_TIMEOUT)
            client.connect((_HOST, port))
        except OSError:
            return False
        finally:
            client.close()
        return True

    def _remove_files(self) -> None:
        """删掉端口文件与 token 文件，不存在也不报错。"""
        try:
            self._endpoint_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log_event(
                "socket.error",
                "删除端口文件失败",
                level="WARNING",
                path=str(self._endpoint_path),
                error=str(exc),
            )
        self._remove_token_file()

    def _remove_token_file(self) -> None:
        """删掉 token 文件，不存在也不报错。"""
        try:
            self._token_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log_event(
                "socket.error",
                "删除 token 文件失败",
                level="WARNING",
                path=str(self._token_path),
                error=str(exc),
            )
