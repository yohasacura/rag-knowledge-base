"""Shared JSON-RPC 2.0 protocol constants and helpers.

Used by both the daemon (server) and the DaemonClient (client) to ensure
consistent framing, message construction and error codes.

Wire format
-----------
Each message is length-prefixed:

    ┌──────────────────┬────────────────────────┐
    │ 4 bytes (uint32) │ N bytes (UTF-8 JSON)   │
    │ payload length   │ JSON-RPC 2.0 message   │
    └──────────────────┴────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import secrets
import struct
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application-level error codes
ERR_RAG_NOT_FOUND = -1
ERR_RAG_ALREADY_EXISTS = -2
ERR_INDEX_ERROR = -3
ERR_FILE_NOT_FOUND = -4
ERR_AUTH_FAILED = -5

JSONRPC_VERSION = "2.0"

# Default daemon settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9527
DEFAULT_IDLE_TIMEOUT = 300  # seconds

# Header size for length-prefix framing
_HEADER_SIZE = 4
_HEADER_STRUCT = struct.Struct("!I")  # 4-byte big-endian unsigned int

# Maximum message size (16 MiB) — safety limit
MAX_MESSAGE_SIZE = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------


def frame_message(msg: dict) -> bytes:
    """Encode a JSON-RPC message dict into a length-prefixed frame."""
    payload = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    return _HEADER_STRUCT.pack(len(payload)) + payload


def read_frame_sync(sock) -> dict:
    """Read one length-prefixed JSON-RPC frame from a blocking socket.

    Parameters
    ----------
    sock : socket.socket
        A connected blocking TCP socket.

    Returns
    -------
    dict
        Parsed JSON-RPC message.

    Raises
    ------
    ConnectionError
        If the remote side closes the connection.
    ValueError
        If the frame exceeds ``MAX_MESSAGE_SIZE``.
    """
    header = _recv_exactly(sock, _HEADER_SIZE)
    (length,) = _HEADER_STRUCT.unpack(header)
    if length > MAX_MESSAGE_SIZE:
        # Detect stale daemon sending unframed JSON (the 4-byte "header" is
        # actually the start of a JSON object like '{"me...').
        if header[:1] == b"{":
            raise ConnectionError(
                "Daemon is sending unframed JSON — likely a stale daemon "
                "from an older version. Stop it with 'rag-kb daemon stop' "
                "and retry."
            )
        raise ValueError(f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})")
    payload = _recv_exactly(sock, length)
    return json.loads(payload.decode("utf-8"))


def _recv_exactly(sock, n: int) -> bytes:
    """Read exactly *n* bytes from a blocking socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


async def read_frame_async(reader: asyncio.StreamReader) -> dict:
    """Read one length-prefixed JSON-RPC frame from an asyncio StreamReader.

    Raises
    ------
    ConnectionError
        If EOF is reached before the full frame is read.
    ValueError
        If the frame exceeds ``MAX_MESSAGE_SIZE``.
    """
    header = await reader.readexactly(_HEADER_SIZE)
    (length,) = _HEADER_STRUCT.unpack(header)
    if length > MAX_MESSAGE_SIZE:
        if header[:1] == b"{":
            raise ConnectionError("Client is sending unframed JSON — protocol mismatch.")
        raise ValueError(f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})")
    payload = await reader.readexactly(length)
    return json.loads(payload.decode("utf-8"))


async def write_frame_async(writer: asyncio.StreamWriter, msg: dict) -> None:
    """Write a length-prefixed JSON-RPC frame to an asyncio StreamWriter."""
    writer.write(frame_message(msg))
    await writer.drain()


# ---------------------------------------------------------------------------
# JSON-RPC message constructors
# ---------------------------------------------------------------------------


def make_request(method: str, params: dict | None = None, request_id: str | None = None) -> dict:
    """Build a JSON-RPC 2.0 request."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:12]
    msg: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "id": request_id,
    }
    if params is not None:
        msg["params"] = params
    return msg


def make_response(request_id: str, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "result": result,
        "id": request_id,
    }


def make_error(request_id: str | None, code: int, message: str, data: Any = None) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": JSONRPC_VERSION,
        "error": err,
        "id": request_id,
    }


def make_notification(method: str, params: dict) -> dict:
    """Build a JSON-RPC 2.0 notification (no ``id``)."""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params,
    }


def make_progress(request_id: str, current: int, total: int, message: str = "") -> dict:
    """Build a progress notification linked to a specific request."""
    return make_notification(
        "progress",
        {
            "request_id": request_id,
            "current": current,
            "total": total,
            "message": message,
        },
    )


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


class RpcError(Exception):
    """Raised client-side when the daemon returns a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------


def _get_token_path() -> Path:
    """Return the path to the daemon auth token file."""
    from rag_kb.config import DATA_DIR

    return DATA_DIR / "daemon.token"


def generate_auth_token() -> str:
    """Generate a cryptographically random auth token and write it to disk.

    The token file is only readable by the current user (best-effort on
    Windows).  Returns the generated token string.
    """
    token = secrets.token_hex(32)
    token_path = _get_token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")

    # Restrict file permissions (owner-only on Unix)
    try:
        import stat

        token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, AttributeError):
        pass  # Best-effort; Windows doesn't support POSIX permissions

    return token


def read_auth_token() -> str | None:
    """Read the auth token from disk.  Returns None if file doesn't exist."""
    token_path = _get_token_path()
    if not token_path.exists():
        return None
    return token_path.read_text(encoding="utf-8").strip()


def remove_auth_token() -> None:
    """Remove the auth token file (cleanup on daemon shutdown)."""
    try:
        _get_token_path().unlink(missing_ok=True)
    except OSError:
        pass
