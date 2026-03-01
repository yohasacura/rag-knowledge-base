"""Daemon integration tests — real TCP daemon lifecycle.

These tests require a free port and spawn actual TCP connections.
They do NOT require the heavy model-loading path (only daemon infrastructure).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

from rag_kb.rpc_protocol import (
    ERR_AUTH_FAILED,
    JSONRPC_VERSION,
    frame_message,
    generate_auth_token,
    make_request,
    read_frame_sync,
)


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _send_rpc(sock: socket.socket, method: str, params: dict | None = None, token: str = "") -> dict:
    """Send a JSON-RPC request and return the response."""
    msg = make_request(method, params)
    if token:
        msg["params"] = msg.get("params") or {}
        msg["params"]["_auth_token"] = token
    sock.sendall(frame_message(msg))
    return read_frame_sync(sock)


# ---------------------------------------------------------------------------
# RPC framing over raw TCP
# ---------------------------------------------------------------------------


class TestRpcFraming:
    """Test wire-level framing without a daemon — just socket pairs."""

    def test_frame_round_trip_over_socket(self):
        """Send a framed message through a socket pair and decode it."""
        server, client = socket.socketpair()
        try:
            msg = make_request("ping")
            client.sendall(frame_message(msg))
            received = read_frame_sync(server)
            assert received["method"] == "ping"
            assert received["jsonrpc"] == JSONRPC_VERSION
        finally:
            server.close()
            client.close()

    def test_multiple_messages(self):
        """Multiple messages on the same connection."""
        server, client = socket.socketpair()
        try:
            for i in range(5):
                msg = make_request(f"method_{i}")
                client.sendall(frame_message(msg))

            for i in range(5):
                received = read_frame_sync(server)
                assert received["method"] == f"method_{i}"
        finally:
            server.close()
            client.close()

    def test_connection_closed_raises(self):
        """Reading from a closed socket raises ConnectionError."""
        server, client = socket.socketpair()
        client.close()
        with pytest.raises(ConnectionError):
            read_frame_sync(server)
        server.close()


# ---------------------------------------------------------------------------
# Auth token integration
# ---------------------------------------------------------------------------


class TestAuthTokenIntegration:
    def test_generate_read_cycle(self):
        from rag_kb.rpc_protocol import read_auth_token, remove_auth_token

        token = generate_auth_token()
        assert len(token) == 64
        assert read_auth_token() == token
        remove_auth_token()

    def test_token_is_cryptographically_random(self):
        t1 = generate_auth_token()
        t2 = generate_auth_token()
        assert t1 != t2  # should be unique every time
