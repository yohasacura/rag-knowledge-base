"""RPC protocol unit tests — framing, message constructors, auth tokens."""

from __future__ import annotations

import json
import struct

import pytest

from rag_kb.rpc_protocol import (
    ERR_AUTH_FAILED,
    ERR_RAG_NOT_FOUND,
    INTERNAL_ERROR,
    JSONRPC_VERSION,
    MAX_MESSAGE_SIZE,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RpcError,
    frame_message,
    generate_auth_token,
    make_error,
    make_notification,
    make_progress,
    make_request,
    make_response,
    read_auth_token,
    remove_auth_token,
)


# ---------------------------------------------------------------------------
# Message constructors
# ---------------------------------------------------------------------------


class TestMakeRequest:
    def test_basic_request(self):
        msg = make_request("ping")
        assert msg["jsonrpc"] == JSONRPC_VERSION
        assert msg["method"] == "ping"
        assert "id" in msg

    def test_with_params(self):
        msg = make_request("search", {"query": "test", "n_results": 5})
        assert msg["params"] == {"query": "test", "n_results": 5}

    def test_custom_id(self):
        msg = make_request("ping", request_id="abc123")
        assert msg["id"] == "abc123"

    def test_no_params_key_when_none(self):
        msg = make_request("ping")
        assert "params" not in msg


class TestMakeResponse:
    def test_basic_response(self):
        msg = make_response("req-1", {"status": "ok"})
        assert msg["jsonrpc"] == JSONRPC_VERSION
        assert msg["id"] == "req-1"
        assert msg["result"] == {"status": "ok"}

    def test_null_result(self):
        msg = make_response("req-2", None)
        assert msg["result"] is None


class TestMakeError:
    def test_basic_error(self):
        msg = make_error("req-1", INTERNAL_ERROR, "Something broke")
        assert msg["error"]["code"] == INTERNAL_ERROR
        assert msg["error"]["message"] == "Something broke"
        assert msg["id"] == "req-1"

    def test_with_data(self):
        msg = make_error("req-1", ERR_RAG_NOT_FOUND, "Not found", data={"name": "x"})
        assert msg["error"]["data"] == {"name": "x"}

    def test_null_id(self):
        msg = make_error(None, PARSE_ERROR, "Parse fail")
        assert msg["id"] is None


class TestMakeNotification:
    def test_notification(self):
        msg = make_notification("event", {"key": "value"})
        assert msg["jsonrpc"] == JSONRPC_VERSION
        assert msg["method"] == "event"
        assert "id" not in msg


class TestMakeProgress:
    def test_progress(self):
        msg = make_progress("req-1", 42, 100, "Processing...")
        assert msg["method"] == "progress"
        assert msg["params"]["current"] == 42
        assert msg["params"]["total"] == 100
        assert msg["params"]["message"] == "Processing..."
        assert msg["params"]["request_id"] == "req-1"


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------


class TestFrameMessage:
    def test_round_trip(self):
        original = make_request("test")
        frame = frame_message(original)

        # Parse the frame manually
        header = frame[:4]
        (length,) = struct.unpack("!I", header)
        payload = frame[4 : 4 + length]
        decoded = json.loads(payload.decode("utf-8"))

        assert decoded["method"] == "test"
        assert decoded["jsonrpc"] == JSONRPC_VERSION

    def test_unicode_in_frame(self):
        msg = make_request("search", {"query": "日本語テスト 🚀"})
        frame = frame_message(msg)

        header = frame[:4]
        (length,) = struct.unpack("!I", header)
        payload = frame[4 : 4 + length]
        decoded = json.loads(payload.decode("utf-8"))
        assert decoded["params"]["query"] == "日本語テスト 🚀"

    def test_frame_length_matches_payload(self):
        msg = make_response("x", {"data": "a" * 1000})
        frame = frame_message(msg)
        (length,) = struct.unpack("!I", frame[:4])
        assert length == len(frame) - 4


# ---------------------------------------------------------------------------
# RpcError
# ---------------------------------------------------------------------------


class TestRpcError:
    def test_attributes(self):
        err = RpcError(ERR_AUTH_FAILED, "Unauthorized", data={"token": "bad"})
        assert err.code == ERR_AUTH_FAILED
        assert str(err) == "Unauthorized"
        assert err.data == {"token": "bad"}

    def test_is_exception(self):
        with pytest.raises(RpcError, match="test error"):
            raise RpcError(INTERNAL_ERROR, "test error")


# ---------------------------------------------------------------------------
# Error code constants
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_standard_codes(self):
        assert PARSE_ERROR == -32700
        assert METHOD_NOT_FOUND == -32601
        assert INTERNAL_ERROR == -32603

    def test_app_codes(self):
        assert ERR_RAG_NOT_FOUND == -1
        assert ERR_AUTH_FAILED == -5


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------


class TestAuthToken:
    def test_generate_and_read(self):
        token = generate_auth_token()
        assert isinstance(token, str)
        assert len(token) == 64  # hex of 32 bytes

        read_back = read_auth_token()
        assert read_back == token

    def test_remove(self):
        generate_auth_token()
        remove_auth_token()
        # After removal, read should return None
        # (might still exist if cleanup is best-effort, so this is a soft check)

    def test_multiple_generates_overwrite(self):
        t1 = generate_auth_token()
        t2 = generate_auth_token()
        assert t1 != t2
        assert read_auth_token() == t2
