"""Tests for scripts/net.py — the dead-proxy guard and rotating token.

These cover a failure that is silent by nature: when a system-wide proxy
variable points at a port nothing is listening on, every cloud call dies with
WinError 10061 before leaving the machine, and the batch simply stops
producing. There is no traceback to follow, so the guard needs its own proof.
"""
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import net  # noqa: E402

KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


@pytest.fixture
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _live_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_no_proxy_configured_is_not_dead(clean_env):
    assert net.dead_proxy_in_env() is False


def test_dead_port_is_detected(clean_env):
    clean_env.setenv("HTTPS_PROXY", "http://127.0.0.1:3067")
    assert net.dead_proxy_in_env() is True


def test_live_proxy_is_preserved(clean_env):
    s, port = _live_port()
    try:
        clean_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
        assert net.dead_proxy_in_env() is False
    finally:
        s.close()


def test_unparsable_value_is_conservative(clean_env):
    clean_env.setenv("HTTPS_PROXY", "not-a-url")
    assert net.dead_proxy_in_env() is False


def test_http_proxy_also_considered(clean_env):
    clean_env.setenv("HTTP_PROXY", "http://127.0.0.1:3067")
    assert net.dead_proxy_in_env() is True


def test_pop_dead_proxy_strips_and_returns(clean_env):
    """A popped var must be restorable, or one dead call poisons the process."""
    import os
    clean_env.setenv("HTTPS_PROXY", "http://127.0.0.1:3067")
    saved = net.pop_dead_proxy()
    assert "HTTPS_PROXY" not in os.environ
    os.environ.update(saved)
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:3067"


def test_pop_dead_proxy_is_a_noop_when_proxy_is_live(clean_env):
    import os
    s, port = _live_port()
    try:
        clean_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
        assert net.pop_dead_proxy() == {}
        assert os.environ["HTTPS_PROXY"] == f"http://127.0.0.1:{port}"
    finally:
        s.close()


def test_pop_dead_proxy_is_a_noop_without_proxy(clean_env):
    assert net.pop_dead_proxy() == {}


def test_live_access_token_prefers_fresh_store(clean_env, tmp_path, monkeypatch):
    import time
    store = tmp_path / "auth.json"
    store.write_text(json.dumps({"records": {"x": {
        "accessToken": "fresh-token",
        "expiresAtMs": int(time.time() * 1000) + 3_600_000,
    }}}), encoding="utf-8")
    monkeypatch.setenv("MINIMAX_AUTH_STORE", str(store))
    clean_env.setenv("MINIMAX_API_KEY", "env-fallback")
    assert net.live_access_token() == "fresh-token"


def test_live_access_token_falls_back_when_stale(clean_env, tmp_path, monkeypatch):
    import time
    store = tmp_path / "auth.json"
    store.write_text(json.dumps({"records": {"x": {
        "accessToken": "stale",
        "expiresAtMs": int(time.time() * 1000) - 1,
    }}}), encoding="utf-8")
    monkeypatch.setenv("MINIMAX_AUTH_STORE", str(store))
    clean_env.setenv("MINIMAX_API_KEY", "env-fallback")
    assert net.live_access_token() == "env-fallback"


def test_live_access_token_survives_missing_store(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_AUTH_STORE", str(tmp_path / "nope.json"))
    clean_env.setenv("MINIMAX_API_KEY", "env-fallback")
    assert net.live_access_token() == "env-fallback"


def test_cloud_auth_header_static_mode_uses_key(clean_env):
    clean_env.delenv("MINIMAX_API_KEY", raising=False)
    assert net.cloud_auth_header(False, "sk-static") == "Bearer sk-static"
