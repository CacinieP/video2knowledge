"""Shared network helpers for the cloud-backed scripts.

Every cloud call in this package goes through :func:`urlopen_direct`, because
the failure it prevents is both common and deeply misleading.

Problem: urllib honours HTTP_PROXY/HTTPS_PROXY from the process environment.
A machine can have a system-wide proxy variable pointing at a local port that
is only sometimes listening (127.0.0.1:3067 here). When that proxy is down,
*every* outgoing request dies with WinError 10061 "connection actively
refused" before a single byte leaves the host. That reads exactly like an
upstream outage, and it does not raise anything unusual -- the batch just
stops producing. Measured cost of getting this wrong: 78 ASR failures and 88
download failures over two silent hours.

The fix is deliberately *not* "unconditionally strip the proxy": a live proxy
is a legitimate configuration and tearing it down would break working setups.
So probe the port first, and only bypass when nothing is listening.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from pathlib import Path

PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

# The desktop agent endpoint authenticates with a short-lived OAuth token
# (audience agent-backend, scope agent.default) that the runtime rotates every
# ~40 minutes. An environment variable cannot carry that: a long batch would
# authenticate once and start 401ing hours later. Re-reading the store per call
# costs a few hundred bytes and keeps unattended runs alive. A real static API
# key in the environment still works, so this is a fallback, not a requirement.
DEFAULT_AUTH_STORE = Path.home() / ".minimax" / "auth" / "prod" / "cn" / \
    "mcode-public" / "auth.json"


def live_access_token(fallback_env: str = "MINIMAX_API_KEY") -> str:
    """Return a currently-valid rotating token, else the env fallback."""
    store = os.environ.get("MINIMAX_AUTH_STORE") or str(DEFAULT_AUTH_STORE)
    try:
        rec = next(iter(json.loads(Path(store).read_text(
            encoding="utf-8"))["records"].values()))
        # 60s of margin: a token that expires mid-upload fails the whole call.
        if int(rec.get("expiresAtMs", 0)) - int(time.time() * 1000) > 60_000:
            return rec["accessToken"]
    except Exception:
        pass
    return os.environ.get(fallback_env, "")


def cloud_auth_header(anthropic_mode: bool, static_key: str) -> str:
    """Authorization value for a cloud call.

    In anthropic mode the credential is the rotating desktop token; otherwise
    the caller's own static key is used unchanged. Kept here so build_notes.py
    and build_knowledge.py do not have to import each other, which had become a
    circular import.
    """
    if anthropic_mode:
        return f"Bearer {live_access_token()}"
    return f"Bearer {static_key}"


def dead_proxy_in_env() -> bool:
    """True when a proxy is configured but nothing is listening on its port.

    Conservative on purpose: anything unparsable or uncertain returns False so
    that a working proxy is never removed. Callers that treat "False" as
    "no proxy configured" will simply not touch the environment.
    """
    for key in PROXY_ENV_VARS:
        val = os.environ.get(key, "")
        m = re.search(r":(\d{2,5})(?:/|$)", val)
        if not m:
            continue
        try:
            with socket.create_connection(("127.0.0.1", int(m.group(1))), timeout=1.5):
                return False        # something is listening: keep the proxy
        except OSError:
            return True             # nothing there: the proxy is dead
    return False


def pop_dead_proxy() -> dict[str, str]:
    """Strip proxy vars *only* when the proxy is dead. Returns what was removed.

    Callers must restore the returned dict (usually in a ``finally``). Kept as a
    function rather than a context manager on purpose: the cloud scripts use the
    module-level ``urllib.request.urlopen`` because their tests monkeypatch it,
    and an opener-based helper would silently defeat every stub.
    """
    if not dead_proxy_in_env():
        return {}
    return {k: os.environ.pop(k) for k in PROXY_ENV_VARS if k in os.environ}
