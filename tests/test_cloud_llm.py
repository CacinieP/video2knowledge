"""Tests for the OpenAI-compatible cloud LLM path in build_knowledge.py.

The local Ollama path is the default and must be untouched. What is worth
locking down here is the failure behaviour: a cloud call that fails must not
become a placeholder-filled document that looks like a finished deliverable.
That exact failure mode is why PR #8 exists for the local path, and it applies
here too.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_knowledge as bk  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cloud():
    """Every test starts and ends with cloud mode off."""
    bk._CLOUD.update(api_base=None, api_model=None, api_key=None)
    yield
    bk._CLOUD.update(api_base=None, api_model=None, api_key=None)


# --------------------------------------------------------------------------
# local path unchanged
# --------------------------------------------------------------------------

def test_local_path_is_the_default():
    assert bk._CLOUD["api_base"] is None
    assert bk.ping("http://localhost:1") is False, "unreachable ollama -> False"


def test_configure_cloud_with_no_base_is_a_no_op():
    bk.configure_cloud("", "m", "K")
    assert bk._CLOUD["api_base"] is None


# --------------------------------------------------------------------------
# key handling — the security-relevant part
# --------------------------------------------------------------------------

def test_missing_key_fails_loudly(monkeypatch):
    """Silently falling back to local Ollama would look like it worked."""
    monkeypatch.delenv("V2K_TEST_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    assert "V2K_TEST_KEY" in str(e.value)


def test_key_is_read_from_env_not_argv(monkeypatch):
    monkeypatch.setenv("V2K_TEST_KEY", "sk-from-env")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    assert bk._CLOUD["api_key"] == "sk-from-env"


def test_trailing_slash_is_stripped(monkeypatch):
    """Otherwise the URL becomes '...//chat/completions'."""
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1/", "m", "V2K_TEST_KEY")
    assert bk._CLOUD["api_base"] == "https://api.example.com/v1"


def test_ping_short_circuits_in_cloud_mode(monkeypatch):
    """No local ollama is involved in cloud mode; probing it would be a false
    negative and block the whole run."""
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    assert bk.ping("http://localhost:1") is True


# --------------------------------------------------------------------------
# think-stripping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("<think>reasoning</think>Answer", "Answer"),
    ("<think>a\nb</think>\nAnswer", "Answer"),
    ("<think>unclosed reasoning", "<think>unclosed reasoning"),
    ("Plain answer", "Plain answer"),
    ("<think>x</think>A<think>y</think>B", "AB"),
])
def test_strip_think(raw, expect):
    assert bk._strip_think(raw) == expect


# --------------------------------------------------------------------------
# ask_llm routing
# --------------------------------------------------------------------------

def test_ask_llm_routes_to_cloud(monkeypatch):
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "cloud-model", "V2K_TEST_KEY")
    called = {}

    def fake(prompt):
        called["prompt"] = prompt
        return "cloud answer"

    monkeypatch.setattr(bk, "ask_llm_cloud", fake)
    assert bk.ask_llm("http://localhost:1", "local-model", "P") == "cloud answer"
    assert called["prompt"] == "P", "host/model must not leak into cloud routing"


def test_ask_llm_unconfigured_cloud_returns_none():
    assert bk.ask_llm_cloud("p") is None


# --------------------------------------------------------------------------
# error visibility
# --------------------------------------------------------------------------

def _http_error(code):
    return urllib.error.HTTPError("u", code, "msg", {}, None)


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_do_not_retry(monkeypatch, code, capsys):
    """A 401 will fail identically on the second attempt; retrying only makes
    the user wait twice as long for the same error."""
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    attempts = []

    def boom(req, timeout=None):
        attempts.append(1)
        raise _http_error(code)

    monkeypatch.setattr(bk.urllib.request, "urlopen", boom)
    monkeypatch.setattr(bk.time, "sleep", lambda s: None)
    assert bk.ask_llm_cloud("p") is None
    assert len(attempts) == 1, f"code {code} should not be retried"
    assert "cloud generate failed" in capsys.readouterr().err


@pytest.mark.parametrize("code", [429, 500, 503])
def test_transient_errors_retry_once(monkeypatch, code, capsys):
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    attempts = []

    def flaky(req, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(code)
        raise AssertionError("unreachable")  # would be a bug if reached

    def raise_second(req, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(code)
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        return _FakeResponse(payload)

    monkeypatch.setattr(bk.urllib.request, "urlopen", raise_second)
    monkeypatch.setattr(bk.time, "sleep", lambda s: None)
    assert bk.ask_llm_cloud("p") == "ok"
    assert len(attempts) == 2


def test_error_is_never_silent(monkeypatch, capsys):
    """The whole point: None must arrive with a reason on stderr."""
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(bk.urllib.request, "urlopen", boom)
    monkeypatch.setattr(bk.time, "sleep", lambda s: None)
    assert bk.ask_llm_cloud("p") is None
    err = capsys.readouterr().err
    assert "cloud generate failed" in err
    assert "connection refused" in err


def test_malformed_response_does_not_crash(monkeypatch, capsys):
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    monkeypatch.setattr(bk.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResponse(b'{"nonsense":1}'))
    assert bk.ask_llm_cloud("p") is None
    assert "cloud generate failed" in capsys.readouterr().err


def test_think_block_never_reaches_the_document(monkeypatch):
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    body = json.dumps(
        {"choices": [{"message": {"content": "<think>inner monologue</think>Real summary"}}]}
    ).encode()
    monkeypatch.setattr(bk.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResponse(body))
    out = bk.ask_llm_cloud("p")
    assert out == "Real summary"
    assert "think" not in out


def test_max_tokens_is_capped(monkeypatch):
    """Unbounded generation on a large subtitle dump is what hung a batch for
    hours on the local path; the cap must carry over."""
    monkeypatch.setenv("V2K_TEST_KEY", "k")
    bk.configure_cloud("https://api.example.com/v1", "m", "V2K_TEST_KEY")
    seen = {}

    def capture(req, timeout=None):
        seen.update(json.loads(req.data.decode()))
        body = json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()
        return _FakeResponse(body)

    monkeypatch.setattr(bk.urllib.request, "urlopen", capture)
    monkeypatch.delenv("V2K_NUM_PREDICT", raising=False)
    bk.ask_llm_cloud("p")
    assert seen["max_tokens"] == 2048
    assert seen["stream"] is False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_api_base_without_api_model_is_rejected(tmp_path):
    """Silently ignoring --api-base would run against local ollama and look
    like the cloud call had worked."""
    sub = tmp_path / "subtitles.json"
    sub.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}),
                   encoding="utf-8")
    script = Path(bk.__file__).resolve()
    r = subprocess.run(
        [sys.executable, str(script), "--subtitles", str(sub),
         "--out-dir", str(tmp_path), "--api-base", "https://api.example.com/v1"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "--api-model" in r.stderr


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
