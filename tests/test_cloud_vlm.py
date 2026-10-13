"""Cloud VLM path in mm_caption.py (--api-base takes Path 1 to the cloud).

Before this, only the text stages (build_knowledge) and frame descriptions
(build_notes) could leave Ollama; Path 1 — silent videos, slide OCR — was
local-only. These tests pin the routing, the OpenAI vision payload shape, the
think-tag/newline handling differences between caption and OCR mode, and the
retry-on-transient-error behaviour.
"""
import base64
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mm_caption as mc  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cloud():
    yield
    mc._CLOUD.update(api_base=None, api_model=None, api_key=None)


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_configure_cloud_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="API key"):
        mc.configure_cloud("https://example.invalid/v1", None, None)


def test_configure_cloud_defaults_to_v47_hosted_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    mc.configure_cloud("https://example.invalid/v1", None, None)
    assert mc._CLOUD["api_model"] == "MiniCPM-V-4.7-1B"
    assert mc._CLOUD["api_base"] == "https://example.invalid/v1"


def test_caption_frame_routes_to_cloud_when_configured(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["payload"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {"content": "板书：二次函数顶点公式"}}]})

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    mc._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="MiniCPM-V-4.7-1B", api_key="k")
    out = mc.caption_frame("http://localhost:11434", "ignored-in-cloud", jpg,
                           mc.PROMPT, ocr=False)
    assert out == "板书：二次函数顶点公式"
    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    # OpenAI vision shape: text + image_url parts, data-URL prefixed
    content = seen["payload"]["messages"][0]["content"]
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert content[0]["text"] == mc.PROMPT
    url = content[1]["image_url"]["url"]
    assert url == f"data:image/jpeg;base64,{base64.b64encode(b'x').decode()}"
    assert seen["payload"]["max_tokens"] == 160  # caption mode stays short


def test_ocr_mode_gets_big_budget_and_keeps_newlines(monkeypatch, tmp_path):
    jpg = tmp_path / "f.png"
    jpg.write_bytes(b"x")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {"content":
            "<think>reading</think>| a | b |\n| 1 | 2 |"}}]})

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    mc._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="MiniCPM-V-4.7-1B", api_key="k")
    out = mc.caption_frame("http://localhost:11434", "ignored", jpg,
                           mc.PROMPT_OCR, ocr=True)
    assert seen["payload"]["max_tokens"] == 1024
    # think chain stripped, Markdown table newline survives (OCR mode)
    assert out == "| a | b |\n| 1 | 2 |"
    assert seen["payload"]["messages"][0]["content"][1]["image_url"]["url"] \
        .startswith("data:image/png;base64,")


def test_caption_mode_collapses_newlines(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")

    def fake_urlopen(req, timeout=None):
        return _Resp({"choices": [{"message": {"content": "一行\n第二行"}}]})

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    mc._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="MiniCPM-V-4.7-1B", api_key="k")
    out = mc.caption_frame("http://localhost:11434", "ignored", jpg,
                           mc.PROMPT, ocr=False)
    assert out == "一行 第二行"


def test_cloud_retries_transient_errors(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("503 from the gateway")
        return _Resp({"choices": [{"message": {"content": "第二次成了"}}]})

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    mc._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="MiniCPM-V-4.7-1B", api_key="k")
    out = mc.caption_frame("http://localhost:11434", "ignored", jpg,
                           mc.PROMPT, ocr=False)
    assert out == "第二次成了"
    assert calls["n"] == 2


def test_caption_frame_stays_local_without_cloud(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")
    ollama_payload = {}

    def fake_http_json(url, payload, timeout=420, retries=1):
        ollama_payload.update(payload)
        return {"response": "乐谱显示 C 大调音阶"}

    monkeypatch.setattr(mc, "http_json", fake_http_json)
    out = mc.caption_frame("http://localhost:11434", "openbmb/minicpm-v4.7:latest",
                           jpg, mc.PROMPT, ocr=False)
    assert out == "乐谱显示 C 大调音阶"
    assert "images" in ollama_payload  # ollama shape, untouched
    assert ollama_payload["think"] is False
