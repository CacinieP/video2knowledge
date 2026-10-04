"""Cloud VLM path in build_notes.py (--api-base routes 画面 descriptions too).

The text side already routed through build_knowledge.configure_cloud(); before
this, describe_frame() kept hitting the Ollama /api/generate shape with a
base64 "images" array, which no /chat/completions provider understands — a
cloud run silently rendered every node without its frame description.
"""
import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_knowledge as bk  # noqa: E402
import build_notes as bn  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cloud():
    yield
    bk._CLOUD.update(api_base=None, api_model=None, api_key=None)


def test_describe_frame_routes_to_cloud_when_configured(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")
    calls = {}

    def fake_cloud_chat(model, content, max_tokens, temperature):
        calls["model"] = model
        calls["content"] = content
        return "板书列出二次函数顶点公式"

    monkeypatch.setattr(bn, "_cloud_chat", fake_cloud_chat)
    bk._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="glm-5.3-flash", api_key="k")
    # main() resolves vlm_model = --vlm-api-model or --api-model before calling;
    # describe_frame passes that through — the caller decides, not the env
    out = bn.describe_frame("http://localhost:11434", "glm-5.3-flash", jpg)
    assert out == "板书列出二次函数顶点公式"
    assert calls["model"] == "glm-5.3-flash"
    # OpenAI vision shape: image_url part + text prompt part
    kinds = [p["type"] for p in calls["content"]]
    assert kinds == ["image_url", "text"]
    url = calls["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert url.endswith(base64.b64encode(b"x").decode())


def test_describe_frame_stays_local_without_cloud(monkeypatch, tmp_path):
    jpg = tmp_path / "f.jpg"
    jpg.write_bytes(b"x")
    ollama_payload = {}

    def fake_http_generate(host, payload, timeout=420, retries=1):
        ollama_payload.update(payload)
        return {"response": "乐谱显示 C 大调音阶"}

    monkeypatch.setattr(bn, "http_generate", fake_http_generate)
    out = bn.describe_frame("http://localhost:11434", "minicpm-v", jpg)
    assert out == "乐谱显示 C 大调音阶"
    assert "images" in ollama_payload  # ollama shape, untouched


def test_cloud_chat_strips_think_and_retries_429(monkeypatch):
    import urllib.error

    class Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    bodies = [{"choices": [{"message": {"content": "<think>hmm</think>答案"}}]}]
    codes = [429, 200]

    def fake_urlopen(req, timeout=240):
        if codes[0] == 429:
            codes.pop(0)
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, None)
        return Resp(bodies[0])

    monkeypatch.setattr(bn.time, "sleep", lambda s: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    bk._CLOUD.update(api_base="https://example.invalid/v1",
                     api_model="m", api_key="k")
    assert bn._cloud_chat("m", "hi", max_tokens=8, temperature=0.1) == "答案"


def test_cloud_chat_empty_content_raises_not_returns_empty():
    bk._CLOUD.update(api_base="https://x.invalid", api_model="m", api_key="k")

    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": ""}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request
    orig = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda req, timeout=240: Resp()
        with pytest.raises(ValueError):
            bn._cloud_chat("m", "hi", max_tokens=8, temperature=0.1)
    finally:
        urllib.request.urlopen = orig


def test_supports_vision_cloud_false_on_any_failure(monkeypatch):
    def boom(model, content, max_tokens, temperature):
        raise RuntimeError("400 no image support")

    monkeypatch.setattr(bn, "_cloud_chat", boom)
    assert bn.supports_vision_cloud("m") is False


def test_collapse_trailing_visual_nodes():
    secs = [{"t": 1, "desc": "a", "excerpt": "x", "note": "n", "speech": True},
            {"t": 2, "desc": "b", "excerpt": "", "note": "", "speech": True},
            {"t": 3, "desc": "b", "excerpt": "", "note": "", "speech": True},
            {"t": 4, "desc": "b", "excerpt": "", "note": "", "speech": True}]
    out = bn.collapse_trailing_visual_nodes(secs)
    assert [s["t"] for s in out] == [1, 4]  # keeps the final settled frame


def test_collapse_keeps_all_when_no_trailing_run():
    secs = [{"t": i, "desc": "d", "excerpt": "e", "note": "n", "speech": True}
            for i in range(3)]
    assert bn.collapse_trailing_visual_nodes(secs) == secs


def test_collapse_noop_for_wordless_video():
    # every node visual-only by design — collapsing would destroy the note
    secs = [{"t": i, "desc": f"d{i}", "excerpt": "", "note": "", "speech": False}
            for i in range(5)]
    assert bn.collapse_trailing_visual_nodes(secs) == secs


def test_pdf_safe_text_maps_missing_superscripts():
    assert bn.pdf_safe_text("1/y=x²/8+Cx⁻⁶") == "1/y=x^2/8+Cx^(-6)"
    assert bn.pdf_safe_text("yⁿ 次方") == "y^n 次方"
    assert bn.pdf_safe_text("普通中文没有上标") == "普通中文没有上标"
