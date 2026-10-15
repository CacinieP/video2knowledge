"""Batch state regressions without models, media decoding or network calls."""
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import batch_run as batch


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    root = tmp_path / "videos"
    root.mkdir()
    video = root / "lesson.mp4"
    video.write_bytes(b"video fixture")
    out = tmp_path / "out"
    state = {"asr": [], "build": [], "fail_asr": False, "fail_build": False}

    class FakeBatch(batch.Batch):
        def stage_a(self, video, run_dir):
            if (run_dir / ".asr_done").exists():
                return
            state["asr"].append(video.name)
            if state["fail_asr"]:
                raise RuntimeError("ASR outage")
            (run_dir / "subtitles.json").write_text(json.dumps({
                "segments": [{"start": 0, "end": 60, "text": "lesson"}]}))
            (run_dir / ".asr_done").touch()

        def stage_b(self, video, run_dir):
            state["build"].append(video.name)
            if state["fail_build"]:
                raise RuntimeError("model outage")
            (run_dir / "knowledge.md").write_text("lesson")
            (run_dir / "cards.apkg").write_bytes(b"deck fixture")
            (run_dir / ".done").touch()
            self.finish(video, run_dir, "done", [0, 0, 0])

    monkeypatch.setattr(batch, "Batch", FakeBatch)
    # Keep actual worker threads and coordination, but shorten the polling tick.
    monkeypatch.setattr(batch.time, "sleep", lambda _: threading.Event().wait(.001))

    def run(*options):
        monkeypatch.setattr(sys, "argv", ["batch_run", "--root", str(root),
                            "--out-root", str(out), "--text-model", "text-v1",
                            *options])
        return batch.main()

    return run, state, video, out


def test_retry_failed_build_reuses_asr_and_clears_failure(pipeline):
    run, state, _, out = pipeline
    state["fail_build"] = True
    assert run() == 1
    state["fail_build"] = False
    assert run() == 0
    assert len(state["asr"]) == 1
    assert len(state["build"]) == 2
    assert not list(out.rglob(".failed"))
    assert len(list(out.rglob(".done"))) == 1


def test_retry_failed_asr_is_counted_and_eventually_builds(pipeline):
    run, state, _, _ = pipeline
    state["fail_asr"] = True
    assert run() == 1
    assert not state["build"]
    state["fail_asr"] = False
    assert run() == 0
    assert len(state["asr"]) == 2
    assert len(state["build"]) == 1


def test_unchanged_completed_video_is_skipped(pipeline):
    run, state, _, _ = pipeline
    assert run() == run() == 0
    assert len(state["asr"]) == len(state["build"]) == 1


def test_text_model_change_rebuilds_without_retranscribing(pipeline):
    run, state, _, _ = pipeline
    assert run() == 0
    assert run("--text-model", "text-v2") == 0
    assert len(state["asr"]) == 1
    assert len(state["build"]) == 2


def test_asr_model_change_retranscribes(pipeline):
    run, state, _, _ = pipeline
    assert run() == 0
    assert run("--asr-model", "medium") == 0
    assert len(state["asr"]) == len(state["build"]) == 2


def test_hotword_file_edit_retranscribes(pipeline, tmp_path):
    run, state, _, _ = pipeline
    terms = tmp_path / "terms.txt"
    terms.write_text("first")
    assert run("--hotwords", "@" + str(terms)) == 0
    terms.write_text("second")
    assert run("--hotwords", "@" + str(terms)) == 0
    assert len(state["asr"]) == len(state["build"]) == 2


def test_changed_source_retranscribes(pipeline):
    run, state, video, _ = pipeline
    assert run() == 0
    video.write_bytes(b"replacement video with different contents")
    assert run() == 0
    assert len(state["asr"]) == len(state["build"]) == 2


def test_missing_artifact_rebuilds_despite_done_marker(pipeline):
    run, state, _, out = pipeline
    assert run() == 0
    next(out.rglob("cards.apkg")).unlink()
    assert run() == 0
    assert len(state["asr"]) == 1
    assert len(state["build"]) == 2


def test_unversioned_markers_are_not_trusted(pipeline):
    run, state, video, out = pipeline
    target = out / batch.sanitize(video.name)
    target.mkdir(parents=True)
    (target / ".done").touch()
    (target / ".asr_done").touch()
    (target / "subtitles.json").write_text("{}")
    assert run() == 0
    assert len(state["asr"]) == len(state["build"]) == 1


def test_limit_applies_to_both_workers(pipeline):
    run, state, video, _ = pipeline
    (video.parent / "second.mp4").touch()
    assert run("--limit", "1") == 0
    assert len(state["asr"]) == len(state["build"]) == 1
    assert run("--limit", "1") == 0
    assert len(state["asr"]) == len(state["build"]) == 2


def test_sanitized_paths_keep_distinct_video_identities():
    names = ["chapter/a b.mp4", "chapter/a.b.mp4", "chapter/a b.mkv",
             "chapter/A B.mp4", "chapter/" + "long" * 70 + "1.mp4",
             "chapter/" + "long" * 70 + "2.mp4"]
    outputs = [batch.sanitize(name) for name in names]
    assert len({name.casefold() for name in outputs}) == len(names)
    assert outputs[0] == batch.sanitize(names[0])
    assert batch.sanitize("chapter\\a b.mp4") == outputs[0]


def test_dry_run_does_not_clear_failure_or_create_manifest(pipeline):
    run, state, video, out = pipeline
    target = out / batch.sanitize(video.name)
    target.mkdir(parents=True)
    (target / ".failed").write_text("prior failure")
    assert run("--dry-run") == 0
    assert (target / ".failed").read_text() == "prior failure"
    assert not (target / "run_manifest.json").exists()
    assert not state["asr"] and not state["build"]
