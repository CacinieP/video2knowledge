"""Exercise actual APKG identities, not just hash implementation details."""
import csv
from contextlib import closing
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gen_apkg


def export(tmp_path, monkeypatch, rows, deck="Course::Lesson", source_id=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "cards.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "cards.apkg"
    monkeypatch.setattr(sys, "argv", ["gen_apkg", "--csv", str(csv_path),
                        "--out", str(out), "--deck", deck, "--source-id", source_id])
    assert gen_apkg.main() == 0
    with zipfile.ZipFile(out) as archive:
        db = tmp_path / "collection.anki2"
        db.write_bytes(archive.read("collection.anki2"))
    with closing(sqlite3.connect(db)) as conn:
        guids = {fields.split("\x1f")[0]: guid
                 for guid, fields in conn.execute("SELECT guid, flds FROM notes")}
        decks = json.loads(conn.execute("SELECT decks FROM col").fetchone()[0])
    return guids, {name: meta["id"] for name, meta in
                   ((d["name"], d) for d in decks.values()) if name != "Default"}


def test_reordering_inserting_and_answer_edits_preserve_note_guids(tmp_path, monkeypatch):
    first = {"question": "What is entropy?", "answer": "old answer",
             "source": "lecture", "timestamp": "00:12"}
    second = {"question": "What is syntax?", "answer": "grammar"}
    old, _ = export(tmp_path, monkeypatch, [first, second])
    updated, _ = export(tmp_path, monkeypatch, [second, {"question": "New question"},
                                               {**first, "answer": "revised answer"}])
    assert all(updated[q] == guid for q, guid in old.items())


def test_anagram_decks_have_distinct_ids_and_notes(tmp_path, monkeypatch):
    rows = [{"question": "Same question", "answer": "answer"}]
    a, decks_a = export(tmp_path, monkeypatch, rows, deck="AB")
    b, decks_b = export(tmp_path, monkeypatch, rows, deck="BA")
    assert decks_a["AB"] != decks_b["BA"]
    assert a["Same question"] != b["Same question"]


def test_same_question_from_different_sources_has_distinct_identity():
    a = {"question": "What happened?", "source": "lesson-a", "timestamp": "00:12"}
    assert gen_apkg.note_guid("Course", a) != gen_apkg.note_guid(
        "Course", {**a, "source": "lesson-b"})


def test_independent_csv_files_do_not_overwrite_each_other(tmp_path, monkeypatch):
    rows = [{"question": "Same question", "source": "subtitles.json"}]
    a, _ = export(tmp_path / "video-a", monkeypatch, rows)
    b, _ = export(tmp_path / "video-b", monkeypatch, rows)
    assert a != b


def test_explicit_source_id_survives_export_directory_move(tmp_path, monkeypatch):
    rows = [{"question": "Same question", "source": "merged.json (10 ASR)"}]
    a, _ = export(tmp_path / "old", monkeypatch, rows, source_id="course/video-a")
    rows[0]["source"] = "merged.json (20 ASR)"
    b, _ = export(tmp_path / "new", monkeypatch, rows, source_id="course/video-a")
    assert a == b
