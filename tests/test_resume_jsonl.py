"""JSONL append is fsync'd and resume skips ids already present. SPEC.md §4/§9."""

from __future__ import annotations

from src.clock import append_jsonl, read_existing_ids, read_jsonl


def test_append_jsonl_then_read_back(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    append_jsonl(path, {"id": "r_1", "shortcut": True})
    append_jsonl(path, {"id": "r_2", "shortcut": False})
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == ["r_1", "r_2"]


def test_read_existing_ids_empty_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    assert read_existing_ids(path) == set()


def test_resume_skips_existing_ids(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    for i in range(5):
        append_jsonl(path, {"id": f"r_{i}", "value": i})

    existing = read_existing_ids(path)
    assert existing == {f"r_{i}" for i in range(5)}

    planned_ids = [f"r_{i}" for i in range(8)]
    todo = [i for i in planned_ids if i not in existing]
    assert todo == ["r_5", "r_6", "r_7"]

    for i in todo:
        append_jsonl(path, {"id": i, "value": i})

    rows = read_jsonl(path)
    assert len(rows) == 8
    assert len({r["id"] for r in rows}) == 8  # no duplicates from a naive re-run


def test_read_jsonl_tolerates_torn_last_line(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    append_jsonl(path, {"id": "r_1"})
    # simulate a crash mid-write: a truncated final line with no trailing newline
    with open(path, "a") as f:
        f.write('{"id": "r_2", "partial": tr')
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == ["r_1"]


def test_append_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "rollouts.jsonl"
    append_jsonl(path, {"id": "r_1"})
    assert path.exists()
