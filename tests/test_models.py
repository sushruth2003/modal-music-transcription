from __future__ import annotations

import json

from music_transcription import models


def configure_paths(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    marker = tmp_path / "READY.json"
    monkeypatch.setattr(models, "MODEL_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(models, "MODEL_CONFIG_PATH", config)
    monkeypatch.setattr(models, "MODEL_READY_PATH", marker)
    monkeypatch.setattr(models, "MIN_EXPECTED_CHECKPOINT_BYTES", 4)


def test_ready_metadata_accepts_complete_snapshot(tmp_path, monkeypatch) -> None:
    configure_paths(tmp_path, monkeypatch)
    models.MODEL_CHECKPOINT_PATH.write_bytes(b"1234")
    models.MODEL_CONFIG_PATH.write_text("{}", encoding="utf-8")
    metadata = {
        "repo_id": models.MODEL_REPO_ID,
        "revision": models.MODEL_REVISION,
        "checkpoint_bytes": 4,
    }
    models.MODEL_READY_PATH.write_text(json.dumps(metadata), encoding="utf-8")

    assert models._ready_metadata() == metadata


def test_ready_metadata_rejects_stale_revision(tmp_path, monkeypatch) -> None:
    configure_paths(tmp_path, monkeypatch)
    models.MODEL_CHECKPOINT_PATH.write_bytes(b"1234")
    models.MODEL_CONFIG_PATH.write_text("{}", encoding="utf-8")
    models.MODEL_READY_PATH.write_text(
        json.dumps({"repo_id": models.MODEL_REPO_ID, "revision": "stale"}),
        encoding="utf-8",
    )

    assert models._ready_metadata() is None


def test_ready_metadata_rejects_incomplete_snapshot(tmp_path, monkeypatch) -> None:
    configure_paths(tmp_path, monkeypatch)
    models.MODEL_READY_PATH.write_text(
        json.dumps({"repo_id": models.MODEL_REPO_ID, "revision": models.MODEL_REVISION}),
        encoding="utf-8",
    )

    assert models._ready_metadata() is None
