from __future__ import annotations

import hashlib
import io
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


def configure_beat_checkpoint(tmp_path, monkeypatch, payload: bytes = b"beat") -> None:
    monkeypatch.setattr(models, "BEAT_CHECKPOINT_PATH", tmp_path / "beat_this-final0.ckpt")
    monkeypatch.setattr(models, "BEAT_CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(models, "BEAT_CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest())


def test_beat_checkpoint_ready_validates_size_and_hash(tmp_path, monkeypatch) -> None:
    payload = b"beat"
    configure_beat_checkpoint(tmp_path, monkeypatch, payload)
    models.BEAT_CHECKPOINT_PATH.write_bytes(payload)

    assert models._beat_checkpoint_ready()

    models.BEAT_CHECKPOINT_PATH.write_bytes(b"bent")
    assert not models._beat_checkpoint_ready()


def test_download_beat_checkpoint_is_verified_and_installed(tmp_path, monkeypatch) -> None:
    payload = b"pinned beat checkpoint"
    configure_beat_checkpoint(tmp_path, monkeypatch, payload)
    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    models._download_beat_checkpoint()

    assert models.BEAT_CHECKPOINT_PATH.read_bytes() == payload
    assert not models.BEAT_CHECKPOINT_PATH.with_suffix(".ckpt.partial").exists()
