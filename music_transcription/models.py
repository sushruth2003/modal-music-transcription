"""One-time, idempotent materialization of transcription model weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from music_transcription.config import (
    BEAT_CHECKPOINT_BYTES,
    BEAT_CHECKPOINT_NAME,
    BEAT_CHECKPOINT_PATH,
    BEAT_CHECKPOINT_SHA256,
    BEAT_CHECKPOINT_URL,
    MIN_EXPECTED_CHECKPOINT_BYTES,
    MODEL_CHECKPOINT_PATH,
    MODEL_CONFIG_PATH,
    MODEL_MOUNT_PATH,
    MODEL_READY_PATH,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT_PATH,
)
from music_transcription.resources import (
    app,
    download_image,
    huggingface_secret,
    model_volume,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _beat_checkpoint_ready() -> bool:
    """Validate the exact Beat This! checkpoint pinned by this deployment."""

    return (
        BEAT_CHECKPOINT_PATH.is_file()
        and BEAT_CHECKPOINT_PATH.stat().st_size == BEAT_CHECKPOINT_BYTES
        and _sha256_path(BEAT_CHECKPOINT_PATH) == BEAT_CHECKPOINT_SHA256
    )


def _download_beat_checkpoint() -> None:
    """Stream and atomically install the pinned Beat This! checkpoint."""

    BEAT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = BEAT_CHECKPOINT_PATH.with_suffix(".ckpt.partial")
    try:
        with (
            urllib.request.urlopen(BEAT_CHECKPOINT_URL, timeout=10 * 60) as response,
            partial_path.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination, length=1024 * 1024)

        checkpoint_bytes = partial_path.stat().st_size
        if checkpoint_bytes != BEAT_CHECKPOINT_BYTES:
            raise RuntimeError(
                f"Beat checkpoint is {checkpoint_bytes:,} bytes; "
                f"expected exactly {BEAT_CHECKPOINT_BYTES:,}"
            )

        checkpoint_sha256 = _sha256_path(partial_path)
        if checkpoint_sha256 != BEAT_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Beat checkpoint SHA-256 mismatch: "
                f"received {checkpoint_sha256}, expected {BEAT_CHECKPOINT_SHA256}"
            )
        partial_path.replace(BEAT_CHECKPOINT_PATH)
    finally:
        partial_path.unlink(missing_ok=True)


def _ready_metadata() -> dict[str, object] | None:
    """Return validated marker metadata, or None for an incomplete snapshot."""

    if not MODEL_READY_PATH.is_file():
        return None

    try:
        metadata = json.loads(MODEL_READY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected_identity = (
        metadata.get("repo_id") == MODEL_REPO_ID and metadata.get("revision") == MODEL_REVISION
    )
    expected_files = MODEL_CHECKPOINT_PATH.is_file() and MODEL_CONFIG_PATH.is_file()
    expected_size = (
        expected_files and MODEL_CHECKPOINT_PATH.stat().st_size >= MIN_EXPECTED_CHECKPOINT_BYTES
    )
    return metadata if expected_identity and expected_size else None


@app.function(
    image=download_image,
    cpu=2.0,
    memory=4096,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(MODEL_MOUNT_PATH): model_volume},
    secrets=[huggingface_secret],
)
def download_model() -> dict[str, object]:
    """Download the pinned MuScriptor and Beat This! checkpoints once."""

    from huggingface_hub import snapshot_download

    # A reused downloader container may have mounted the Volume before another
    # invocation committed its changes.
    model_volume.reload()

    existing = _ready_metadata()
    beat_metadata = {
        "name": BEAT_CHECKPOINT_NAME,
        "bytes": BEAT_CHECKPOINT_BYTES,
        "sha256": BEAT_CHECKPOINT_SHA256,
    }
    beat_ready = _beat_checkpoint_ready()
    if existing is not None and beat_ready and existing.get("beat_checkpoint") == beat_metadata:
        result = {
            **existing,
            "downloaded": False,
            "model_downloaded": False,
            "beat_checkpoint_downloaded": False,
        }
        print(json.dumps(result, sort_keys=True))
        return result

    model_downloaded = existing is None
    if model_downloaded:
        MODEL_SNAPSHOT_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_SNAPSHOT_PATH),
            token=os.environ["HF_TOKEN"],
        )

        if not MODEL_CHECKPOINT_PATH.is_file() or not MODEL_CONFIG_PATH.is_file():
            raise RuntimeError("Downloaded snapshot is missing model.safetensors or config.json")

        checkpoint_bytes = MODEL_CHECKPOINT_PATH.stat().st_size
        if checkpoint_bytes < MIN_EXPECTED_CHECKPOINT_BYTES:
            raise RuntimeError(
                f"Checkpoint is unexpectedly small: {checkpoint_bytes:,} bytes; "
                f"expected at least {MIN_EXPECTED_CHECKPOINT_BYTES:,}"
            )
        metadata: dict[str, object] = {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "checkpoint_bytes": checkpoint_bytes,
        }
    else:
        metadata = dict(existing)

    beat_downloaded = not beat_ready
    if beat_downloaded:
        _download_beat_checkpoint()

    metadata.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "beat_checkpoint": beat_metadata,
        }
    )
    MODEL_READY_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_volume.commit()
    result = {
        **metadata,
        "downloaded": model_downloaded or beat_downloaded,
        "model_downloaded": model_downloaded,
        "beat_checkpoint_downloaded": beat_downloaded,
    }
    print(json.dumps(result, sort_keys=True))
    return result
