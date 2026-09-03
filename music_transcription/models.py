"""One-time, idempotent materialization of MuScriptor Large weights."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from music_transcription.config import (
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


def _ready_metadata() -> dict[str, str | int] | None:
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
def download_model() -> dict[str, str | int | bool]:
    """Download the pinned checkpoint once and commit it to the model Volume."""

    from huggingface_hub import snapshot_download

    # A reused downloader container may have mounted the Volume before another
    # invocation committed its changes.
    model_volume.reload()

    existing = _ready_metadata()
    if existing is not None:
        result = {**existing, "downloaded": False}
        print(json.dumps(result, sort_keys=True))
        return result

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

    metadata: dict[str, str | int] = {
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "checkpoint_bytes": checkpoint_bytes,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    MODEL_READY_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_volume.commit()
    result = {**metadata, "downloaded": True}
    print(json.dumps(result, sort_keys=True))
    return result
