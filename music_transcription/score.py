"""Turn generated MIDI into approximate printable notation on a CPU worker."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    SCORE_RENDER_TIMEOUT_SECONDS,
)
from music_transcription.resources import app, artifact_volume, score_image
from music_transcription.storage import job_paths, mounted_artifact_path


def render_command(midi_path: Path, output_path: Path) -> list[str]:
    return ["mscore3", "-o", str(output_path), str(midi_path)]


def render_score_file(midi_path: Path, output_path: Path) -> None:
    """Use MuseScore's headless CLI to import MIDI and export one score format."""

    runtime = Path("/tmp/auto-transcribe-musescore")
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "XDG_RUNTIME_DIR": str(runtime),
    }
    completed = subprocess.run(
        render_command(midi_path, output_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=SCORE_RENDER_TIMEOUT_SECONDS,
        env=environment,
    )
    if completed.returncode != 0 or not output_path.is_file():
        detail = (completed.stderr or completed.stdout).strip()[-2_000:]
        raise RuntimeError(f"MuseScore could not render {output_path.suffix}: {detail}")


@app.function(
    image=score_image,
    cpu=2.0,
    memory=4096,
    timeout=SCORE_RENDER_TIMEOUT_SECONDS,
    max_containers=4,
    volumes={str(ARTIFACT_MOUNT_PATH): artifact_volume},
)
def render_score(job_id: str, source_suffix: str) -> dict[str, object]:
    """Render a PDF score, then merge score metadata into the job result."""

    paths = job_paths(job_id, source_suffix)
    artifact_volume.reload()
    midi_path = mounted_artifact_path(paths["midi"])
    pdf_path = mounted_artifact_path(paths["score_pdf"])

    started = time.perf_counter()
    render_score_file(midi_path, pdf_path)
    score_seconds = time.perf_counter() - started

    metrics_path = mounted_artifact_path(paths["metrics"])
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    result["score"] = {
        "seconds": score_seconds,
        "pdf_bytes": pdf_path.stat().st_size,
        "automatic_quantization": True,
    }
    artifacts = dict(result.get("artifacts", {}))
    artifacts["score_pdf"] = paths["score_pdf"]
    result["artifacts"] = artifacts
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_volume.commit()
    return result
