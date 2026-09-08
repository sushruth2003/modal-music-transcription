"""Turn generated MIDI into approximate printable notation on a CPU worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    SCORE_ENGINE_COMMAND,
    SCORE_ENGINE_VERSION,
    SCORE_RENDER_TIMEOUT_SECONDS,
)
from music_transcription.resources import app, artifact_volume, score_image
from music_transcription.storage import get_job, job_paths, mounted_artifact_path, update_job


def render_command(midi_path: Path, output_path: Path) -> list[str]:
    return ["xvfb-run", "-a", SCORE_ENGINE_COMMAND, "-o", str(output_path), str(midi_path)]


def validate_score_pdf(path: Path) -> dict[str, object]:
    """Parse the complete PDF, rejecting corrupt, empty or blank-page exports."""
    from pypdf import PdfReader

    reader = PdfReader(path, strict=True)
    if reader.is_encrypted or not reader.pages:
        raise ValueError("Score renderer returned an encrypted or empty PDF")
    for page in reader.pages:
        content = page.get_contents()
        if (
            float(page.mediabox.width) <= 0
            or float(page.mediabox.height) <= 0
            or content is None
            or not content.get_data().strip()
        ):
            raise ValueError("Score renderer returned a blank or invalid page")
    return {
        "pages": len(reader.pages),
        "pdf_bytes": path.stat().st_size,
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def render_score_file(midi_path: Path, output_path: Path) -> dict[str, object]:
    """Render in a private workspace; publish only a validated complete PDF."""
    with midi_path.open("rb") as source:
        if source.read(4) != b"MThd":
            raise ValueError("Input is not a Standard MIDI file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auto-transcribe-score-") as folder:
        work = Path(folder)
        runtime = work / "runtime"
        runtime.mkdir(mode=0o700)
        temporary_pdf = work / "score.pdf"
        environment = {
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "XDG_RUNTIME_DIR": str(runtime),
            "XDG_CONFIG_HOME": str(work / "config"),
            "XDG_DATA_HOME": str(work / "data"),
        }
        try:
            completed = subprocess.run(
                render_command(midi_path.resolve(), temporary_pdf),
                capture_output=True,
                text=True,
                check=False,
                timeout=SCORE_RENDER_TIMEOUT_SECONDS - 30,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("PDF engraving exceeded the time limit") from error
        if completed.returncode != 0 or not temporary_pdf.is_file():
            detail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
            raise RuntimeError(
                f"MuseScore could not render PDF (exit {completed.returncode}): {detail}"
            )
        metadata = validate_score_pdf(temporary_pdf)
        # The destination sibling guarantees an atomic replace on the Volume.
        staging = output_path.with_suffix(".pdf.partial")
        try:
            staging.write_bytes(temporary_pdf.read_bytes())
            staging.replace(output_path)
        finally:
            staging.unlink(missing_ok=True)
    return {**metadata, "engine": "MuseScore Studio", "engine_version": SCORE_ENGINE_VERSION}


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

    metrics_path = mounted_artifact_path(paths["metrics"])
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    if result.get("note_count") == 0:
        result["score"] = {
            "status": "skipped",
            "skipped_reason": "No notes were detected, so there is no PDF score to render.",
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_volume.commit()
        return result

    started = time.perf_counter()
    validation = render_score_file(midi_path, pdf_path)
    score_seconds = time.perf_counter() - started

    metrics_path = mounted_artifact_path(paths["metrics"])
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    result["score"] = {
        **validation,
        "status": "completed",
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


@app.function(
    image=score_image,
    cpu=2,
    memory=4096,
    timeout=SCORE_RENDER_TIMEOUT_SECONDS + 30,
    max_containers=1,
    volumes={str(ARTIFACT_MOUNT_PATH): artifact_volume},
)
def create_score_for_job(job_id: str, source_suffix: str) -> None:
    """Add a score to a completed transcription, keeping MIDI/audio available."""
    record = get_job(job_id)
    if record.get("score_state") in {"completed", "skipped"}:
        return
    update_job(job_id, "completed", score_state="rendering")
    try:
        result = render_score.local(job_id, source_suffix)
        update_job(
            job_id,
            "completed",
            result=result,
            score_state=result["score"]["status"],
            score_error=None,
        )
    except Exception:
        # A PDF failure must not hide an already successful transcription.
        update_job(
            job_id,
            "completed",
            score_state="failed",
            score_error="The score could not be generated. You can retry or download the MIDI.",
        )
        raise
