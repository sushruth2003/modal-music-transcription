"""CPU audio normalization and durable job orchestration."""

from __future__ import annotations

import json
import math
import subprocess
import time
import wave
from pathlib import Path

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    AUDIO_SAMPLE_RATE,
    MAX_AUDIO_SECONDS,
    SUPPORTED_SOURCE_SUFFIXES,
)
from music_transcription.resources import app, artifact_volume, audio_image
from music_transcription.schemas import JobSpec, PreprocessingMetrics
from music_transcription.score import render_score
from music_transcription.storage import job_paths, mounted_artifact_path, update_job


def audio_duration_path(wav_path: Path, source_name: str) -> float:
    """Read duration from a PCM WAV without loading the artifact into memory."""

    with wave.open(str(wav_path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

    if frame_count == 0 or frame_rate == 0:
        raise ValueError(f"Decoded audio contains no samples: {source_name!r}")
    return frame_count / frame_rate


def probe_audio_duration(source_path: Path) -> float:
    """Read container-level duration without decoding the complete recording."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-4_000:]
        raise RuntimeError(f"ffprobe could not inspect {source_path.name!r}:\n{stderr}")
    try:
        duration = float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"ffprobe returned no duration for {source_path.name!r}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Audio duration is invalid for {source_path.name!r}")
    return duration


def enforce_audio_duration_limit(duration: float, source_name: str) -> None:
    if duration > MAX_AUDIO_SECONDS:
        maximum_minutes = MAX_AUDIO_SECONDS // 60
        raise ValueError(
            f"Audio {source_name!r} is {duration / 60:.1f} minutes; the public demo limit is "
            f"{maximum_minutes} minutes"
        )


def normalize_audio_file(source_path: Path, wav_path: Path) -> tuple[float, PreprocessingMetrics]:
    """Normalize one filesystem input directly to another without a byte payload."""

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"Unsupported source suffix: {suffix!r}")

    enforce_audio_duration_limit(probe_audio_duration(source_path), source_path.name)

    started = time.perf_counter()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-4_000:]
        raise RuntimeError(f"ffmpeg could not decode {source_path.name!r}:\n{stderr}")

    duration = audio_duration_path(wav_path, source_path.name)
    metrics: PreprocessingMetrics = {
        "seconds": time.perf_counter() - started,
        "source_bytes": source_path.stat().st_size,
        "normalized_bytes": wav_path.stat().st_size,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": 1,
    }
    return duration, metrics


@app.function(
    image=audio_image,
    cpu=2.0,
    memory=2048,
    timeout=60 * 60,
    max_containers=8,
    volumes={str(ARTIFACT_MOUNT_PATH): artifact_volume},
)
def process_job(spec: JobSpec) -> dict[str, object]:
    """Run one durable CPU-to-GPU job using only Volume references."""

    from music_transcription.beat_grid import BeatGridDetector
    from music_transcription.transcribe import MuScriptorTranscriber

    job_id = spec["job_id"]
    paths = job_paths(job_id, spec["source_suffix"])

    try:
        update_job(job_id, "preprocessing")

        artifact_volume.reload()
        source_path = mounted_artifact_path(paths["source"])
        normalized_path = mounted_artifact_path(paths["normalized"])
        audio_seconds, metrics = normalize_audio_file(source_path, normalized_path)

        preprocessing_record = {
            "audio_seconds": audio_seconds,
            "metrics": metrics,
        }
        mounted_artifact_path(paths["preprocessing"]).write_text(
            json.dumps(preprocessing_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_volume.commit()

        beat_detection = BeatGridDetector().detect_artifact.remote(
            job_id,
            spec["source_suffix"],
        )
        preprocessing_record["beat_grid"] = beat_detection
        mounted_artifact_path(paths["preprocessing"]).write_text(
            json.dumps(preprocessing_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_volume.commit()

        update_job(job_id, "transcribing")
        result = MuScriptorTranscriber().transcribe_artifact.remote(
            job_id,
            spec["source_suffix"],
            spec["instruments"],
            beat_detection,
        )
        if spec.get("generate_score", False):
            update_job(job_id, "rendering", result=result)
            result = render_score.remote(job_id, spec["source_suffix"])
        update_job(job_id, "completed", result=result)
        return result
    except Exception as error:
        update_job(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise
