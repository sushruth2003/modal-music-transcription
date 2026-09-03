"""Stable job identities, artifact paths, and persistent status records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from music_transcription.config import ARTIFACT_MOUNT_PATH, SUPPORTED_AUDIO_SUFFIXES
from music_transcription.resources import job_states
from music_transcription.schemas import JobPaths, JobRecord, JobSpec, JobState


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_job_spec(source_name: str, instruments: list[str] | None) -> JobSpec:
    """Create the small value sent to a remote M1 Function."""

    suffix = Path(source_name).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported source suffix: {suffix!r}")
    return {
        "job_id": uuid4().hex,
        "source_name": Path(source_name).name,
        "source_suffix": suffix,
        "instruments": instruments,
    }


def validate_job_id(job_id: str) -> str:
    """Reject path traversal and mistyped IDs before building Volume paths."""

    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise ValueError("job_id must be a 32-character lowercase hexadecimal value")
    return job_id


def job_paths(job_id: str, source_suffix: str) -> JobPaths:
    validate_job_id(job_id)
    if source_suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported source suffix: {source_suffix!r}")

    prefix = f"jobs/{job_id}"
    return {
        "source": f"{prefix}/source{source_suffix}",
        "request": f"{prefix}/request.json",
        "normalized": f"{prefix}/normalized.wav",
        "preprocessing": f"{prefix}/preprocessing.json",
        "events": f"{prefix}/events.jsonl",
        "midi": f"{prefix}/transcription.mid",
        "metrics": f"{prefix}/metrics.json",
    }


def mounted_artifact_path(relative_path: str) -> Path:
    """Resolve a validated Volume-relative artifact path inside a container."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact path must be safe and relative: {relative_path!r}")
    return ARTIFACT_MOUNT_PATH / relative


def initial_job_record(spec: JobSpec) -> JobRecord:
    now = utc_now()
    return {
        "job_id": spec["job_id"],
        "state": "submitted",
        "source_name": spec["source_name"],
        "instruments": spec["instruments"],
        "paths": job_paths(spec["job_id"], spec["source_suffix"]),
        "created_at": now,
        "updated_at": now,
    }


def get_job(job_id: str) -> JobRecord:
    validate_job_id(job_id)
    record = job_states.get(job_id)
    if record is None:
        raise KeyError(f"Unknown job: {job_id}")
    return record


def update_job(job_id: str, state: JobState, **fields: Any) -> JobRecord:
    """Replace one job record after merging a sequential pipeline update."""

    record = get_job(job_id)
    updated: JobRecord = {
        **record,
        **fields,
        "state": state,
        "updated_at": utc_now(),
    }
    job_states[job_id] = updated
    return updated
