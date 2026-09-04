"""Stable job identities, artifact paths, and persistent status records."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    SUPPORTED_SOURCE_SUFFIXES,
)
from music_transcription.resources import artifact_volume, job_states
from music_transcription.schemas import JobPaths, JobRecord, JobSpec, JobState


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_job_spec(
    source_name: str,
    instruments: list[str] | None,
    *,
    generate_score: bool = False,
) -> JobSpec:
    """Create the small value sent to a remote M1 Function."""

    suffix = Path(source_name).suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"Unsupported source suffix: {suffix!r}")
    return {
        "job_id": uuid4().hex,
        "source_name": Path(source_name).name,
        "source_suffix": suffix,
        "instruments": instruments,
        "generate_score": generate_score,
    }


def parse_instruments(value: str | None) -> list[str] | None:
    """Normalize comma-separated instrument hints from a CLI or form."""

    if value is None:
        return None
    instruments = [item.strip() for item in value.split(",") if item.strip()]
    return instruments or None


def validate_job_id(job_id: str) -> str:
    """Reject path traversal and mistyped IDs before building Volume paths."""

    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise ValueError("job_id must be a 32-character lowercase hexadecimal value")
    return job_id


def job_paths(job_id: str, source_suffix: str) -> JobPaths:
    validate_job_id(job_id)
    if source_suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"Unsupported source suffix: {source_suffix!r}")

    prefix = f"jobs/{job_id}"
    return {
        "source": f"{prefix}/source{source_suffix}",
        "request": f"{prefix}/request.json",
        "normalized": f"{prefix}/normalized.wav",
        "preprocessing": f"{prefix}/preprocessing.json",
        "events": f"{prefix}/events.jsonl",
        "midi": f"{prefix}/transcription.mid",
        "score_pdf": f"{prefix}/score.pdf",
        "musicxml": f"{prefix}/score.musicxml",
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
        "generate_score": bool(spec.get("generate_score", False)),
        "paths": job_paths(spec["job_id"], spec["source_suffix"]),
        "created_at": now,
        "updated_at": now,
    }


def stage_job_sources(sources: list[Path], specs: list[JobSpec]) -> None:
    """Commit source media and immutable request metadata, then publish status."""

    if len(sources) != len(specs):
        raise ValueError("sources and specs must have the same length")

    with artifact_volume.batch_upload() as batch:
        for source, spec in zip(sources, specs, strict=True):
            paths = job_paths(spec["job_id"], spec["source_suffix"])
            request = json.dumps(spec, indent=2, sort_keys=True).encode() + b"\n"
            batch.put_file(source, f"/{paths['source']}")
            batch.put_file(io.BytesIO(request), f"/{paths['request']}")

    for spec in specs:
        job_states[spec["job_id"]] = initial_job_record(spec)


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
