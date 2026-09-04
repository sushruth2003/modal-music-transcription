"""JSON- and cloudpickle-safe payload shapes crossing remote boundaries."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class PreprocessingMetrics(TypedDict):
    seconds: float
    source_bytes: int
    normalized_bytes: int
    sample_rate: int
    channels: int


class SerializedEvent(TypedDict):
    type: Literal["note_start", "note_end", "progress"]
    index: NotRequired[int]
    pitch: NotRequired[int]
    instrument: NotRequired[str]
    time: NotRequired[float]
    completed: NotRequired[int]
    total: NotRequired[int]


JobState = Literal[
    "submitted",
    "fetching",
    "preprocessing",
    "transcribing",
    "rendering",
    "completed",
    "failed",
]


class JobPaths(TypedDict):
    source: str
    request: str
    normalized: str
    preprocessing: str
    events: str
    midi: str
    score_pdf: str
    musicxml: str
    metrics: str


class JobSpec(TypedDict):
    job_id: str
    source_name: str
    source_suffix: str
    instruments: list[str] | None
    source_url: NotRequired[str]
    generate_score: NotRequired[bool]


class JobRecord(TypedDict):
    job_id: str
    state: JobState
    source_name: str
    instruments: list[str] | None
    source_kind: Literal["upload", "url"]
    generate_score: bool
    paths: JobPaths
    created_at: str
    updated_at: str
    error: NotRequired[str]
    result: NotRequired[dict[str, object]]
