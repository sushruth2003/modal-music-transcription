"""CPU-only FastAPI service for durable transcription jobs and artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, BinaryIO
from urllib.parse import quote

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from music_transcription.config import (
    FRONTEND_MOUNT_PATH,
    L4_PRICE_PER_SECOND_USD,
    MAX_INSTRUMENT_HINTS,
    WEB_MAX_CONCURRENT_INPUTS,
    WEB_MAX_CONTAINERS,
    WEB_MAX_UPLOAD_BYTES,
    WEB_RATE_LIMIT_WINDOW_SECONDS,
    WEB_SUBMISSIONS_GLOBAL_DAY,
    WEB_SUBMISSIONS_PER_IP_HOUR,
    WEB_TARGET_CONCURRENT_INPUTS,
    WEB_UPLOAD_CHUNK_BYTES,
)
from music_transcription.ingest import validate_media_url
from music_transcription.preprocess import process_job
from music_transcription.resources import app, artifact_volume, rate_limit_states, web_image
from music_transcription.schemas import JobRecord, JobSpec, SerializedEvent
from music_transcription.storage import (
    get_job,
    new_job_spec,
    new_url_job_spec,
    parse_instruments,
    stage_job_sources,
    stage_url_job,
    update_job,
)

LOCAL_FRONTEND_PATH = Path(__file__).parent / "frontend"
TERMINAL_STATES = frozenset({"completed", "failed"})
STATE_PROGRESS = {
    "submitted": 10,
    "fetching": 20,
    "preprocessing": 30,
    "transcribing": 65,
    "rendering": 88,
    "completed": 100,
    "failed": 100,
}
RATE_LIMIT_STATE_KEY = "submission-limits-v1"


class UploadTooLargeError(ValueError):
    """Raised after a streamed upload crosses the configured byte limit."""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    ip_remaining: int
    global_remaining: int


def evaluate_submission_limit(
    stored: dict[str, object] | None,
    client_key: str,
    now: float,
) -> tuple[dict[str, object], RateLimitDecision]:
    """Apply rolling per-client and UTC-day global limits to a persisted record."""

    current = datetime.fromtimestamp(now, UTC)
    day = current.date().isoformat()
    if stored is None or stored.get("day") != day:
        state: dict[str, object] = {"day": day, "global_count": 0, "clients": {}}
    else:
        state = {
            "day": day,
            "global_count": int(stored.get("global_count", 0)),
            "clients": dict(stored.get("clients", {})),
        }

    clients = state["clients"]
    assert isinstance(clients, dict)
    cutoff = now - WEB_RATE_LIMIT_WINDOW_SECONDS
    events = [
        float(timestamp) for timestamp in clients.get(client_key, []) if float(timestamp) > cutoff
    ]
    clients[client_key] = events
    global_count = int(state["global_count"])

    if global_count >= WEB_SUBMISSIONS_GLOBAL_DAY:
        next_day = datetime.combine(current.date() + timedelta(days=1), datetime.min.time(), UTC)
        retry_after = max(1, int((next_day - current).total_seconds()))
        return state, RateLimitDecision(False, retry_after, 0, 0)
    if len(events) >= WEB_SUBMISSIONS_PER_IP_HOUR:
        retry_after = max(1, int(events[0] + WEB_RATE_LIMIT_WINDOW_SECONDS - now) + 1)
        return state, RateLimitDecision(
            False,
            retry_after,
            0,
            WEB_SUBMISSIONS_GLOBAL_DAY - global_count,
        )

    events.append(now)
    state["global_count"] = global_count + 1
    return state, RateLimitDecision(
        True,
        0,
        WEB_SUBMISSIONS_PER_IP_HOUR - len(events),
        WEB_SUBMISSIONS_GLOBAL_DAY - global_count - 1,
    )


def reserve_submission(client_ip: str, now: float | None = None) -> RateLimitDecision:
    """Atomically reserve one quota slot inside the single web container."""

    client_key = hashlib.sha256(client_ip.encode()).hexdigest()[:24]
    stored = rate_limit_states.get(RATE_LIMIT_STATE_KEY)
    state, decision = evaluate_submission_limit(
        stored,
        client_key,
        datetime.now(UTC).timestamp() if now is None else now,
    )
    rate_limit_states[RATE_LIMIT_STATE_KEY] = state
    return decision


def copy_limited(source: BinaryIO, destination: Path, max_bytes: int) -> int:
    """Copy a file-like object in bounded chunks while enforcing a hard limit."""

    total = 0
    with destination.open("wb") as output:
        while chunk := source.read(WEB_UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLargeError(f"Audio files are limited to {max_bytes} bytes")
            output.write(chunk)
    if total == 0:
        raise ValueError("The uploaded audio file is empty")
    return total


def validate_instruments(value: str | None) -> list[str] | None:
    instruments = parse_instruments(value)
    if instruments is None:
        return None
    if len(instruments) > MAX_INSTRUMENT_HINTS:
        raise ValueError(f"At most {MAX_INSTRUMENT_HINTS} instrument hints are allowed")
    if any(len(instrument) > 64 for instrument in instruments):
        raise ValueError("Instrument hints must be 64 characters or fewer")
    return instruments


def stage_uploaded_file(
    source: BinaryIO,
    source_name: str,
    instruments: list[str] | None,
    generate_score: bool = False,
) -> tuple[JobSpec, int]:
    """Spool one request to ephemeral disk, then commit it to the artifact Volume."""

    spec = new_job_spec(source_name, instruments, generate_score=generate_score)
    with tempfile.TemporaryDirectory(prefix="music-transcription-upload-") as directory:
        local_source = Path(directory) / f"source{spec['source_suffix']}"
        source_bytes = copy_limited(source, local_source, WEB_MAX_UPLOAD_BYTES)
        stage_job_sources([local_source], [spec])
    return spec, source_bytes


def spawn_process_job(spec: JobSpec) -> str:
    """Start the durable worker and return Modal's execution handle ID."""

    return process_job.spawn(spec).object_id


def read_artifact_bytes(relative_path: str, max_bytes: int = WEB_MAX_UPLOAD_BYTES) -> bytes:
    """Read a bounded artifact using the Volume client API, without mounting it."""

    chunks: list[bytes] = []
    total = 0
    for chunk in artifact_volume.read_file(relative_path):
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError(f"Artifact exceeds the {max_bytes}-byte response limit")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_event_stream(payload: bytes) -> list[SerializedEvent]:
    events: list[SerializedEvent] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid event JSON on line {line_number}") from error
        if not isinstance(event, dict) or event.get("type") not in {
            "note_start",
            "note_end",
            "progress",
        }:
            raise ValueError(f"Invalid event record on line {line_number}")
        events.append(event)
    return events


def pair_note_events(events: Iterable[SerializedEvent]) -> list[dict[str, object]]:
    """Pair MuScriptor start/end events into browser-friendly piano-roll notes."""

    starts: dict[int, dict[str, object]] = {}
    notes: list[dict[str, object]] = []
    for event in events:
        event_type = event["type"]
        if event_type == "note_start":
            index = event.get("index")
            pitch = event.get("pitch")
            start = event.get("time")
            instrument = event.get("instrument")
            if not isinstance(index, int) or not isinstance(pitch, int):
                continue
            if not isinstance(start, (int, float)) or not isinstance(instrument, str):
                continue
            starts[index] = {
                "index": index,
                "pitch": pitch,
                "instrument": instrument,
                "start": float(start),
            }
        elif event_type == "note_end":
            index = event.get("index")
            end = event.get("time")
            if not isinstance(index, int) or not isinstance(end, (int, float)):
                continue
            note = starts.pop(index, None)
            if note is None or float(end) <= float(note["start"]):
                continue
            notes.append({**note, "end": float(end)})
    return sorted(notes, key=lambda note: (float(note["start"]), int(note["pitch"])))


def public_job_record(record: JobRecord) -> dict[str, object]:
    """Expose useful state without leaking internal filesystem paths."""

    job_id = record["job_id"]
    result = dict(record.get("result", {}))
    result.pop("artifacts", None)
    inference = result.get("inference")
    if isinstance(inference, dict) and isinstance(inference.get("seconds"), (int, float)):
        inference = dict(inference)
        inference["estimated_gpu_cost_usd"] = round(
            float(inference["seconds"]) * L4_PRICE_PER_SECOND_USD,
            6,
        )
        result["inference"] = inference

    response: dict[str, object] = {
        "job_id": job_id,
        "state": record["state"],
        "progress": STATE_PROGRESS[record["state"]],
        "source_name": record["source_name"],
        "instruments": record["instruments"],
        "source_kind": record.get("source_kind", "upload"),
        "generate_score": record.get("generate_score", False),
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "links": {
            "self": f"/transcriptions/{job_id}",
            "audio": f"/transcriptions/{job_id}/audio",
            "events": f"/transcriptions/{job_id}/events",
            "piano_roll": f"/transcriptions/{job_id}/piano-roll",
            "midi": f"/transcriptions/{job_id}/midi",
        },
    }
    if record.get("generate_score"):
        links = response["links"]
        assert isinstance(links, dict)
        links["score_pdf"] = f"/transcriptions/{job_id}/score.pdf"
        links["musicxml"] = f"/transcriptions/{job_id}/musicxml"
    if result:
        response["result"] = result
    if "error" in record:
        response["error"] = record["error"]
    if record["state"] not in TERMINAL_STATES:
        response["retry_after_seconds"] = 2
    return response


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    return f"{disposition}; filename*=UTF-8''{quote(Path(filename).name)}"


def _frontend_directory() -> Path:
    return FRONTEND_MOUNT_PATH if FRONTEND_MOUNT_PATH.is_dir() else LOCAL_FRONTEND_PATH


def create_web_app(*, enforce_submission_limits: bool = True) -> FastAPI:
    """Construct the ASGI app. Kept separate so HTTP behavior is unit-testable."""

    web_app = FastAPI(
        title="Auto Transcribe API",
        description="Asynchronous audio-to-MIDI transcription",
        version="0.3.0",
    )
    submission_lock = asyncio.Lock()

    @web_app.middleware("http")
    async def limit_submissions(request: Request, call_next: Any) -> Response:
        decision: RateLimitDecision | None = None
        if (
            enforce_submission_limits
            and request.method == "POST"
            and request.url.path == "/transcriptions"
        ):
            client_ip = request.client.host if request.client is not None else "unknown"
            try:
                async with submission_lock:
                    decision = await asyncio.to_thread(reserve_submission, client_ip)
            except Exception:  # noqa: BLE001 - fail closed if quota storage is unavailable
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Submission quota is temporarily unavailable"},
                    headers={"Retry-After": "30", "Cache-Control": "no-store"},
                )
            if not decision.allowed:
                detail = (
                    "Today's public demo quota is full"
                    if decision.global_remaining == 0
                    else "This network has reached its hourly submission limit"
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": detail,
                        "retry_after_seconds": decision.retry_after_seconds,
                    },
                    headers={
                        "Retry-After": str(decision.retry_after_seconds),
                        "X-RateLimit-IP-Remaining": str(decision.ip_remaining),
                        "X-RateLimit-Global-Remaining": str(decision.global_remaining),
                        "Cache-Control": "no-store",
                    },
                )

        response = await call_next(request)
        if request.url.path.startswith("/transcriptions"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path in {"/", "/index.html", "/app.js", "/styles.css"} or (
            request.url.path.startswith("/jobs/")
        ):
            response.headers["Cache-Control"] = "no-cache"
        if decision is not None:
            response.headers["X-RateLimit-IP-Remaining"] = str(decision.ip_remaining)
            response.headers["X-RateLimit-Global-Remaining"] = str(decision.global_remaining)
        return response

    async def lookup_job(job_id: str) -> JobRecord:
        try:
            return await asyncio.to_thread(get_job, job_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Transcription job not found") from error

    async def completed_job(job_id: str) -> JobRecord:
        record = await lookup_job(job_id)
        if record["state"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Artifacts are not ready while the job is {record['state']!r}",
            )
        return record

    async def artifact(record: JobRecord, name: str) -> bytes:
        try:
            return await asyncio.to_thread(read_artifact_bytes, record["paths"][name])
        except Exception as error:
            raise HTTPException(status_code=404, detail="Artifact not found") from error

    @web_app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web_app.post("/transcriptions", status_code=status.HTTP_202_ACCEPTED)
    async def submit_transcription(
        request: Request,
        audio: Annotated[UploadFile | None, File()] = None,
        source_url: Annotated[str | None, Form()] = None,
        instruments: Annotated[str | None, Form()] = None,
        generate_score: Annotated[bool, Form()] = False,
    ) -> JSONResponse:
        normalized_url = source_url.strip() if source_url else None
        if (audio is None) == (normalized_url is None):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one source: an upload or a public YouTube URL",
            )

        try:
            hints = validate_instruments(instruments)
            if audio is not None:
                content_length = request.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > WEB_MAX_UPLOAD_BYTES + WEB_UPLOAD_CHUNK_BYTES
                ):
                    raise UploadTooLargeError("Audio upload is too large")
                if not audio.filename:
                    raise ValueError("The upload needs a filename")
                await audio.seek(0)
                spec, source_bytes = await asyncio.to_thread(
                    stage_uploaded_file,
                    audio.file,
                    audio.filename,
                    hints,
                    generate_score,
                )
            else:
                assert normalized_url is not None
                safe_url = validate_media_url(normalized_url)
                spec = new_url_job_spec(
                    safe_url,
                    hints,
                    generate_score=generate_score,
                )
                await asyncio.to_thread(stage_url_job, spec)
                source_bytes = None
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if audio is not None:
                await audio.close()

        try:
            call_id = await asyncio.to_thread(spawn_process_job, spec)
        except Exception as error:
            await asyncio.to_thread(
                update_job,
                spec["job_id"],
                "failed",
                error=f"{type(error).__name__}: worker submission failed",
            )
            raise HTTPException(
                status_code=503, detail="The transcription worker could not start"
            ) from error

        job_id = spec["job_id"]
        content: dict[str, object] = {
            "job_id": job_id,
            "state": "submitted",
            "source_kind": "url" if normalized_url else "upload",
            "generate_score": generate_score,
            "function_call_id": call_id,
            "status_url": f"/transcriptions/{job_id}",
            "result_url": f"/jobs/{job_id}",
        }
        if source_bytes is not None:
            content["source_bytes"] = source_bytes
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=content,
            headers={"Location": f"/transcriptions/{job_id}", "Retry-After": "2"},
        )

    @web_app.get("/transcriptions/{job_id}")
    async def transcription_status(job_id: str) -> dict[str, object]:
        return public_job_record(await lookup_job(job_id))

    @web_app.get("/transcriptions/{job_id}/events")
    async def transcription_events(job_id: str) -> Response:
        record = await completed_job(job_id)
        payload = await artifact(record, "events")
        return Response(
            content=payload,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": content_disposition(
                    f"{Path(record['source_name']).stem}.jsonl"
                )
            },
        )

    @web_app.get("/transcriptions/{job_id}/piano-roll")
    async def transcription_piano_roll(job_id: str) -> dict[str, object]:
        record = await completed_job(job_id)
        payload = await artifact(record, "events")
        try:
            notes = pair_note_events(parse_event_stream(payload))
        except ValueError as error:
            raise HTTPException(status_code=500, detail="Stored event data is invalid") from error
        duration = 0.0
        result = record.get("result")
        if isinstance(result, dict) and isinstance(result.get("audio_seconds"), (int, float)):
            duration = float(result["audio_seconds"])
        elif notes:
            duration = max(float(note["end"]) for note in notes)
        return {
            "job_id": job_id,
            "duration": duration,
            "instruments": sorted({str(note["instrument"]) for note in notes}),
            "notes": notes,
        }

    @web_app.get("/transcriptions/{job_id}/midi")
    async def transcription_midi(job_id: str) -> Response:
        record = await completed_job(job_id)
        payload = await artifact(record, "midi")
        return Response(
            content=payload,
            media_type="audio/midi",
            headers={
                "Content-Disposition": content_disposition(
                    f"{Path(record['source_name']).stem}.mid"
                )
            },
        )

    @web_app.get("/transcriptions/{job_id}/score.pdf")
    async def transcription_score_pdf(job_id: str) -> Response:
        record = await completed_job(job_id)
        if not record.get("generate_score"):
            raise HTTPException(status_code=404, detail="This job did not request a score")
        payload = await artifact(record, "score_pdf")
        return Response(
            content=payload,
            media_type="application/pdf",
            headers={
                "Content-Disposition": content_disposition(
                    f"{Path(record['source_name']).stem}-score.pdf",
                    "inline",
                )
            },
        )

    @web_app.get("/transcriptions/{job_id}/musicxml")
    async def transcription_musicxml(job_id: str) -> Response:
        record = await completed_job(job_id)
        if not record.get("generate_score"):
            raise HTTPException(status_code=404, detail="This job did not request a score")
        payload = await artifact(record, "musicxml")
        return Response(
            content=payload,
            media_type="application/vnd.recordare.musicxml+xml",
            headers={
                "Content-Disposition": content_disposition(
                    f"{Path(record['source_name']).stem}-score.musicxml"
                )
            },
        )

    @web_app.get("/transcriptions/{job_id}/audio")
    async def transcription_audio(job_id: str) -> Response:
        record = await lookup_job(job_id)
        payload = await artifact(record, "source")
        media_type = mimetypes.guess_type(record["source_name"])[0] or "application/octet-stream"
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Disposition": content_disposition(record["source_name"], "inline"),
                "Accept-Ranges": "none",
            },
        )

    frontend = _frontend_directory()
    index = frontend / "index.html"

    @web_app.get("/jobs/{job_id}", include_in_schema=False)
    async def job_page(job_id: str) -> FileResponse:
        return FileResponse(index)

    web_app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return web_app


@app.function(
    image=web_image,
    cpu=1.0,
    memory=1024,
    timeout=15 * 60,
    max_containers=WEB_MAX_CONTAINERS,
)
@modal.concurrent(
    max_inputs=WEB_MAX_CONCURRENT_INPUTS,
    target_inputs=WEB_TARGET_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def web() -> Any:
    """Expose the public M2 application from a CPU-only Modal container."""

    return create_web_app()
