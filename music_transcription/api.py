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
    MUSCRIPTOR_INSTRUMENT_GROUPS,
    PUBLIC_BETA_JOB_RESERVATION_USD,
    PUBLIC_BETA_MONTHLY_BUDGET_USD,
    SUPPORTED_VIDEO_SUFFIXES,
    WEB_MAX_CONCURRENT_INPUTS,
    WEB_MAX_CONTAINERS,
    WEB_MAX_UPLOAD_BYTES,
    WEB_RATE_LIMIT_WINDOW_SECONDS,
    WEB_SUBMISSIONS_GLOBAL_DAY,
    WEB_SUBMISSIONS_PER_IP_HOUR,
    WEB_TARGET_CONCURRENT_INPUTS,
    WEB_UPLOAD_CHUNK_BYTES,
)
from music_transcription.preprocess import process_job
from music_transcription.resources import app, artifact_volume, rate_limit_states, web_image
from music_transcription.schemas import JobRecord, JobSpec, SerializedEvent
from music_transcription.storage import (
    get_job,
    new_job_spec,
    parse_instruments,
    stage_job_sources,
    update_job,
)

LOCAL_FRONTEND_PATH = Path(__file__).parent / "frontend"
TERMINAL_STATES = frozenset({"completed", "failed"})
STATE_PROGRESS = {
    "submitted": 10,
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
    budget_remaining_usd: float
    reason: str | None = None


class InvalidByteRangeError(ValueError):
    """Raised when an HTTP byte range cannot be satisfied."""


def parse_byte_range(header: str, total_bytes: int) -> tuple[int, int]:
    """Parse one RFC 9110 byte range and return inclusive start/end offsets."""

    if total_bytes <= 0 or not header.startswith("bytes="):
        raise InvalidByteRangeError("Invalid byte range")
    value = header.removeprefix("bytes=").strip()
    if not value or "," in value or "-" not in value:
        raise InvalidByteRangeError("Only one byte range is supported")
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix_bytes = int(end_text)
            if suffix_bytes <= 0:
                raise InvalidByteRangeError("Invalid suffix byte range")
            return max(0, total_bytes - suffix_bytes), total_bytes - 1

        start = int(start_text)
        end = total_bytes - 1 if not end_text else int(end_text)
    except ValueError as error:
        raise InvalidByteRangeError("Invalid byte range") from error
    if start < 0 or end < start or start >= total_bytes:
        raise InvalidByteRangeError("Unsatisfiable byte range")
    return start, min(end, total_bytes - 1)


def evaluate_submission_limit(
    stored: dict[str, object] | None,
    client_key: str,
    now: float,
) -> tuple[dict[str, object], RateLimitDecision]:
    """Apply rolling, daily, and monthly public-beta limits to a persisted record."""

    current = datetime.fromtimestamp(now, UTC)
    day = current.date().isoformat()
    month = day[:7]
    same_day = stored is not None and stored.get("day") == day
    same_month = stored is not None and stored.get("month") == month
    state: dict[str, object] = {
        "day": day,
        "month": month,
        "global_count": int(stored.get("global_count", 0)) if same_day else 0,
        "monthly_reserved_usd": (
            float(stored.get("monthly_reserved_usd", 0.0)) if same_month else 0.0
        ),
        "clients": dict(stored.get("clients", {})) if same_day else {},
    }

    clients = state["clients"]
    assert isinstance(clients, dict)
    cutoff = now - WEB_RATE_LIMIT_WINDOW_SECONDS
    events = [
        float(timestamp) for timestamp in clients.get(client_key, []) if float(timestamp) > cutoff
    ]
    clients[client_key] = events
    global_count = int(state["global_count"])
    monthly_reserved = float(state["monthly_reserved_usd"])
    budget_remaining = max(0.0, PUBLIC_BETA_MONTHLY_BUDGET_USD - monthly_reserved)

    if monthly_reserved + PUBLIC_BETA_JOB_RESERVATION_USD > PUBLIC_BETA_MONTHLY_BUDGET_USD:
        if current.month == 12:
            next_month = current.replace(
                year=current.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            next_month = current.replace(
                month=current.month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        retry_after = max(1, int((next_month - current).total_seconds()))
        return state, RateLimitDecision(
            False,
            retry_after,
            WEB_SUBMISSIONS_PER_IP_HOUR - len(events),
            WEB_SUBMISSIONS_GLOBAL_DAY - global_count,
            round(budget_remaining, 2),
            "monthly_budget",
        )

    if global_count >= WEB_SUBMISSIONS_GLOBAL_DAY:
        next_day = datetime.combine(current.date() + timedelta(days=1), datetime.min.time(), UTC)
        retry_after = max(1, int((next_day - current).total_seconds()))
        return state, RateLimitDecision(
            False,
            retry_after,
            WEB_SUBMISSIONS_PER_IP_HOUR - len(events),
            0,
            round(budget_remaining, 2),
            "daily_limit",
        )
    if len(events) >= WEB_SUBMISSIONS_PER_IP_HOUR:
        retry_after = max(1, int(events[0] + WEB_RATE_LIMIT_WINDOW_SECONDS - now) + 1)
        return state, RateLimitDecision(
            False,
            retry_after,
            0,
            WEB_SUBMISSIONS_GLOBAL_DAY - global_count,
            round(budget_remaining, 2),
            "hourly_limit",
        )

    events.append(now)
    state["global_count"] = global_count + 1
    state["monthly_reserved_usd"] = monthly_reserved + PUBLIC_BETA_JOB_RESERVATION_USD
    return state, RateLimitDecision(
        True,
        0,
        WEB_SUBMISSIONS_PER_IP_HOUR - len(events),
        WEB_SUBMISSIONS_GLOBAL_DAY - global_count - 1,
        round(budget_remaining - PUBLIC_BETA_JOB_RESERVATION_USD, 2),
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
                raise UploadTooLargeError(f"Media files are limited to {max_bytes} bytes")
            output.write(chunk)
    if total == 0:
        raise ValueError("The uploaded media file is empty")
    return total


def validate_instruments(value: str | None) -> list[str] | None:
    return parse_instruments(value)


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
        description="Asynchronous audio/video-to-MIDI transcription",
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
                details = {
                    "monthly_budget": "This month's public beta compute allowance is full",
                    "daily_limit": "Today's public beta quota is full",
                    "hourly_limit": "This network has reached its hourly submission limit",
                }
                detail = details.get(decision.reason, "The public beta quota is full")
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
                        "X-Public-Budget-Remaining": f"{decision.budget_remaining_usd:.2f}",
                        "Cache-Control": "no-store",
                    },
                )

        response = await call_next(request)
        if request.url.path.startswith("/transcriptions"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path in {
            "/",
            "/index.html",
            "/how-it-works",
            "/how-it-works.html",
            "/app.js",
            "/styles.css",
        } or request.url.path.startswith("/jobs/"):
            response.headers["Cache-Control"] = "no-cache"
        if decision is not None:
            response.headers["X-RateLimit-IP-Remaining"] = str(decision.ip_remaining)
            response.headers["X-RateLimit-Global-Remaining"] = str(decision.global_remaining)
            response.headers["X-Public-Budget-Remaining"] = f"{decision.budget_remaining_usd:.2f}"
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

    @web_app.get("/api/instruments")
    async def instruments() -> dict[str, object]:
        return {
            "groups": [
                {
                    "label": group,
                    "options": [
                        {
                            "value": name,
                            "label": name.replace("_and_", " & ").replace("_", " ").title(),
                        }
                        for name in names
                    ],
                }
                for group, names in MUSCRIPTOR_INSTRUMENT_GROUPS
            ]
        }

    @web_app.post("/transcriptions", status_code=status.HTTP_202_ACCEPTED)
    async def submit_transcription(
        request: Request,
        media: Annotated[UploadFile | None, File()] = None,
        instruments: Annotated[str | None, Form()] = None,
        generate_score: Annotated[bool, Form()] = False,
    ) -> JSONResponse:
        if media is None:
            raise HTTPException(status_code=400, detail="Upload one audio or video file")

        try:
            hints = validate_instruments(instruments)
            content_length = request.headers.get("content-length")
            if (
                content_length
                and content_length.isdigit()
                and int(content_length) > WEB_MAX_UPLOAD_BYTES + WEB_UPLOAD_CHUNK_BYTES
            ):
                raise UploadTooLargeError("Media upload is too large")
            if not media.filename:
                raise ValueError("The upload needs a filename")
            await media.seek(0)
            spec, source_bytes = await asyncio.to_thread(
                stage_uploaded_file,
                media.file,
                media.filename,
                hints,
                generate_score,
            )
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            await media.close()

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
            "generate_score": generate_score,
            "function_call_id": call_id,
            "status_url": f"/transcriptions/{job_id}",
            "result_url": f"/jobs/{job_id}",
        }
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

    @web_app.get("/transcriptions/{job_id}/audio")
    async def transcription_audio(job_id: str, request: Request) -> Response:
        record = await lookup_job(job_id)
        source_suffix = Path(record["paths"]["source"]).suffix.lower()
        if source_suffix in SUPPORTED_VIDEO_SUFFIXES:
            payload = await artifact(record, "normalized")
            playback_name = f"{Path(record['source_name']).stem}.wav"
            media_type = "audio/wav"
        else:
            payload = await artifact(record, "source")
            playback_name = record["source_name"]
            media_type = mimetypes.guess_type(playback_name)[0] or "application/octet-stream"
        total_bytes = len(payload)
        headers = {
            "Content-Disposition": content_disposition(playback_name, "inline"),
            "Accept-Ranges": "bytes",
        }
        range_header = request.headers.get("range")
        if not range_header:
            headers["Content-Length"] = str(total_bytes)
            return Response(content=payload, media_type=media_type, headers=headers)
        try:
            start, end = parse_byte_range(range_header, total_bytes)
        except InvalidByteRangeError:
            return Response(
                status_code=416,
                headers={
                    **headers,
                    "Content-Range": f"bytes */{total_bytes}",
                    "Content-Length": "0",
                },
            )
        partial = payload[start : end + 1]
        headers.update(
            {
                "Content-Range": f"bytes {start}-{end}/{total_bytes}",
                "Content-Length": str(len(partial)),
            }
        )
        return Response(content=partial, status_code=206, media_type=media_type, headers=headers)

    frontend = _frontend_directory()
    index = frontend / "index.html"
    how_it_works = frontend / "how-it-works.html"

    @web_app.get("/jobs/{job_id}", include_in_schema=False)
    async def job_page(job_id: str) -> FileResponse:
        return FileResponse(index)

    @web_app.get("/how-it-works", include_in_schema=False)
    async def how_it_works_page() -> FileResponse:
        return FileResponse(how_it_works)

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
@modal.asgi_app(label="transcribe")
def web() -> Any:
    """Expose the public M2 application from a CPU-only Modal container."""

    return create_web_app()
