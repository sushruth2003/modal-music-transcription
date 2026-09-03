"""CPU-only FastAPI service for durable transcription jobs and artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import tempfile
from collections.abc import Iterable
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
    WEB_ACCESS_TOKEN_ENV,
    WEB_MAX_CONCURRENT_INPUTS,
    WEB_MAX_CONTAINERS,
    WEB_MAX_UPLOAD_BYTES,
    WEB_SESSION_COOKIE,
    WEB_SESSION_SECONDS,
    WEB_TARGET_CONCURRENT_INPUTS,
    WEB_UPLOAD_CHUNK_BYTES,
)
from music_transcription.preprocess import process_job
from music_transcription.resources import app, artifact_volume, web_image, web_secret
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
    "completed": 100,
    "failed": 100,
}
SESSION_PURPOSE = b"music-transcription-browser-session-v1"


class UploadTooLargeError(ValueError):
    """Raised after a streamed upload crosses the configured byte limit."""


def configured_access_token() -> str:
    """Return the server-side access token injected by a Modal Secret."""

    token = os.environ.get(WEB_ACCESS_TOKEN_ENV, "")
    if len(token) < 24:
        raise RuntimeError(f"{WEB_ACCESS_TOKEN_ENV} must contain at least 24 characters")
    return token


def browser_session_token(access_token: str) -> str:
    """Derive a cookie value without putting the access token itself in the cookie."""

    return hmac.new(access_token.encode(), SESSION_PURPOSE, hashlib.sha256).hexdigest()


def request_is_authenticated(request: Request) -> bool:
    """Accept either the HttpOnly browser session or an API bearer token."""

    access_token = configured_access_token()
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        return hmac.compare_digest(authorization[7:], access_token)
    session = request.cookies.get(WEB_SESSION_COOKIE, "")
    return hmac.compare_digest(session, browser_session_token(access_token))


def require_authentication(request: Request) -> None:
    if not request_is_authenticated(request):
        raise HTTPException(
            status_code=401,
            detail="Sign in to use this transcription workspace",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
) -> tuple[JobSpec, int]:
    """Spool one request to ephemeral disk, then commit it to the artifact Volume."""

    spec = new_job_spec(source_name, instruments)
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


def create_web_app() -> FastAPI:
    """Construct the ASGI app. Kept separate so HTTP behavior is unit-testable."""

    web_app = FastAPI(
        title="MuScriptor on Modal",
        description="Asynchronous audio-to-MIDI transcription",
        version="0.2.0",
    )

    @web_app.middleware("http")
    async def prevent_private_response_caching(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        if request.url.path.startswith(("/auth/", "/transcriptions")):
            response.headers["Cache-Control"] = "no-store"
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

    @web_app.get("/auth/session")
    async def auth_session(request: Request) -> dict[str, bool]:
        return {"authenticated": request_is_authenticated(request)}

    @web_app.post("/auth/session")
    async def create_auth_session(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="Expected a JSON request") from error
        supplied = payload.get("access_token") if isinstance(payload, dict) else None
        access_token = configured_access_token()
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, access_token):
            raise HTTPException(status_code=401, detail="Incorrect access code")

        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            key=WEB_SESSION_COOKIE,
            value=browser_session_token(access_token),
            max_age=WEB_SESSION_SECONDS,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @web_app.delete("/auth/session")
    async def delete_auth_session() -> JSONResponse:
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(
            key=WEB_SESSION_COOKIE,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @web_app.post("/transcriptions", status_code=status.HTTP_202_ACCEPTED)
    async def submit_transcription(
        request: Request,
        audio: Annotated[UploadFile, File()],
        instruments: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        require_authentication(request)
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > WEB_MAX_UPLOAD_BYTES + WEB_UPLOAD_CHUNK_BYTES
        ):
            raise HTTPException(status_code=413, detail="Audio upload is too large")
        if not audio.filename:
            raise HTTPException(status_code=400, detail="The upload needs a filename")

        try:
            hints = validate_instruments(instruments)
            await audio.seek(0)
            spec, source_bytes = await asyncio.to_thread(
                stage_uploaded_file,
                audio.file,
                audio.filename,
                hints,
            )
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
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
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "state": "submitted",
                "source_bytes": source_bytes,
                "function_call_id": call_id,
                "status_url": f"/transcriptions/{job_id}",
                "result_url": f"/jobs/{job_id}",
            },
            headers={"Location": f"/transcriptions/{job_id}", "Retry-After": "2"},
        )

    @web_app.get("/transcriptions/{job_id}")
    async def transcription_status(job_id: str, request: Request) -> dict[str, object]:
        require_authentication(request)
        return public_job_record(await lookup_job(job_id))

    @web_app.get("/transcriptions/{job_id}/events")
    async def transcription_events(job_id: str, request: Request) -> Response:
        require_authentication(request)
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
    async def transcription_piano_roll(job_id: str, request: Request) -> dict[str, object]:
        require_authentication(request)
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
    async def transcription_midi(job_id: str, request: Request) -> Response:
        require_authentication(request)
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

    @web_app.get("/transcriptions/{job_id}/audio")
    async def transcription_audio(job_id: str, request: Request) -> Response:
        require_authentication(request)
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
    secrets=[web_secret],
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
