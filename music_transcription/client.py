"""Local M1 client for submission, polling, and durable artifact download."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import modal

from music_transcription.config import APP_NAME, MAX_M1_BATCH_FILES, SUPPORTED_AUDIO_SUFFIXES
from music_transcription.resources import artifact_volume, job_states
from music_transcription.schemas import JobRecord, JobSpec
from music_transcription.storage import get_job, initial_job_record, job_paths, new_job_spec

TERMINAL_STATES = frozenset({"completed", "failed"})


def parse_instruments(value: str | None) -> list[str] | None:
    """Turn a comma-separated CLI value into MuScriptor instrument names."""

    if value is None:
        return None
    instruments = [item.strip() for item in value.split(",") if item.strip()]
    return instruments or None


def validate_source(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ValueError(f"Unsupported audio type {path.suffix!r}; expected one of: {supported}")
    return path


def discover_sources(directory: str, limit: int) -> list[Path]:
    if not 1 <= limit <= MAX_M1_BATCH_FILES:
        raise ValueError(f"--limit must be between 1 and {MAX_M1_BATCH_FILES}")

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Batch directory does not exist: {root}")
    sources = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not sources:
        raise ValueError(f"No supported audio files found in {root}")
    if len(sources) > limit:
        raise ValueError(
            f"Found {len(sources)} audio files but --limit is {limit}; "
            "raise the limit deliberately or use a smaller directory"
        )
    return sources


def upload_sources(sources: list[Path], instruments: list[str] | None) -> list[JobSpec]:
    """Upload source audio and immutable request metadata in one Volume commit."""

    specs = [new_job_spec(source.name, instruments) for source in sources]
    with artifact_volume.batch_upload() as batch:
        for source, spec in zip(sources, specs, strict=True):
            paths = job_paths(spec["job_id"], spec["source_suffix"])
            request = json.dumps(spec, indent=2, sort_keys=True).encode() + b"\n"
            batch.put_file(source, f"/{paths['source']}")
            batch.put_file(io.BytesIO(request), f"/{paths['request']}")

    for spec in specs:
        job_states[spec["job_id"]] = initial_job_record(spec)
    return specs


def wait_for_jobs(job_ids: list[str], poll_seconds: float = 2.0) -> list[JobRecord]:
    """Poll the durable Dict; no FunctionCall handle is required."""

    previous_states: dict[str, str] = {}
    while True:
        records = [get_job(job_id) for job_id in job_ids]
        for record in records:
            state = record["state"]
            if previous_states.get(record["job_id"]) != state:
                print(f"{record['job_id']}  {state}")
                previous_states[record["job_id"]] = state
        if all(record["state"] in TERMINAL_STATES for record in records):
            return records
        time.sleep(poll_seconds)


def print_submission(specs: list[JobSpec], call_id: str | None = None) -> None:
    response: dict[str, object] = {"job_ids": [spec["job_id"] for spec in specs]}
    if call_id is not None:
        response["function_call_id"] = call_id
    print(json.dumps(response, indent=2))
    print("\nThe jobs are durable; this command can exit while Modal continues processing them.")


def submit_one(audio: str, instruments: str | None, wait: bool) -> None:
    specs = upload_sources([validate_source(audio)], parse_instruments(instruments))
    process_job = modal.Function.from_name(APP_NAME, "process_job")
    call = process_job.spawn(specs[0])
    print_submission(specs, call.object_id)
    if wait:
        records = wait_for_jobs([specs[0]["job_id"]])
        print(json.dumps(records[0], indent=2, sort_keys=True))


def submit_batch(directory: str, instruments: str | None, limit: int, wait: bool) -> None:
    sources = discover_sources(directory, limit)
    specs = upload_sources(sources, parse_instruments(instruments))
    process_job = modal.Function.from_name(APP_NAME, "process_job")
    process_job.spawn_map(specs)
    print_submission(specs)
    if wait:
        records = wait_for_jobs([spec["job_id"] for spec in specs])
        print(json.dumps(records, indent=2, sort_keys=True))


def show_status(job_id: str) -> None:
    print(json.dumps(get_job(job_id), indent=2, sort_keys=True))


def download_artifacts(job_id: str, output_dir: str) -> None:
    record = get_job(job_id)
    if record["state"] != "completed":
        raise RuntimeError(f"Job {job_id} is {record['state']!r}, not 'completed'")

    destination = Path(output_dir).expanduser().resolve() / job_id
    destination.mkdir(parents=True, exist_ok=True)
    for artifact_name in ("events", "midi", "metrics"):
        remote_path = record["paths"][artifact_name]
        local_path = destination / Path(remote_path).name
        with local_path.open("wb") as output:
            for chunk in artifact_volume.read_file(remote_path):
                output.write(chunk)
    print(f"Artifacts written to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable MuScriptor jobs on Modal")
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="submit one audio file")
    submit.add_argument("--audio", required=True)
    submit.add_argument("--instruments")
    submit.add_argument("--wait", action="store_true")

    batch = commands.add_parser("submit-batch", help="submit one directory of audio files")
    batch.add_argument("--directory", required=True)
    batch.add_argument("--instruments")
    batch.add_argument("--limit", type=int, default=MAX_M1_BATCH_FILES)
    batch.add_argument("--wait", action="store_true")

    status = commands.add_parser("status", help="read persistent job state")
    status.add_argument("--job-id", required=True)

    download = commands.add_parser("download", help="download completed MIDI and metadata")
    download.add_argument("--job-id", required=True)
    download.add_argument("--output-dir", default="outputs/m1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "submit":
        submit_one(args.audio, args.instruments, args.wait)
    elif args.command == "submit-batch":
        submit_batch(args.directory, args.instruments, args.limit, args.wait)
    elif args.command == "status":
        show_status(args.job_id)
    elif args.command == "download":
        download_artifacts(args.job_id, args.output_dir)


if __name__ == "__main__":
    main()
