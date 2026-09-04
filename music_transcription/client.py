"""Local M1 client for submission, polling, and durable artifact download."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

from music_transcription.config import APP_NAME, MAX_M1_BATCH_FILES, SUPPORTED_AUDIO_SUFFIXES
from music_transcription.ingest import validate_media_url
from music_transcription.resources import artifact_volume
from music_transcription.schemas import JobRecord, JobSpec
from music_transcription.storage import (
    get_job,
    new_job_spec,
    new_url_job_spec,
    parse_instruments,
    stage_job_sources,
    stage_url_job,
)

TERMINAL_STATES = frozenset({"completed", "failed"})


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


def upload_sources(
    sources: list[Path],
    instruments: list[str] | None,
    *,
    generate_score: bool = False,
) -> list[JobSpec]:
    """Upload source audio and immutable request metadata in one Volume commit."""

    specs = [
        new_job_spec(source.name, instruments, generate_score=generate_score)
        for source in sources
    ]
    stage_job_sources(sources, specs)
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


def submit_one(
    audio: str | None,
    source_url: str | None,
    instruments: str | None,
    generate_score: bool,
    wait: bool,
) -> None:
    hints = parse_instruments(instruments)
    if source_url is not None:
        spec = new_url_job_spec(
            validate_media_url(source_url),
            hints,
            generate_score=generate_score,
        )
        stage_url_job(spec)
        specs = [spec]
    else:
        assert audio is not None
        specs = upload_sources(
            [validate_source(audio)],
            hints,
            generate_score=generate_score,
        )
    process_job = modal.Function.from_name(APP_NAME, "process_job")
    call = process_job.spawn(specs[0])
    print_submission(specs, call.object_id)
    if wait:
        records = wait_for_jobs([specs[0]["job_id"]])
        print(json.dumps(records[0], indent=2, sort_keys=True))


def submit_batch(
    directory: str,
    instruments: str | None,
    limit: int,
    generate_score: bool,
    wait: bool,
) -> None:
    sources = discover_sources(directory, limit)
    specs = upload_sources(
        sources,
        parse_instruments(instruments),
        generate_score=generate_score,
    )
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
    artifact_names = ["events", "midi", "metrics"]
    if record.get("generate_score"):
        artifact_names.extend(["score_pdf", "musicxml"])
    for artifact_name in artifact_names:
        remote_path = record["paths"][artifact_name]
        local_path = destination / Path(remote_path).name
        with local_path.open("wb") as output:
            for chunk in artifact_volume.read_file(remote_path):
                output.write(chunk)
    print(f"Artifacts written to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable MuScriptor jobs on Modal")
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="submit one audio file or public YouTube URL")
    source = submit.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio")
    source.add_argument("--url")
    submit.add_argument("--instruments")
    submit.add_argument("--score", action="store_true", help="also render PDF and MusicXML")
    submit.add_argument("--wait", action="store_true")

    batch = commands.add_parser("submit-batch", help="submit one directory of audio files")
    batch.add_argument("--directory", required=True)
    batch.add_argument("--instruments")
    batch.add_argument("--limit", type=int, default=MAX_M1_BATCH_FILES)
    batch.add_argument("--score", action="store_true", help="also render PDF and MusicXML")
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
        submit_one(args.audio, args.url, args.instruments, args.score, args.wait)
    elif args.command == "submit-batch":
        submit_batch(args.directory, args.instruments, args.limit, args.score, args.wait)
    elif args.command == "status":
        show_status(args.job_id)
    elif args.command == "download":
        download_artifacts(args.job_id, args.output_dir)


if __name__ == "__main__":
    main()
