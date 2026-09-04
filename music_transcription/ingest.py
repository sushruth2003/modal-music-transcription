"""Restricted public-media ingestion for URL-backed jobs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    MAX_AUDIO_SECONDS,
    SUPPORTED_YOUTUBE_HOSTS,
    URL_DOWNLOAD_TIMEOUT_SECONDS,
    WEB_MAX_UPLOAD_BYTES,
)
from music_transcription.resources import app, artifact_volume, ingest_image
from music_transcription.storage import job_paths, mounted_artifact_path


def validate_media_url(value: str) -> str:
    """Accept only ordinary HTTPS YouTube page URLs."""

    if len(value) > 2_048:
        raise ValueError("The YouTube URL is too long")
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError("YouTube URLs must use HTTPS")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("YouTube URLs cannot contain credentials or a custom port")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname not in SUPPORTED_YOUTUBE_HOSTS:
        raise ValueError("Only public YouTube URLs are supported")
    if not parsed.path or parsed.path == "/":
        raise ValueError("The URL must point to a specific YouTube video")
    return urlunsplit(("https", hostname, parsed.path, parsed.query, ""))


def download_command(source_url: str, destination: Path) -> list[str]:
    """Build a bounded, single-item yt-dlp invocation with deterministic output."""

    output_template = str(destination.with_suffix(".%(ext)s"))
    return [
        "yt-dlp",
        "--no-playlist",
        "--max-downloads",
        "1",
        "--match-filter",
        # `<=?` accepts missing pre-download duration metadata; ffprobe enforces
        # the real limit after download and before any GPU work starts.
        f"!is_live & duration <=? {MAX_AUDIO_SECONDS}",
        "--max-filesize",
        str(WEB_MAX_UPLOAD_BYTES),
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "--no-part",
        "--force-overwrites",
        "--extract-audio",
        "--audio-format",
        "flac",
        "--audio-quality",
        "0",
        "--output",
        output_template,
        "--print",
        "after_move:%(title)j\t%(filepath)j",
        source_url,
    ]


def safe_source_name(title: str) -> str:
    """Turn untrusted remote metadata into a harmless display/download name."""

    printable = "".join(
        " " if not character.isprintable() or character in {"/", "\\", "\x00"} else character
        for character in title
    )
    compact = re.sub(r"\s+", " ", printable).strip(" .")
    return f"{(compact or 'Imported recording')[:180]}.flac"


def download_media(source_url: str, destination: Path) -> dict[str, object]:
    """Download one public post and return sanitized source metadata."""

    normalized_url = validate_media_url(source_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        download_command(normalized_url, destination),
        capture_output=True,
        text=True,
        check=False,
        timeout=URL_DOWNLOAD_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not import that media. Confirm it is public, is not a playlist or live "
            "stream, and is no longer than 10 minutes."
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("The media importer did not produce an audio file")
    if destination.stat().st_size > WEB_MAX_UPLOAD_BYTES:
        raise ValueError("Imported audio exceeds the 100 MB public demo limit")

    title = "Imported recording"
    for line in reversed(completed.stdout.splitlines()):
        if "\t" not in line:
            continue
        encoded_title, _encoded_path = line.split("\t", 1)
        try:
            candidate = json.loads(encoded_title)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, str) and candidate.strip():
            title = candidate.strip()[:180]
            break
    return {
        "source_name": safe_source_name(title),
        "source_bytes": destination.stat().st_size,
        "source_url": normalized_url,
    }


@app.function(
    image=ingest_image,
    cpu=2.0,
    memory=2048,
    timeout=URL_DOWNLOAD_TIMEOUT_SECONDS,
    max_containers=4,
    volumes={str(ARTIFACT_MOUNT_PATH): artifact_volume},
)
def fetch_media_url(job_id: str, source_suffix: str, source_url: str) -> dict[str, object]:
    """Materialize a URL job's source audio into the shared artifact Volume."""

    paths = job_paths(job_id, source_suffix)
    artifact_volume.reload()
    metadata = download_media(source_url, mounted_artifact_path(paths["source"]))
    artifact_volume.commit()
    return metadata
