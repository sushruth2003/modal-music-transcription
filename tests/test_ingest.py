from __future__ import annotations

from pathlib import Path

import pytest

from music_transcription.ingest import (
    download_command,
    download_error_message,
    safe_source_name,
    validate_media_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/abc123",
        "https://www.youtube.com/watch?v=abc123",
    ],
)
def test_validate_media_url_accepts_supported_public_pages(url: str) -> None:
    assert validate_media_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://youtu.be/abc123",
        "https://www.instagram.com/reel/abc123/",
        "https://example.com/video",
        "https://youtube.com.evil.example/watch?v=abc123",
        "https://user:password@youtube.com/watch?v=abc123",
        "https://youtube.com:8443/watch?v=abc123",
    ],
)
def test_validate_media_url_rejects_unsafe_or_unsupported_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_media_url(url)


def test_download_command_is_bounded_and_deterministic() -> None:
    command = download_command("https://youtu.be/abc123", Path("/artifacts/source.flac"))

    assert "--no-playlist" in command
    # yt-dlp exits with code 101 when --max-downloads reaches its limit, even
    # after successfully producing the first file. --no-playlist is sufficient.
    assert "--max-downloads" not in command
    assert command[command.index("--match-filter") + 1] == "!is_live & duration <=? 600"
    assert command[command.index("--audio-format") + 1] == "flac"
    assert command[command.index("--output") + 1] == "/artifacts/source.%(ext)s"


def test_safe_source_name_removes_path_and_header_characters() -> None:
    assert safe_source_name("  Song / demo\\take\n2  ") == "Song demo take 2.flac"


def test_download_error_explains_youtube_bot_block() -> None:
    message = download_error_message("Sign in to confirm you’re not a bot")

    assert "YouTube blocked the server download" in message
    assert "upload the file instead" in message
