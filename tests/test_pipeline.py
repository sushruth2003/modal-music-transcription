from __future__ import annotations

import pytest

from music_transcription.client import discover_sources, parse_instruments, validate_source


def test_parse_instruments() -> None:
    assert parse_instruments(None) is None
    assert parse_instruments("  piano, drums ,, bass ") == ["piano", "drums", "bass"]
    assert parse_instruments(" , ") is None


def test_discover_sources_is_sorted_and_filters_unsupported_files(tmp_path) -> None:
    (tmp_path / "b.wav").write_bytes(b"b")
    (tmp_path / "a.mp3").write_bytes(b"a")
    (tmp_path / "c.mp4").write_bytes(b"c")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in discover_sources(str(tmp_path), limit=3)] == [
        "a.mp3",
        "b.wav",
        "c.mp4",
    ]


def test_discover_sources_requires_deliberate_batch_limit(tmp_path) -> None:
    (tmp_path / "a.wav").write_bytes(b"a")
    (tmp_path / "b.wav").write_bytes(b"b")

    with pytest.raises(ValueError, match="raise the limit deliberately"):
        discover_sources(str(tmp_path), limit=1)


def test_validate_source(tmp_path) -> None:
    audio = tmp_path / "demo.FLAC"
    audio.write_bytes(b"audio")
    assert validate_source(str(audio)) == audio.resolve()

    with pytest.raises(ValueError, match="does not exist"):
        validate_source(str(tmp_path / "missing.wav"))

    video = tmp_path / "performance.WEBM"
    video.write_bytes(b"video")
    assert validate_source(str(video)) == video.resolve()
