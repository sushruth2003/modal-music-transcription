from __future__ import annotations

import io
import wave

import pytest

from music_transcription.preprocess import (
    audio_duration_path,
    normalize_audio_file,
)


def wav_bytes(frame_count: int, frame_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def test_audio_duration_path(tmp_path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(wav_bytes(4_000))
    assert audio_duration_path(wav_path, "sample.wav") == 0.25


def test_audio_duration_path_rejects_empty_audio(tmp_path) -> None:
    wav_path = tmp_path / "empty.wav"
    wav_path.write_bytes(wav_bytes(0))
    with pytest.raises(ValueError, match="contains no samples"):
        audio_duration_path(wav_path, "empty.wav")


def test_normalize_rejects_unsupported_suffix_before_ffmpeg(tmp_path) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported source suffix"):
        normalize_audio_file(source_path, tmp_path / "normalized.wav")
