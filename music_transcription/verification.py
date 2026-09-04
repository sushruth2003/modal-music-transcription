"""Cheap remote checks to run before downloading weights or requesting a GPU."""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import tempfile
import wave
from importlib.metadata import version
from pathlib import Path

from music_transcription.config import AUDIO_SAMPLE_RATE, BEAT_PACKAGE, MODEL_PACKAGE
from music_transcription.resources import app, audio_image, model_image


def silent_wav_bytes(duration_seconds: float = 0.25) -> bytes:
    """Create a small valid WAV without adding an audio dependency."""

    frame_count = round(AUDIO_SAMPLE_RATE * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(AUDIO_SAMPLE_RATE)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


@app.function(image=audio_image, cpu=1.0, memory=512, timeout=5 * 60)
def verify_audio_image() -> dict[str, str | float | int]:
    """Prove that the CPU Image can invoke ffmpeg and normalize audio."""

    from music_transcription.preprocess import normalize_audio_file

    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="audio-preflight-") as directory:
        source_path = Path(directory) / "preflight.wav"
        normalized_path = Path(directory) / "normalized.wav"
        source_path.write_bytes(silent_wav_bytes())
        audio_seconds, metrics = normalize_audio_file(source_path, normalized_path)
    result: dict[str, str | float | int] = {
        "ffmpeg": ffmpeg.stdout.splitlines()[0],
        "audio_seconds": audio_seconds,
        "sample_rate": metrics["sample_rate"],
        "channels": metrics["channels"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


@app.function(image=model_image, cpu=2.0, memory=2048, timeout=10 * 60)
def verify_model_image() -> dict[str, str | bool]:
    """Prove that MuScriptor, Beat This!, and their dependencies import on CPU."""

    import torch
    from beat_this.inference import Audio2Beats
    from muscriptor import TranscriptionModel
    from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
    from muscriptor.utils.beats import BeatGrid

    expected_version = MODEL_PACKAGE.partition("==")[2]
    expected_beat_version = BEAT_PACKAGE.partition("==")[2]
    installed_version = version("muscriptor")
    installed_beat_version = version("beat-this")
    result: dict[str, str | bool] = {
        "muscriptor_version": installed_version,
        "beat_this_version": installed_beat_version,
        "version_is_pinned": installed_version == expected_version,
        "beat_this_version_is_pinned": installed_beat_version == expected_beat_version,
        # TorchVersion is a str subclass that requires torch when unpickled.
        # Convert all framework metadata to built-in values at the boundary.
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda or "none"),
        "load_model_api": hasattr(TranscriptionModel, "load_model"),
        "transcribe_api": hasattr(TranscriptionModel, "transcribe"),
        "midi_api": hasattr(TranscriptionModel, "events_to_midi_bytes"),
        "beat_detector_api": callable(Audio2Beats),
        "beat_grid_onset_api": hasattr(BeatGrid, "with_onset_delay"),
        "load_model_signature": str(inspect.signature(TranscriptionModel.load_model)),
        "transcribe_signature": str(inspect.signature(TranscriptionModel.transcribe)),
        "midi_signature": str(inspect.signature(TranscriptionModel.events_to_midi_bytes)),
        "event_types": ",".join(
            event_type.__name__ for event_type in (NoteStartEvent, NoteEndEvent, ProgressEvent)
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return result
