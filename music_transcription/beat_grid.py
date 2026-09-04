"""Warm CPU beat tracking for tempo, meter, and onset correction."""

from __future__ import annotations

import time

import modal

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    AUDIO_SAMPLE_RATE,
    BEAT_CHECKPOINT_BYTES,
    BEAT_CHECKPOINT_PATH,
    BEAT_MAX_CONTAINERS,
    BEAT_SCALEDOWN_WINDOW_SECONDS,
    MODEL_MOUNT_PATH,
)
from music_transcription.resources import app, artifact_volume, model_image, model_volume
from music_transcription.schemas import BeatGridDetection
from music_transcription.storage import job_paths, mounted_artifact_path


@app.cls(
    image=model_image,
    cpu=2.0,
    memory=4096,
    max_containers=BEAT_MAX_CONTAINERS,
    min_containers=0,
    scaledown_window=BEAT_SCALEDOWN_WINDOW_SECONDS,
    timeout=30 * 60,
    volumes={
        str(MODEL_MOUNT_PATH): model_volume.with_mount_options(read_only=True),
        str(ARTIFACT_MOUNT_PATH): artifact_volume.with_mount_options(read_only=True),
    },
)
class BeatGridDetector:
    """Keep Beat This! resident on CPU while a worker remains warm."""

    @modal.enter()
    def load_detector(self) -> None:
        from beat_this.inference import Audio2Beats

        if (
            not BEAT_CHECKPOINT_PATH.is_file()
            or BEAT_CHECKPOINT_PATH.stat().st_size != BEAT_CHECKPOINT_BYTES
        ):
            raise RuntimeError(
                "Beat This! is not materialized. Run "
                "`uv run modal run -m music_transcription.models::download_model` first."
            )

        self.detector = Audio2Beats(
            checkpoint_path=str(BEAT_CHECKPOINT_PATH),
            device="cpu",
            dbn=False,
        )

    @modal.method()
    def detect_artifact(self, job_id: str, source_suffix: str) -> BeatGridDetection:
        """Fit a constant-tempo grid, falling back when the recording has none."""

        import numpy as np
        from muscriptor.utils.audio import load_audio
        from muscriptor.utils.beats import (
            MAX_TEMPO_RESIDUAL,
            MIN_BEATS,
            BeatDetectionError,
            fit_tempo,
            infer_beats_per_bar,
        )

        started = time.perf_counter()
        artifact_volume.reload()
        paths = job_paths(job_id, source_suffix)
        wav = load_audio(
            mounted_artifact_path(paths["normalized"]),
            target_sr=AUDIO_SAMPLE_RATE,
        )

        try:
            if wav.shape[-1] < AUDIO_SAMPLE_RATE:
                raise BeatDetectionError(
                    f"Audio is {wav.shape[-1] / AUDIO_SAMPLE_RATE:.2f}s long, "
                    "too short to detect a tempo"
                )

            signal = wav.mean(dim=0).detach().cpu().numpy()
            beats, downbeats = self.detector(signal, AUDIO_SAMPLE_RATE)
            beats = np.asarray(beats, dtype=float)
            downbeats = np.asarray(downbeats, dtype=float)

            if len(beats) < MIN_BEATS:
                raise BeatDetectionError(
                    f"Only {len(beats)} beats detected, need at least {MIN_BEATS}"
                )

            bpm, residual = fit_tempo(beats)
            beat_seconds = 60.0 / bpm
            if residual > MAX_TEMPO_RESIDUAL * beat_seconds:
                raise BeatDetectionError(
                    "The recording has no fixed tempo "
                    f"(beats deviate {residual * 1000:.0f} ms RMS from a constant "
                    f"{bpm:.1f} BPM)"
                )

            beats_per_bar = infer_beats_per_bar(beats, downbeats)
            first_downbeat = float(downbeats[0]) if len(downbeats) else float(beats[0])
            return {
                "seconds": time.perf_counter() - started,
                "grid": {
                    "bpm": float(bpm),
                    "beats_per_bar": beats_per_bar,
                    "first_downbeat": first_downbeat,
                    "beats": [float(beat) for beat in beats],
                },
                "reason": None,
            }
        except BeatDetectionError as error:
            return {
                "seconds": time.perf_counter() - started,
                "grid": None,
                "reason": str(error),
            }
