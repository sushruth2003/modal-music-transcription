"""L4-backed MuScriptor inference and note-event serialization."""

from __future__ import annotations

import json
import time
import uuid

import modal

from music_transcription.config import (
    ARTIFACT_MOUNT_PATH,
    GPU_MAX_CONTAINERS,
    GPU_SCALEDOWN_WINDOW_SECONDS,
    GPU_TYPE,
    MODEL_CHECKPOINT_PATH,
    MODEL_MOUNT_PATH,
    MODEL_READY_PATH,
    MODEL_REVISION,
)
from music_transcription.resources import app, artifact_volume, model_image, model_volume
from music_transcription.schemas import BeatGridDetection, SerializedEvent
from music_transcription.storage import job_paths, mounted_artifact_path


def corrected_event_time(seconds: float, onset_delay_seconds: float) -> float:
    """Apply MuScriptor's measured global onset correction to an event time."""

    return float(seconds) - onset_delay_seconds


def _serialize_event(event: object, onset_delay_seconds: float = 0.0) -> SerializedEvent:
    """Convert MuScriptor's dataclass events into stable JSON records."""

    from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

    if isinstance(event, NoteStartEvent):
        return {
            "type": "note_start",
            "index": event.index,
            "pitch": event.pitch,
            "instrument": event.instrument,
            "time": corrected_event_time(event.start_time, onset_delay_seconds),
        }
    if isinstance(event, NoteEndEvent):
        return {
            "type": "note_end",
            "index": event.start_event_index,
            "time": corrected_event_time(event.end_time, onset_delay_seconds),
        }
    if isinstance(event, ProgressEvent):
        return {
            "type": "progress",
            "completed": event.completed,
            "total": event.total,
        }
    raise TypeError(f"Unexpected MuScriptor event type: {type(event).__name__}")


@app.cls(
    image=model_image,
    gpu=GPU_TYPE,
    max_containers=GPU_MAX_CONTAINERS,
    min_containers=0,
    scaledown_window=GPU_SCALEDOWN_WINDOW_SECONDS,
    timeout=30 * 60,
    volumes={
        str(MODEL_MOUNT_PATH): model_volume.with_mount_options(read_only=True),
        str(ARTIFACT_MOUNT_PATH): artifact_volume,
    },
)
class MuScriptorTranscriber:
    """One MuScriptor Large instance per temporary L4 container."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the pinned local checkpoint once for this container."""

        import torch
        from muscriptor import TranscriptionModel

        if not MODEL_READY_PATH.is_file() or not MODEL_CHECKPOINT_PATH.is_file():
            raise RuntimeError(
                "MuScriptor Large is not materialized. Run "
                "`uv run modal run -m music_transcription.models::download_model` first."
            )

        ready = json.loads(MODEL_READY_PATH.read_text(encoding="utf-8"))
        if ready.get("revision") != MODEL_REVISION:
            raise RuntimeError("Model READY marker does not match the pinned revision")

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.model = TranscriptionModel.load_model(
            MODEL_CHECKPOINT_PATH,
            device="cuda",
            dtype="float16",
        )
        torch.cuda.synchronize()

        self.container_id = uuid.uuid4().hex[:12]
        self.model_load_seconds = time.perf_counter() - started
        self.model_memory_bytes = torch.cuda.memory_allocated()
        self.gpu_name = torch.cuda.get_device_name(0)

    @modal.method()
    def transcribe_artifact(
        self,
        job_id: str,
        source_suffix: str,
        instruments: list[str] | None = None,
        beat_detection: BeatGridDetection | None = None,
    ) -> dict[str, object]:
        """Read normalized audio and commit MIDI/events without returning bytes."""

        import numpy as np
        import torch
        from muscriptor.events import NoteStartEvent
        from muscriptor.utils.beats import BeatGrid

        paths = job_paths(job_id, source_suffix)
        artifact_volume.reload()
        wav_path = mounted_artifact_path(paths["normalized"])
        preprocessing = json.loads(
            mounted_artifact_path(paths["preprocessing"]).read_text(encoding="utf-8")
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw_events = list(
            self.model.transcribe(
                wav_path,
                instruments=instruments,
                use_sampling=False,
                beam_size=1,
                prelude_forcing=True,
            )
        )
        note_starts = [event for event in raw_events if isinstance(event, NoteStartEvent)]

        beat_payload = beat_detection["grid"] if beat_detection is not None else None
        beat_grid = None
        if beat_payload is not None:
            beat_grid = BeatGrid(
                bpm=float(beat_payload["bpm"]),
                beats_per_bar=beat_payload["beats_per_bar"],
                first_downbeat=float(beat_payload["first_downbeat"]),
                beats=np.asarray(beat_payload["beats"], dtype=float),
            ).with_onset_delay([event.start_time for event in note_starts])

        onset_delay_seconds = float(beat_grid.onset_delay or 0.0) if beat_grid is not None else 0.0
        midi_bytes = self.model.events_to_midi_bytes(iter(raw_events), beat_grid=beat_grid)
        torch.cuda.synchronize()

        inference_seconds = time.perf_counter() - started
        audio_seconds = float(preprocessing["audio_seconds"])
        events = [_serialize_event(event, onset_delay_seconds) for event in raw_events]
        detected_instruments = sorted({event.instrument for event in note_starts})

        mounted_artifact_path(paths["midi"]).write_bytes(midi_bytes)
        jsonl = "\n".join(json.dumps(event, sort_keys=True) for event in events)
        mounted_artifact_path(paths["events"]).write_text(f"{jsonl}\n", encoding="utf-8")

        result: dict[str, object] = {
            "job_id": job_id,
            "note_count": len(note_starts),
            "instruments": detected_instruments,
            "audio_seconds": audio_seconds,
            "preprocessing": preprocessing["metrics"],
            "timing": {
                "beat_grid_detected": beat_payload is not None,
                "beat_detection_seconds": (
                    float(beat_detection["seconds"]) if beat_detection is not None else 0.0
                ),
                "bpm": float(beat_payload["bpm"]) if beat_payload is not None else None,
                "beats_per_bar": (
                    beat_payload["beats_per_bar"] if beat_payload is not None else None
                ),
                "first_downbeat_seconds": (
                    float(beat_payload["first_downbeat"]) if beat_payload is not None else None
                ),
                "onset_delay_seconds": onset_delay_seconds,
                "fallback_reason": (
                    beat_detection["reason"] if beat_detection is not None else None
                ),
            },
            "model": {
                "container_id": self.container_id,
                "checkpoint_revision": MODEL_REVISION,
                "gpu_name": self.gpu_name,
                "load_seconds": self.model_load_seconds,
                "loaded_memory_bytes": self.model_memory_bytes,
            },
            "inference": {
                "seconds": inference_seconds,
                "real_time_factor": inference_seconds / audio_seconds,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            },
            "artifacts": {
                "events": paths["events"],
                "midi": paths["midi"],
                "metrics": paths["metrics"],
            },
        }
        mounted_artifact_path(paths["metrics"]).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_volume.commit()
        return result
