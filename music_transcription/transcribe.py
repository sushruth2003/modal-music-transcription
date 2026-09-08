"""L4-backed YourMT3+ MoE inference using the frozen evaluated checkpoint."""

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
    MODEL_MOUNT_PATH,
    YOURMT3_ARGS,
    YOURMT3_CHECKPOINT_BYTES,
    YOURMT3_CHECKPOINT_PATH,
    YOURMT3_CHECKPOINT_RELATIVE,
    YOURMT3_CHECKPOINT_SHA256,
    YOURMT3_READY_PATH,
    YOURMT3_REVISION,
)
from music_transcription.resources import app, artifact_volume, model_volume, yourmt3_image
from music_transcription.schemas import BeatGridDetection
from music_transcription.storage import job_paths, mounted_artifact_path


def corrected_event_time(seconds: float, onset_delay_seconds: float) -> float:
    """Apply MuScriptor's measured global onset correction to an event time."""

    return float(seconds) - onset_delay_seconds


@app.cls(
    image=yourmt3_image,
    gpu=GPU_TYPE,
    cpu=2,
    memory=12288,
    max_containers=GPU_MAX_CONTAINERS,
    min_containers=0,
    scaledown_window=GPU_SCALEDOWN_WINDOW_SECONDS,
    timeout=30 * 60,
    volumes={
        str(MODEL_MOUNT_PATH): model_volume.with_mount_options(read_only=True),
        str(ARTIFACT_MOUNT_PATH): artifact_volume,
    },
)
class YourMT3Transcriber:
    """One YourMT3+ MoE instance per temporary L4 container."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the pinned local checkpoint once for this container."""

        import os
        import sys
        from pathlib import Path

        import torch

        from music_transcription.models import _sha256_path

        if not YOURMT3_READY_PATH.is_file() or not YOURMT3_CHECKPOINT_PATH.is_file():
            raise RuntimeError("Run music_transcription.models::download_yourmt3 before deploying")
        ready = json.loads(YOURMT3_READY_PATH.read_text())
        if (
            ready.get("revision") != YOURMT3_REVISION
            or ready.get("checkpoint_sha256") != YOURMT3_CHECKPOINT_SHA256
            or YOURMT3_CHECKPOINT_PATH.stat().st_size != YOURMT3_CHECKPOINT_BYTES
            or _sha256_path(YOURMT3_CHECKPOINT_PATH) != YOURMT3_CHECKPOINT_SHA256
        ):
            raise RuntimeError("YourMT3 checkpoint identity/integrity check failed")
        os.chdir("/opt/yourmt3")
        sys.path[:0] = ["/opt/yourmt3", "/opt/yourmt3/amt/src"]
        checkpoint_link = Path(YOURMT3_CHECKPOINT_RELATIVE)
        checkpoint_link.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_link.symlink_to(YOURMT3_CHECKPOINT_PATH)
        from model_helper import load_model_checkpoint

        torch.set_num_threads(2)
        torch.manual_seed(20260905)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.model = load_model_checkpoint(YOURMT3_ARGS, device="cpu").to("cuda").eval()
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

        import torch

        from music_transcription.note_export import midi_bytes, normalize_notes, serialize_notes

        paths = job_paths(job_id, source_suffix)
        artifact_volume.reload()
        wav_path = mounted_artifact_path(paths["normalized"])
        preprocessing = json.loads(
            mounted_artifact_path(paths["preprocessing"]).read_text(encoding="utf-8")
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw_notes, decoding_errors = infer_notes(self.model, wav_path)
        audio_seconds = float(preprocessing["audio_seconds"])
        notes, export_metrics = normalize_notes(raw_notes, audio_seconds, instruments)
        beat_payload = beat_detection["grid"] if beat_detection is not None else None
        midi = midi_bytes(notes, beat_payload)
        events = serialize_notes(notes)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        detected_instruments = sorted({note.instrument for note in notes})

        mounted_artifact_path(paths["midi"]).write_bytes(midi)
        jsonl = "\n".join(json.dumps(event, sort_keys=True) for event in events)
        mounted_artifact_path(paths["events"]).write_text(f"{jsonl}\n", encoding="utf-8")

        result: dict[str, object] = {
            "job_id": job_id,
            "note_count": len(notes),
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
                "onset_delay_seconds": 0.0,
                "bar_offset_seconds": 0.0,
                "note_timing": "performance; no quantization",
                "fallback_reason": (
                    beat_detection["reason"] if beat_detection is not None else None
                ),
            },
            "export": {
                **export_metrics,
                "decoding_errors": decoding_errors,
                "instrument_selection": "filter predicted tracks",
            },
            "model": {
                "name": "YourMT3+ MoE",
                "precision": "float32",
                "checkpoint_sha256": YOURMT3_CHECKPOINT_SHA256,
                "container_id": self.container_id,
                "checkpoint_revision": YOURMT3_REVISION,
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


def infer_notes(model, wav_path):
    """Author's evaluated batched decode and cross-segment tie merging."""
    from collections import Counter

    import torch
    import torchaudio
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    audio, sr = torchaudio.load(str(wav_path))
    audio = audio.mean(dim=0, keepdim=True)
    sample_rate = model.audio_cfg["sample_rate"]
    audio = torchaudio.functional.resample(audio, sr, sample_rate)
    frames = model.audio_cfg["input_frames"]
    segments = slice_padded_array(audio, frames, frames)
    segments = torch.from_numpy(segments.astype("float32")).to("cuda").unsqueeze(1)
    tokens, _ = model.inference_file(bsz=8, audio_segments=segments)
    start_seconds = [frames * i / sample_rate for i in range(len(segments))]
    notes_by_channel = []
    errors = Counter()
    for channel in range(model.task_manager.num_decoding_channels):
        zipped, _, token_errors = model.task_manager.detokenize_list_batches(
            [arr[:, channel, :] for arr in tokens],
            start_seconds,
            return_events=True,
        )
        notes, note_errors = merge_zipped_note_events_and_ties_to_notes(zipped)
        notes_by_channel.append(notes)
        errors.update(token_errors)
        errors.update(note_errors)
    return mix_notes(notes_by_channel), dict(errors)
