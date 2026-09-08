"""Bounded, opt-in MuScriptor quality experiments on a separate ephemeral Modal app.

Preview: python -m music_transcription.benchmarks.quality_modal
Execute: modal run -m music_transcription.benchmarks.quality_modal --execute
No public endpoint, job state, quota, or artifact Volume is touched. Reference MIDI
stays local: the GPU receives only audio and optional instrument identities.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from music_transcription.config import (
    L4_PRICE_PER_SECOND_USD,
    MODEL_CHECKPOINT_PATH,
    MODEL_MOUNT_PATH,
    MODEL_PACKAGE,
    MODEL_READY_PATH,
    MODEL_REVISION,
    MODEL_VOLUME_NAME,
)

app = modal.App("music-transcription-quality-eval")
volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(MODEL_PACKAGE)
    .env({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source("music_transcription")
)
VARIANTS = {
    "baseline": {"conditioned": False, "beam_size": 1, "shift_seconds": 0.0},
    "conditioned": {"conditioned": True, "beam_size": 1, "shift_seconds": 0.0},
    "beam2": {"conditioned": True, "beam_size": 2, "shift_seconds": 0.0},
    "beam4": {"conditioned": True, "beam_size": 4, "shift_seconds": 0.0},
    "shifted": {"conditioned": True, "beam_size": 1, "shift_seconds": 2.5},
}


@app.cls(
    image=image,
    gpu="L4",
    max_containers=1,
    min_containers=0,
    scaledown_window=30,
    timeout=300,
    volumes={str(MODEL_MOUNT_PATH): volume.with_mount_options(read_only=True)},
)
class QualityTranscriber:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from muscriptor import TranscriptionModel

        ready = json.loads(MODEL_READY_PATH.read_text())
        if ready.get("revision") != MODEL_REVISION:
            raise RuntimeError("Pinned model checkpoint is not ready")
        started = time.perf_counter()
        self.model = TranscriptionModel.load_model(
            MODEL_CHECKPOINT_PATH, device="cuda", dtype="float16"
        )
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.gpu_name = torch.cuda.get_device_name(0)

    @modal.method()
    def transcribe(
        self,
        wav_bytes: bytes,
        instruments: list[str] | None = None,
        beam_size: int = 1,
        shift_seconds: float = 0.0,
    ) -> dict:
        import warnings

        import numpy as np
        import soundfile as sf
        import torch
        from muscriptor.events import NoteEndEvent, NoteStartEvent

        if len(wav_bytes) > 4_000_000 or beam_size not in (1, 2, 4):
            raise ValueError("Experiment exceeds bounded input/beam settings")
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if sr != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 60 * sr:
            raise ValueError("Expected at most 60 seconds of mono 16 kHz audio")
        if shift_seconds not in (0.0, 2.5):
            raise ValueError("Only zero or half-chunk shift supported")
        duration = len(audio) / sr
        # Prefix silence changes chunk placement without dropping any source audio.
        padded = np.pad(audio, (round(shift_seconds * sr), 0))
        waveform = torch.from_numpy(padded).unsqueeze(0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            events = list(
                self.model.transcribe(
                    (waveform, sr),
                    instruments=instruments,
                    use_sampling=False,
                    beam_size=beam_size,
                    cfg_coef=1.0,
                    prelude_forcing=True,
                )
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        active = {}
        notes = []
        for event in events:
            if isinstance(event, NoteStartEvent):
                active[event.index] = event
            elif isinstance(event, NoteEndEvent):
                onset = active.pop(event.start_event_index)
                start = max(0.0, float(onset.start_time) - shift_seconds)
                end = min(duration, float(event.end_time) - shift_seconds)
                if start < end:
                    notes.append(
                        {
                            "pitch": onset.pitch,
                            "start": start,
                            "end": end,
                            "instrument": onset.instrument,
                        }
                    )
        if active:
            raise RuntimeError("Unpaired note starts in model output")
        return {
            "notes": notes,
            "seconds": seconds,
            "model_load_seconds": self.load_seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu_name": self.gpu_name,
            "checkpoint_revision": MODEL_REVISION,
            "warnings": [str(w.message) for w in caught],
        }


def summarize(records: list[dict]) -> dict:
    import numpy as np

    summary = {}
    for variant in dict.fromkeys(r["variant"] for r in records):
        rows = [r for r in records if r["variant"] == variant and r.get("metrics")]
        if not rows:
            continue
        entry = {
            "examples": len(rows),
            "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
            "total_seconds": sum(r["seconds"] for r in rows),
        }
        for metric in [
            "onset",
            "onset_offset",
            "instrument_onset",
            "instrument_onset_offset",
            "frame",
        ]:
            ms = [r["metrics"][metric] for r in rows]
            tp, fp, fn = (sum(m[k] for m in ms) for k in ("tp", "fp", "fn"))
            entry[metric] = {
                "macro_f1": float(np.mean([m["f1"] for m in ms])),
                "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        summary[variant] = entry
    return summary


@app.local_entrypoint()
def main(
    manifest: str = "data/quality-urmp/manifest.json",
    output: str = "",
    variants: str = "baseline,conditioned,beam2,beam4,shifted",
    execute: bool = False,
    max_seconds: int = 900,
) -> None:
    selected = variants.split(",")
    if any(v not in VARIANTS for v in selected) or len(selected) != len(set(selected)):
        raise ValueError(f"Select unique variants from {list(VARIANTS)}")
    manifest_path = Path(manifest)
    data = json.loads(manifest_path.read_text())
    examples = data["examples"]
    if not 1 <= len(examples) <= 5 or not 1 <= max_seconds <= 1800:
        raise ValueError("Use 1–5 examples and a run budget of at most 1800 seconds")
    plan = {
        "examples": [e["id"] for e in examples],
        "variants": selected,
        "gpu_calls": len(examples) * len(selected),
        "checkpoint": MODEL_REVISION,
        "compute_budget_seconds": max_seconds,
        "inference_cost_at_budget_usd": max_seconds * L4_PRICE_PER_SECOND_USD,
        "cost_note": "Rate from project config, not billing. Excludes model load, idle, CPU, and image build. Budget checked between calls; one call can run for up to 300 seconds.",
        "evaluation": "Raw performance seconds; no beat correction, grid snapping, or ground-truth alignment. Pitched onset F1 at 50 ms/50 cents; offset tolerance max(50 ms, 20% reference duration). Drums separate.",
        "conditioning": "Oracle instrument identities from dataset metadata; no note/timing labels sent to GPU.",
        "consensus": "Conditioned + shifted intersection, one-to-one onset matches within 80 ms; averaged timing. Cost includes both passes.",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not execute:
        print("Preview only. Add --execute to run the GPU experiment.")
        return
    from music_transcription.benchmarks.quality import (
        Note,
        consensus,
        evaluate,
        program_for_name,
        read_midi,
        write_midi,
        write_notes,
    )

    root = (
        Path(output)
        if output
        else Path("outputs") / datetime.now(UTC).strftime("quality-%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    report = {
        "plan": plan,
        "created_at": datetime.now(UTC).isoformat(),
        "records": [],
        "completed": False,
        "status": "running",
    }
    worker = QualityTranscriber()
    elapsed = 0.0

    def save() -> None:
        report["summary"] = summarize(report["records"])
        report["actual_inference_seconds"] = elapsed
        report["estimated_inference_cost_usd"] = elapsed * L4_PRICE_PER_SECOND_USD
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for example in examples:
            audio_path = manifest_path.parent / example["audio"]
            reference_path = manifest_path.parent / example["reference"]
            audio = audio_path.read_bytes()
            if hashlib.sha256(audio).hexdigest() != example["audio_sha256"]:
                raise ValueError("Audio manifest checksum mismatch")
            if (
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                != example["reference_sha256"]
            ):
                raise ValueError("Reference manifest checksum mismatch")
            ref = read_midi(reference_path)
            folder = root / example["id"]
            folder.mkdir()
            (folder / "source.wav").write_bytes(audio)
            (folder / "reference.mid").write_bytes(reference_path.read_bytes())
            outputs = {}
            timings = {}

            def record(
                variant: str,
                notes: list[Note],
                stats: dict,
                *,
                folder: Path = folder,
                ref: list[Note] = ref,
                example: dict = example,
            ) -> None:
                write_notes(notes, folder / f"{variant}.notes.json")
                write_midi(notes, folder / f"{variant}.mid")
                # Score the actual exported MIDI, exercising serialization as well.
                metric = evaluate(
                    ref,
                    read_midi(folder / f"{variant}.mid"),
                    start=example["score_start"],
                    end=example["score_end"],
                    audio_end=example["audio_seconds"],
                )
                report["records"].append(
                    {"example": example["id"], "variant": variant, **stats, "metrics": metric}
                )
                save()
                print(
                    f"{example['id']} {variant}: onset F1={metric['onset']['f1']:.3f}, "
                    f"instrument F1={metric['instrument_onset']['f1']:.3f}, "
                    f"offset F1={metric['onset_offset']['f1']:.3f}, "
                    f"{stats['seconds']:.2f}s",
                    flush=True,
                )

            for variant in selected:
                if elapsed >= max_seconds:
                    raise TimeoutError("Inference budget reached; partial results saved")
                config = VARIANTS[variant]
                response = worker.transcribe.remote(
                    audio,
                    instruments=example["instruments"] if config["conditioned"] else None,
                    beam_size=config["beam_size"],
                    shift_seconds=config["shift_seconds"],
                )
                elapsed += response["seconds"]
                notes = [
                    Note(n["pitch"], n["start"], n["end"], program_for_name(n["instrument"]))
                    for n in response.pop("notes")
                ]
                outputs[variant] = notes
                timings[variant] = response["seconds"]
                record(variant, notes, {**response, "config": config})
            if "conditioned" in outputs and "shifted" in outputs:
                record(
                    "consensus",
                    consensus(outputs["conditioned"], outputs["shifted"]),
                    {"seconds": timings["conditioned"] + timings["shifted"], "derived": True},
                )
        report["completed"] = True
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        print(f"Saved results and MIDI outputs: {root.resolve()}", flush=True)


if __name__ == "__main__":
    # Plain Python is preview-only; Modal is needed only for the opt-in remote run.
    main()
