"""Isolated, resumable 32-recording evaluation. Three temporary L4 workers maximum."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from music_transcription.config import (
    BEAT_CHECKPOINT_PATH,
    BEAT_PACKAGE,
    L4_PRICE_PER_SECOND_USD,
    MODEL_CHECKPOINT_PATH,
    MODEL_MOUNT_PATH,
    MODEL_PACKAGE,
    MODEL_READY_PATH,
    MODEL_REVISION,
    MODEL_VOLUME_NAME,
)

app = modal.App("music-transcription-quality-suite")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(MODEL_PACKAGE, BEAT_PACKAGE)
    .env({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source("music_transcription")
)
volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
VARIANTS = {
    "baseline": {"beam_size": 1, "conditioned": False},
    "auto_beam2": {"beam_size": 2, "conditioned": False},
    "auto_beam4": {"beam_size": 4, "conditioned": False},
    "conditioned": {"beam_size": 1, "conditioned": True},
    "beam2": {"beam_size": 2, "conditioned": True},
    "beam4": {"beam_size": 4, "conditioned": True},
}


@app.cls(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    max_containers=3,
    min_containers=0,
    scaledown_window=30,
    timeout=1200,
    volumes={str(MODEL_MOUNT_PATH): volume.with_mount_options(read_only=True)},
)
class SuiteTranscriber:
    @modal.enter()
    def load(self):
        import torch
        from beat_this.inference import Audio2Beats
        from muscriptor import TranscriptionModel

        torch.set_num_threads(2)
        if json.loads(MODEL_READY_PATH.read_text())["revision"] != MODEL_REVISION:
            raise RuntimeError("Checkpoint revision mismatch")
        t = time.perf_counter()
        self.model = TranscriptionModel.load_model(
            MODEL_CHECKPOINT_PATH, device="cuda", dtype="float16"
        )
        self.detector = Audio2Beats(
            checkpoint_path=str(BEAT_CHECKPOINT_PATH), device="cpu", dbn=False
        )
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t

    @modal.method()
    def run_clip(self, audio_bytes: bytes, instruments: list[str], rotation: int = 0) -> dict:
        import warnings

        import mido
        import numpy as np
        import soundfile as sf
        import torch
        from muscriptor.events import NoteEndEvent, NoteStartEvent
        from muscriptor.utils.beats import (
            MAX_TEMPO_RESIDUAL,
            MIN_BEATS,
            BeatDetectionError,
            BeatGrid,
            fit_tempo,
            infer_beats_per_bar,
            read_bar_offset,
        )

        if len(audio_bytes) > 2_000_000:
            raise ValueError("Input too large")
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 30 * sr:
            raise ValueError("Expected <=30s mono 16kHz")
        waveform = torch.from_numpy(audio).unsqueeze(0)
        # Same detector and acceptance rules as the deployed CPU beat stage.
        t = time.perf_counter()
        grid = None
        reason = None
        beats, downbeats = self.detector(audio, sr)
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        try:
            if len(beats) < MIN_BEATS:
                raise BeatDetectionError("Insufficient beats")
            bpm, residual = fit_tempo(beats)
            if residual > MAX_TEMPO_RESIDUAL * 60 / bpm:
                raise BeatDetectionError("No stable constant tempo")
            grid = BeatGrid(
                bpm=bpm,
                beats_per_bar=infer_beats_per_bar(beats, downbeats),
                first_downbeat=float(downbeats[0]) if len(downbeats) else float(beats[0]),
                beats=beats,
            )
        except BeatDetectionError as err:
            reason = str(err)
        beat_seconds = time.perf_counter() - t
        rows = []
        names = list(VARIANTS)
        names = names[rotation % 6 :] + names[: rotation % 6]
        for variant in names:
            cfg = VARIANTS[variant]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                events = list(
                    self.model.transcribe(
                        (waveform, sr),
                        instruments=instruments if cfg["conditioned"] else None,
                        beam_size=cfg["beam_size"],
                        use_sampling=False,
                        cfg_coef=1.0,
                        prelude_forcing=True,
                    )
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            active = {}
            notes = []
            for ev in events:
                if isinstance(ev, NoteStartEvent):
                    active[ev.index] = ev
                elif isinstance(ev, NoteEndEvent):
                    onset = active.pop(ev.start_event_index)
                    start = max(0.0, float(onset.start_time))
                    end = min(len(audio) / sr, max(float(ev.end_time), start + 0.01))
                    if end > start:
                        notes.append(
                            {
                                "pitch": onset.pitch,
                                "start": start,
                                "end": end,
                                "instrument": onset.instrument,
                            }
                        )
            if active:
                raise RuntimeError("Unpaired starts")
            t = time.perf_counter()
            adjusted = (
                grid.with_onset_delay(
                    [ev.start_time for ev in events if isinstance(ev, NoteStartEvent)]
                )
                if grid is not None
                else None
            )
            product = self.model.events_to_midi_bytes(iter(events), beat_grid=adjusted)
            bar_offset = read_bar_offset(mido.MidiFile(file=io.BytesIO(product)))
            export_seconds = time.perf_counter() - t
            rows.append(
                {
                    "variant": variant,
                    "config": cfg,
                    "notes": notes,
                    "product_midi": product,
                    "seconds": seconds,
                    "export_seconds": export_seconds,
                    "bar_offset_seconds": bar_offset,
                    "onset_delay_seconds": float(adjusted.onset_delay or 0) if adjusted else 0.0,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                    "warnings": [str(w.message) for w in caught],
                }
            )
            print(f"{variant}: {len(notes)} notes, {seconds:.1f}s", flush=True)
        return {
            "rows": rows,
            "beat_seconds": beat_seconds,
            "beat_grid": {
                "bpm": float(grid.bpm),
                "beats_per_bar": grid.beats_per_bar,
                "first_downbeat": grid.first_downbeat,
                "beats": beats.tolist(),
            }
            if grid
            else None,
            "beat_fallback": reason,
            "load_seconds": self.load_seconds,
            "gpu": torch.cuda.get_device_name(0),
            "checkpoint": MODEL_REVISION,
        }


@app.local_entrypoint()
def main(
    execute: bool = False,
    output: str = "outputs/quality-suite-20260905",
    manifest: str = "data/quality-suite/manifest.json",
    max_seconds: int = 9000,
    limit: int = 32,
    resume: bool = False,
):
    from music_transcription.benchmarks.quality import (
        Note,
        program_for_name,
        write_midi,
        write_notes,
    )

    source = Path(manifest)
    data = json.loads(source.read_text())
    examples = data["examples"][:limit]
    if not 1 <= len(examples) <= 40 or not 1 <= max_seconds <= 9000:
        raise ValueError("Suite bounds exceeded")
    plan = {
        "examples": len(examples),
        "model_inferences": len(examples) * 6,
        "variants": VARIANTS,
        "checkpoint": MODEL_REVISION,
        "max_inference_seconds": max_seconds,
        "inference_budget_usd": max_seconds * L4_PRICE_PER_SECOND_USD,
        "max_workers": 3,
        "budget_note": "Checked between completed clips; up to three in-flight clips may finish beyond budget. Excludes startup, CPU, idle and image builds.",
        "timing": "All six variants evaluated in raw performance time and after actual app beat/MIDI export. Only the export-recorded bar padding is removed for scoring; no reference alignment.",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not execute:
        return
    root = Path(output)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Resume needs an existing run")
    payload_path = root / "run.json"
    payload = (
        json.loads(payload_path.read_text())
        if resume
        else {
            "plan": plan,
            "created_at": datetime.now(UTC).isoformat(),
            "clips": {},
            "failures": {},
            "completed": False,
        }
    )
    if resume and json.loads((root / "manifest.json").read_text()) != data:
        raise ValueError("Cannot resume a different dataset")
    (root / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")

    def save():
        payload["actual_inference_seconds"] = sum(
            r["seconds"] for c in payload["clips"].values() for r in c["rows"]
        )
        payload["estimated_inference_cost_usd"] = (
            payload["actual_inference_seconds"] * L4_PRICE_PER_SECOND_USD
        )
        tmp = root / "run.partial.json"
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(payload_path)

    worker = SuiteTranscriber()
    save()
    remaining = [(i, e) for i, e in enumerate(examples) if e["id"] not in payload["clips"]]

    def run(item):
        i, e = item
        audio = source.parent / e["audio"]
        reference = source.parent / e["reference"]
        for path, key in [(audio, "audio_sha256"), (reference, "reference_sha256")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != e[key]:
                raise ValueError("Manifest checksum mismatch")
        response = worker.run_clip.remote(audio.read_bytes(), e["instruments"], i)
        folder = root / e["id"]
        folder.mkdir(exist_ok=True)
        (folder / "source.wav").write_bytes(audio.read_bytes())
        (folder / "reference.mid").write_bytes(reference.read_bytes())
        for r in response["rows"]:
            variant = r["variant"]
            (folder / f"{variant}.product.mid").write_bytes(r.pop("product_midi"))
            notes = [
                Note(n["pitch"], n["start"], n["end"], program_for_name(n["instrument"]))
                for n in r.pop("notes")
            ]
            write_notes(notes, folder / f"{variant}.notes.json")
            write_midi(notes, folder / f"{variant}.raw.mid")
        return e["id"], response

    # At most three clip calls in flight. Persist each completed clip before scheduling another.
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        pending = {}
        iterator = iter(remaining)
        for item in list(remaining)[:3]:
            pending[pool.submit(run, item)] = item
        for _ in range(min(3, len(remaining))):
            next(iterator)
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                item = pending.pop(future)
                try:
                    key, response = future.result()
                    payload["clips"][key] = response
                    payload["failures"].pop(key, None)
                    print(f"Completed {len(payload['clips'])}/{len(examples)}: {key}", flush=True)
                except Exception as err:  # noqa: BLE001 - persist remote failures for explicit resume
                    payload["failures"][item[1]["id"]] = f"{type(err).__name__}: {err}"
                    print(f"Failed {item[1]['id']}: {err}", flush=True)
                save()
                if payload["actual_inference_seconds"] < max_seconds:
                    following = next(iterator, None)
                    if following:
                        pending[pool.submit(run, following)] = following
    payload["completed"] = all(e["id"] in payload["clips"] for e in examples)
    save()
    print(f"Saved {root.resolve()}", flush=True)
    if not payload["completed"]:
        raise RuntimeError("Incomplete suite; inspect run.json and resume explicitly")


if __name__ == "__main__":
    main()
