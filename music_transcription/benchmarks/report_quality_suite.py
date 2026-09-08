"""Offline suite scoring, paired uncertainty estimates and an interactive HTML report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from music_transcription.benchmarks import quality, quality_refinement
from music_transcription.benchmarks.quality import Note, evaluate, read_midi, write_midi
from music_transcription.benchmarks.quality_refinement import refine_onsets

METRICS = [
    "onset",
    "onset_offset",
    "instrument_onset",
    "instrument_onset_offset",
    "frame",
    "drum_onset",
]
LABELS = {
    "baseline": "Automatic · greedy",
    "auto_beam2": "Automatic · beam 2",
    "auto_beam4": "Automatic · beam 4",
    "conditioned": "Known instruments · greedy",
    "beam2": "Known instruments · beam 2",
    "beam4": "Known instruments · beam 4",
    "refined": "Automatic · audio refinement",
}
VARIANTS = list(LABELS)


def product_notes(path: Path, expected_offset: float) -> list[Note]:
    """Undo only documented bar padding, retaining actual acoustic timing correction."""
    midi = mido.MidiFile(path)
    offset = 0.0
    for track in midi.tracks:
        for msg in track:
            if msg.type == "marker" and msg.text.startswith("muscriptor:bar_offset="):
                offset = float(msg.text.split("=", 1)[1])
    if abs(offset - expected_offset) > 1e-7:
        raise ValueError("Recorded bar padding disagrees with MIDI")
    return [
        replace(n, start=max(0.0, n.start - offset), end=n.end - offset)
        for n in read_midi(path)
        if n.end > offset
    ]


def bootstrap(rows: list[dict], baseline: dict, metric: str, *, samples: int = 5000) -> dict:
    """Paired recording bootstrap, stratified by source; no notes treated as iid."""
    if metric == "drum_onset":
        rows = [r for r in rows if r["metrics"][metric]["tp"] + r["metrics"][metric]["fn"] > 0]
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "delta": None,
            "low": None,
            "high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    values = np.array([r["metrics"][metric]["f1"] for r in rows])
    base = np.array([baseline[r["example"]]["metrics"][metric]["f1"] for r in rows])
    deltas = values - base
    groups = [
        [i for i, r in enumerate(rows) if r["dataset"] == d]
        for d in sorted({r["dataset"] for r in rows})
    ]
    rng = np.random.default_rng(20260905)
    indices = np.concatenate(
        [rng.choice(g, (samples, len(g)), replace=True) for g in groups], axis=1
    )
    distribution = deltas[indices].mean(axis=1)
    mean_distribution = values[indices].mean(axis=1)
    tp = sum(r["metrics"][metric]["tp"] for r in rows)
    fp = sum(r["metrics"][metric]["fp"] for r in rows)
    fn = sum(r["metrics"][metric]["fn"] for r in rows)
    return {
        "n": len(rows),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_low": float(np.quantile(mean_distribution, 0.025)),
        "mean_high": float(np.quantile(mean_distribution, 0.975)),
        "delta": float(deltas.mean()),
        "low": float(np.quantile(distribution, 0.025)),
        "high": float(np.quantile(distribution, 0.975)),
        "wins": int((deltas > 1e-8).sum()),
        "ties": int((abs(deltas) <= 1e-8).sum()),
        "losses": int((deltas < -1e-8).sum()),
        "regressions_over_5pp": int((deltas < -0.05).sum()),
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
    }


def aggregate(records: list[dict]) -> dict:
    output = {}
    for mode in ("product", "raw"):
        output[mode] = {}
        for dataset in ["All", "URMP", "Slakh", "GuitarSet", "MAESTRO"]:
            selected = [
                r
                for r in records
                if r["mode"] == mode and (dataset == "All" or r["dataset"] == dataset)
            ]
            baseline = {r["example"]: r for r in selected if r["variant"] == "baseline"}
            output[mode][dataset] = {
                v: {
                    metric: bootstrap([r for r in selected if r["variant"] == v], baseline, metric)
                    for metric in METRICS
                }
                for v in VARIANTS
            }
    return output


def packed(notes: list[Note]) -> list[list]:
    return [[n.pitch, round(n.start, 5), round(n.end, 5), n.program, n.velocity] for n in notes]


def report(root: Path) -> Path:
    manifest = json.loads((root / "manifest.json").read_text())
    run = json.loads((root / "run.json").read_text())
    all_examples = manifest["examples"]
    examples = [e for e in all_examples if e["id"] in run["clips"]]
    records = []
    audition = {}
    diagnostics = []
    for e in examples:
        folder = root / e["id"]
        ref = read_midi(folder / "reference.mid")
        if (
            hashlib.sha256((folder / "reference.mid").read_bytes()).hexdigest()
            != e["reference_sha256"]
        ):
            raise ValueError("Reference checksum mismatch")
        if hashlib.sha256((folder / "source.wav").read_bytes()).hexdigest() != e["audio_sha256"]:
            raise ValueError("Audio checksum mismatch")
        audio, sr = sf.read(folder / "source.wav", dtype="float32")
        clip = run["clips"][e["id"]]
        audition[e["id"]] = {"reference": packed(ref)}
        baseline_notes = {}
        baseline_stats = {}
        for stats in clip["rows"]:
            v = stats["variant"]
            raw = read_midi(folder / f"{v}.raw.mid")
            prod = product_notes(folder / f"{v}.product.mid", stats["bar_offset_seconds"])
            write_midi(prod, folder / f"{v}.aligned.mid")
            for mode, notes in [("raw", raw), ("product", prod)]:
                metrics = evaluate(
                    ref,
                    notes,
                    start=e["score_start"],
                    end=e["score_end"],
                    audio_end=e["audio_seconds"],
                )
                seconds = stats["seconds"] + (
                    clip["beat_seconds"] + stats["export_seconds"] if mode == "product" else 0.0
                )
                records.append(
                    {
                        "example": e["id"],
                        "dataset": e["dataset"],
                        "variant": v,
                        "mode": mode,
                        "seconds": seconds,
                        "metrics": metrics,
                    }
                )
                audition[e["id"]][mode + "_" + v] = packed(notes)
                if v == "baseline":
                    baseline_notes[mode] = notes
                    baseline_stats[mode] = seconds
            diagnostics.append(
                {
                    "example": e["id"],
                    "variant": v,
                    "onset_delay_ms": 1000 * stats["onset_delay_seconds"],
                    "bar_padding_seconds": stats["bar_offset_seconds"],
                    "beat_grid_detected": clip["beat_grid"] is not None,
                }
            )
        for mode in ("raw", "product"):
            started = time.perf_counter()
            notes, stats = refine_onsets(audio, sr, baseline_notes[mode])
            seconds = time.perf_counter() - started
            write_midi(notes, folder / f"refined.{mode}.mid")
            notes = read_midi(folder / f"refined.{mode}.mid")
            metrics = evaluate(
                ref, notes, start=e["score_start"], end=e["score_end"], audio_end=e["audio_seconds"]
            )
            records.append(
                {
                    "example": e["id"],
                    "dataset": e["dataset"],
                    "variant": "refined",
                    "mode": mode,
                    "seconds": baseline_stats[mode] + seconds,
                    "refinement_seconds": seconds,
                    "refinement": stats,
                    "metrics": metrics,
                }
            )
            audition[e["id"]][mode + "_refined"] = packed(notes)
        print("Scored " + e["id"], flush=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "completed": run["completed"],
        "expected_examples": len(all_examples),
        "examples": examples,
        "sources": manifest["sources"],
        "program_names": {
            **{str(v[0]): k.replace("_", " ") for k, v in quality.PROGRAM_GROUPS.items()},
            "128": "drums",
        },
        "selection": manifest["selection"],
        "selection_sha256": manifest["selection_sha256"],
        "mirror": manifest["mirror"],
        "revision": manifest["revision"],
        "plan": run["plan"],
        "records": records,
        "summary": aggregate(records),
        "audition": audition,
        "diagnostics": diagnostics,
        "run": {k: v for k, v in run.items() if k != "clips"},
        "beat_grids": {
            e["id"]: {k: v for k, v in run["clips"][e["id"]].items() if k != "rows"}
            for e in examples
        },
        "interpretation": json.loads((root / "interpretation.json").read_text())
        if (root / "interpretation.json").exists()
        else None,
        "interruption": json.loads((root / "interruption.json").read_text())
        if (root / "interruption.json").exists()
        else None,
        "source_hashes": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in [("scoring", quality), ("refinement", quality_refinement)]
        },
    }
    (root / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (root / "per-recording.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["example", "dataset", "variant", "mode", "seconds"]
            + METRICS
            + ["precision", "recall", "extra_notes", "missed_notes"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    **{k: r[k] for k in ["example", "dataset", "variant", "mode", "seconds"]},
                    **{m: r["metrics"][m]["f1"] for m in METRICS},
                    "precision": r["metrics"]["onset"]["precision"],
                    "recall": r["metrics"]["onset"]["recall"],
                    "extra_notes": r["metrics"]["onset"]["fp"],
                    "missed_notes": r["metrics"]["onset"]["fn"],
                }
            )
    template = Path(__file__).with_name("quality_report_template.html").read_text()
    embedded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__REPORT_DATA__", embedded)
    destination = root / "index.html"
    destination.write_text(html)
    print(destination.resolve())
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
