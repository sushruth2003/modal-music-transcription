"""Rescore saved note predictions and produce a readable report without GPU calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from music_transcription.benchmarks import quality
from music_transcription.benchmarks.quality import Note, evaluate, read_midi, write_midi
from music_transcription.benchmarks.quality_modal import summarize


def report(root: Path) -> Path:
    payload = json.loads((root / "results.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    examples = {e["id"]: e for e in manifest["examples"]}
    for row in payload["records"]:
        example = examples[row["example"]]
        folder = root / row["example"]
        reference = folder / "reference.mid"
        if hashlib.sha256(reference.read_bytes()).hexdigest() != example["reference_sha256"]:
            raise ValueError("Reference MIDI differs from the run manifest")
        notes = [
            Note(**n) for n in json.loads((folder / f"{row['variant']}.notes.json").read_text())
        ]
        prediction = folder / f"{row['variant']}.mid"
        write_midi(notes, prediction)
        row["metrics"] = evaluate(
            read_midi(reference),
            read_midi(prediction),
            start=example["score_start"],
            end=example["score_end"],
            audio_end=example["audio_seconds"],
        )
    payload["summary"] = summarize(payload["records"])
    payload["scoring_source_sha256"] = hashlib.sha256(
        Path(quality.__file__).read_bytes()
    ).hexdigest()
    payload["rescored_at"] = datetime.now(UTC).isoformat()
    (root / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    summaries = payload["summary"]
    lines = [
        "# MuScriptor quality experiment",
        "",
        (
            "Three preselected real URMP ensemble excerpts, 30 seconds each; onsets scored "
            "from 2.5 to 27.5 seconds. This is a small exploratory comparison, not a "
            "held-out claim about all genres or production mixtures."
        ),
        "",
        "## Aggregate results",
        "",
        (
            "F1 scores below are percentages averaged equally over the three recordings. "
            "Note F1 requires correct pitch and onset within 50 ms. Duration F1 also requires "
            "offset within max(50 ms, 20% of reference duration). Instrument-aware scores "
            "require the correct MuScriptor instrument group."
        ),
        "",
        "| Variant | Note F1 | Note+duration F1 | Instrument+note F1 | Instrument+duration F1 | Frame F1 | Mean inference seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        values = [
            s[k]["macro_f1"] * 100
            for k in [
                "onset",
                "onset_offset",
                "instrument_onset",
                "instrument_onset_offset",
                "frame",
            ]
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{x:.2f}" for x in values)
            + f" | {s['mean_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Per-recording results",
        "",
        "| Recording | Variant | Precision | Recall | Note F1 | Instrument+note F1 | Extra notes | Missed notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    with (root / "per-recording.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recording",
                "variant",
                "precision",
                "recall",
                "note_f1",
                "instrument_note_f1",
                "extra_notes",
                "missed_notes",
                "seconds",
            ]
        )
        for row in payload["records"]:
            m = row["metrics"]
            n = m["onset"]
            writer.writerow(
                [
                    row["example"],
                    row["variant"],
                    n["precision"],
                    n["recall"],
                    n["f1"],
                    m["instrument_onset"]["f1"],
                    n["fp"],
                    n["fn"],
                    row["seconds"],
                ]
            )
            lines.append(
                f"| {row['example']} | {row['variant']} | {n['precision'] * 100:.2f} | "
                f"{n['recall'] * 100:.2f} | {n['f1'] * 100:.2f} | "
                f"{m['instrument_onset']['f1'] * 100:.2f} | {n['fp']} | {n['fn']} |"
            )
    lines += [
        "",
        "## Onset timing diagnostic",
        "",
        (
            "The official comparison above stays at 50 ms. This tolerance sweep shows how "
            "much performance changes when timing requirements are relaxed; it does not "
            "correct or align predictions. High scores at 100/200 ms do not mean precise MIDI."
        ),
        "",
        "| Variant | F1 at 25 ms | F1 at 50 ms | F1 at 100 ms | F1 at 200 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in summaries:
        rows = [r for r in payload["records"] if r["variant"] == name]
        values = [
            sum(r["metrics"]["onset_tolerance_curve"][str(ms)]["f1"] for r in rows)
            / len(rows)
            * 100
            for ms in (25, 50, 100, 200)
        ]
        lines.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in values) + " |")
    lines += [
        "",
        "## Method and limitations",
        "",
        (
            "- Baseline matches the deployed model's decoding settings: Large, fp16, greedy, "
            "CFG 1, prelude forcing. Beat-grid onset correction and MIDI bar shifting are "
            "excluded from every variant to isolate note decoding; this is not the full "
            "web application's end-to-end evaluation."
        ),
        (
            "- Conditioned variants use the correct instrument list from dataset metadata. "
            "This measures user-supplied/oracle instrumentation, not automatic instrument recognition."
        ),
        (
            "- Beam2/beam4 add beam search to conditioning. Shifted adds 2.5 seconds of silence "
            "before decoding and subtracts it afterward, retaining the complete source excerpt."
        ),
        (
            "- Consensus retains one-to-one pitch/instrument/onset agreements within 80 ms "
            "between conditioned and shifted passes and averages their timing. It intentionally "
            "trades recall for precision. Its cost is the sum of both passes, although those "
            "passes were reused during this experiment."
        ),
        (
            "- Reference MIDI is constructed from URMP's performance-aligned Notes_* annotations. "
            "Downloaded Sco_* MIDI is a score with a different clock and is not used for timing "
            "evaluation. Predictions are never shifted or warped using reference labels."
        ),
        (
            "- Overlapping same-pitch notes are preserved on separate MIDI tracks. We do not "
            "remove difficult unisons from the reference; these can exceed MuScriptor's representation."
        ),
        (
            "- Offset labels in real acoustic recordings have ambiguity; velocity is not evaluated. "
            "No drums occur in this set. Dense produced pop/rock, vocals, effects, and other genres "
            "remain untested. Training-set overlap with MuScriptor cannot be independently excluded."
        ),
        (
            "- One run per configuration/example; runtime is descriptive, not a repeated latency "
            "benchmark. The first run may include additional kernel warm-up. GPU model load is "
            "recorded separately in results.json."
        ),
        (
            f"- Actual decoded GPU-call time: {payload['actual_inference_seconds']:.2f} seconds. "
            f"Estimated inference-only cost: ${payload['estimated_inference_cost_usd']:.4f}, "
            "using the project's historical rate. This excludes startup/idle/CPU/build overhead "
            "and is not the Modal bill."
        ),
        "",
        "## Inspect the artifacts",
        "",
    ]
    for name in examples:
        lines.append(
            f"- {name}: [source WAV]({name}/source.wav), "
            f"[reference MIDI]({name}/reference.mid), "
            f"[baseline MIDI]({name}/baseline.mid), "
            f"[conditioned MIDI]({name}/conditioned.mid)."
        )
    lines += [
        "",
        "## Sources",
        "",
        "- [URMP dataset and citation](https://labsites.rochester.edu/air/projects/URMP.html)",
        "- [Official annotation format](https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf)",
        f"- [Pinned public mirror](https://huggingface.co/datasets/{manifest['mirror']}/tree/{manifest['revision']})",
        (
            "- Exact URLs and SHA-256 checksums for source WAV, score MIDI, annotations, "
            "and prepared references are in manifest.json."
        ),
        "",
    ]
    path = root / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    print(report(parser.parse_args().run_directory).resolve())
