"""Compare native model exports against the frozen suite, without reference alignment."""

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from music_transcription.benchmarks.quality import evaluate, read_midi
from music_transcription.benchmarks.report_quality_suite import METRICS, bootstrap, packed

MODELS = {
    "muscriptor_raw": {
        "label": "MuScriptor",
        "family": "Autoregressive music-token model",
        "scope": "Full mixtures",
        "device": "L4 · FP16",
        "color": "#60758a",
    },
    "muscriptor_product": {
        "label": "MuScriptor · app export",
        "family": "Same model + beat export",
        "scope": "Full mixtures",
        "device": "L4 + CPU beats",
        "color": "#60758a",
    },
    "muscriptor_beam4": {
        "label": "MuScriptor · beam 4",
        "family": "Same model, wider search",
        "scope": "Full mixtures",
        "device": "L4 · FP16",
        "color": "#9c8fbb",
    },
    "yourmt3_t5": {
        "label": "YourMT3+ · T5",
        "family": "T5 encoder + single token decoder",
        "scope": "Full mixtures",
        "device": "L4 · FP32",
        "color": "#d48447",
        "source": "https://huggingface.co/spaces/mimbres/YourMT3/blob/5e66c1ea173a8186e0d20432b841d3180cc015b5/app.py",
    },
    "yourmt3_moe": {
        "label": "YourMT3+ · MoE",
        "family": "Perceiver-TF + experts + multi-channel decoder",
        "scope": "Full mixtures",
        "device": "L4 · FP32",
        "color": "#18867b",
        "source": "https://arxiv.org/html/2407.04822v2",
    },
    "transkun": {
        "label": "TransKun v2",
        "family": "Transformer interval scores + semi-CRF decoder",
        "scope": "Piano only",
        "device": "L4 · FP32",
        "color": "#bb5673",
        "source": "https://github.com/Yujia-Yan/Transkun",
    },
    "basic_pitch": {
        "label": "Basic Pitch",
        "family": "Lightweight convolutional pitch/onset model",
        "scope": "Pitch only · best on a single instrument",
        "device": "2 CPU cores · ONNX",
        "color": "#b4983f",
        "source": "https://github.com/spotify/basic-pitch",
    },
}


def subset(example, name):
    if name == "All 32":
        return True
    if name == "Official test 16":
        return example["dataset"] in {"MAESTRO", "Slakh"}
    return example["dataset"] == name


def aggregate_architectures(records, examples):
    summary = {}
    for comparator in ["muscriptor_raw", "muscriptor_product"]:
        summary[comparator] = {}
        for group in ["Official test 16", "All 32", "Slakh", "MAESTRO", "URMP", "GuitarSet"]:
            ids = {e["id"] for e in examples if subset(e, group)}
            selected = [r for r in records if r["example"] in ids]
            baseline = {r["example"]: r for r in selected if r["model"] == comparator}
            summary[comparator][group] = {}
            for model in MODELS:
                rows = [r for r in selected if r["model"] == model]
                # Never rank a piano-only model's eight results against 16/32 results.
                if not ids or len(rows) != len(ids) or {r["example"] for r in rows} != ids:
                    continue
                summary[comparator][group][model] = {
                    metric: bootstrap(rows, baseline, metric)
                    if all(r["metrics"][metric] is not None for r in rows)
                    else None
                    for metric in METRICS
                }
    return summary


def build(
    root=Path("outputs/quality-architectures-20260906"),
    previous=Path("outputs/quality-suite-20260905"),
):
    old = json.loads((previous / "analysis.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    examples = manifest["examples"]
    runs = {
        k: json.loads((root / f"{k}.run.json").read_text())
        for k in MODELS
        if (root / f"{k}.run.json").exists()
    }
    records = []
    demo_records = []
    audition = {}
    for e in examples:
        folder = root / e["id"]
        folder.mkdir(exist_ok=True)
        for name, key in [("source.wav", "audio_sha256"), ("reference.mid", "reference_sha256")]:
            source = previous / e["id"] / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != e[key]:
                raise ValueError("Frozen input checksum mismatch")
            shutil.copyfile(source, folder / name)
        ref = read_midi(folder / "reference.mid")
        audition[e["id"]] = {"reference": packed(ref)}
        for key, variant, mode, filename in [
            ("muscriptor_raw", "baseline", "raw", "baseline.raw.mid"),
            ("muscriptor_product", "baseline", "product", "baseline.aligned.mid"),
            ("muscriptor_beam4", "auto_beam4", "raw", "auto_beam4.raw.mid"),
        ]:
            record = next(
                r
                for r in old["records"]
                if r["example"] == e["id"] and r["variant"] == variant and r["mode"] == mode
            )
            records.append({**record, "model": key})
            shutil.copyfile(previous / e["id"] / filename, folder / f"{key}.mid")
            audition[e["id"]][key] = packed(read_midi(folder / f"{key}.mid"))
        for key, run in runs.items():
            if e["id"] not in run["clips"]:
                continue
            clip = run["clips"][e["id"]]
            path = folder / f"{key}.mid"
            if hashlib.sha256(path.read_bytes()).hexdigest() != clip["midi_sha256"]:
                raise ValueError("Prediction checksum mismatch")
            notes = read_midi(path)
            demo_path = folder / f"{key}.demo.mid"
            if demo_path.exists() and "demo_midi_sha256" in clip:
                if hashlib.sha256(demo_path.read_bytes()).hexdigest() != clip["demo_midi_sha256"]:
                    raise ValueError("Demo export checksum mismatch")
                demo_records.append(
                    {
                        "example": e["id"],
                        "dataset": e["dataset"],
                        "model": key,
                        "seconds": clip["seconds"],
                        "metrics": evaluate(
                            ref,
                            read_midi(demo_path),
                            start=e["score_start"],
                            end=e["score_end"],
                            audio_end=e["audio_seconds"],
                        ),
                    }
                )
            metrics = evaluate(
                ref, notes, start=e["score_start"], end=e["score_end"], audio_end=e["audio_seconds"]
            )
            # A pitch-only model's default playback patch is not an instrument prediction.
            if key == "basic_pitch":
                for metric in ["instrument_onset", "instrument_onset_offset", "drum_onset"]:
                    metrics[metric] = None
            records.append(
                {
                    "example": e["id"],
                    "dataset": e["dataset"],
                    "model": key,
                    "seconds": clip["seconds"],
                    "metrics": metrics,
                }
            )
            audition[e["id"]][key] = packed(notes)
    summary = aggregate_architectures(records, examples)
    interpretation = (
        json.loads((root / "interpretation.json").read_text())
        if (root / "interpretation.json").exists()
        else {"title": "The architecture comparison is running.", "findings": []}
    )
    initial_runs = {
        p.stem: json.loads(p.read_text()) for p in (root / "initial-demo-export").glob("*.run.json")
    }
    demo_summary = {}
    for group in ["Official test 16", "All 32", "Slakh", "MAESTRO", "URMP", "GuitarSet"]:
        ids = {e["id"] for e in examples if subset(e, group)}
        demo_summary[group] = {}
        for model in ["yourmt3_t5", "yourmt3_moe"]:
            demo = [r for r in demo_records if r["example"] in ids and r["model"] == model]
            detailed = {
                r["example"]: r for r in records if r["example"] in ids and r["model"] == model
            }
            if len(demo) == len(ids):
                demo_summary[group][model] = {
                    metric: bootstrap(demo, detailed, metric) for metric in METRICS
                }
    data = {
        "created_at": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "examples": examples,
        "records": records,
        "summary": summary,
        "audition": audition,
        "runs": runs,
        "demo_summary": demo_summary,
        "demo_records": demo_records,
        "initial_runs": initial_runs,
        "interpretation": interpretation,
        "sources": old["sources"],
        "completed": all(
            k in runs and len(runs[k]["clips"]) == (8 if k == "transkun" else 32)
            for k in ["yourmt3_t5", "yourmt3_moe", "transkun", "basic_pitch"]
        ),
    }
    (root / "analysis.json").write_text(json.dumps(data, indent=2) + "\n")
    with (root / "per-recording.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "dataset", "model", "seconds"] + METRICS)
        for r in records:
            writer.writerow(
                [r["example"], r["dataset"], r["model"], r["seconds"]]
                + [r["metrics"][m]["f1"] if r["metrics"][m] else "" for m in METRICS]
            )
    template = Path(__file__).with_name("architecture_report_template.html").read_text()
    (root / "index.html").write_text(
        template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    )
    print(root / "index.html")
    for group in ["Official test 16", "MAESTRO", "Slakh", "All 32"]:
        print(group)
        for k, metrics in summary["muscriptor_raw"][group].items():
            x = metrics["onset"]
            print(
                k,
                round(x["mean"] * 100, 2),
                round(metrics["onset_offset"]["mean"] * 100, 2),
                round(x["mean_seconds"], 2),
                "delta CI",
                round(x["low"] * 100, 2),
                round(x["high"] * 100, 2),
            )


if __name__ == "__main__":
    build()
