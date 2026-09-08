"""CPU-only real MIDI-to-PDF probe. Budget: under $0.03, no GPU or audio quota.

modal run -m music_transcription.probe_score
modal run -m music_transcription.probe_score --deployed
"""

import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

from music_transcription.config import APP_NAME, SCORE_ENGINE_COMMAND, SCORE_ENGINE_VERSION
from music_transcription.resources import app, score_base_image

probe_image = (
    score_base_image.apt_install("poppler-utils")
    .uv_pip_install("mido==1.3.3", "Pillow==11.3.0")
    .add_local_python_source("music_transcription")
)


def fixture_midi(kind):
    """Deterministic polyphony, named parts, percussion and a multi-page score."""
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    header = mido.MidiTrack()
    midi.tracks.append(header)
    header.extend(
        [
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.MetaMessage("time_signature", numerator=4, denominator=4),
            mido.MetaMessage("key_signature", key="C"),
        ]
    )
    bars = 128 if kind == "multipage" else 4
    parts = [("Piano", 0, 0, 60)]
    if kind == "ensemble":
        parts = [
            ("Violin", 40, 0, 72),
            ("Cello", 42, 1, 48),
            ("Flute", 73, 2, 84),
            ("Drums", 0, 9, 36),
        ]
    counts = {}
    for name, program, channel, base_pitch in parts:
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.extend(
            [
                mido.MetaMessage("track_name", name=name),
                mido.Message("program_change", program=program, channel=channel),
            ]
        )
        events = []
        for bar in range(bars):
            for beat in range(4):
                pitches = (
                    [base_pitch + (0, 2, 4, 7)[beat]]
                    if channel != 9
                    else [36 if beat % 2 == 0 else 38]
                )
                if kind == "polyphonic":
                    pitches = [base_pitch, base_pitch + 4, base_pitch + 7]
                for pitch in pitches:
                    onset = (bar * 4 + beat) * 480
                    events.extend([(onset, 1, pitch), (onset + 420, 0, pitch)])
                if kind == "polyphonic" and beat == 0:
                    events.extend([(bar * 4 * 480, 1, 48), (bar * 4 * 480 + 1800, 0, 48)])
        previous = 0
        for tick, on, pitch in sorted(events):
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    channel=channel,
                    note=pitch,
                    velocity=80 if on else 0,
                    time=tick - previous,
                )
            )
            previous = tick
        counts[name] = sum(on for _, on, _ in events)
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue(), counts


@app.function(image=probe_image, cpu=2, memory=4096, timeout=300)
def probe_score_engine():
    import xml.etree.ElementTree as ET

    from PIL import Image, ImageChops, ImageStat
    from pypdf import PdfReader

    from music_transcription.score import render_command, render_score_file

    version = subprocess.run(
        ["xvfb-run", "-a", SCORE_ENGINE_COMMAND, "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    if version.returncode:
        raise RuntimeError(f"Engine startup failed: {version.stderr} {version.stdout}")
    if SCORE_ENGINE_VERSION not in version.stdout + version.stderr:
        raise AssertionError("Unexpected engraving engine version")
    results = []
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for kind in ["polyphonic", "ensemble", "multipage"]:
            print(f"Engraving probe: {kind}", flush=True)
            midi, counts = fixture_midi(kind)
            source = folder / f"{kind}.mid"
            source.write_bytes(midi)
            output = folder / f"{kind}.pdf"
            start = time.perf_counter()
            metadata = render_score_file(source, output)
            notation = folder / f"{kind}.musicxml"
            conversion = subprocess.run(
                render_command(source, notation),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if conversion.returncode or not notation.exists():
                raise AssertionError("Could not inspect imported notation")
            tree = ET.parse(notation)
            imported_notes = [
                note
                for note in tree.findall(".//note")
                if note.find("rest") is None
                and not any(tie.get("type") == "stop" for tie in note.findall("tie"))
            ]
            if len(imported_notes) != sum(counts.values()):
                raise AssertionError(
                    f"MIDI import lost or added notes: {len(imported_notes)} vs {sum(counts.values())}"
                )
            metadata["imported_notes"] = len(imported_notes)
            metadata["imported_parts"] = len(tree.findall("part"))

            if kind == "multipage" and metadata["pages"] < 2:
                raise AssertionError("Multi-page fixture did not paginate")
            reader = PdfReader(output)
            if not any(page.get("/Resources", {}).get("/Font") for page in reader.pages):
                raise AssertionError("Score PDF has no fonts")
            subprocess.run(
                ["pdftoppm", "-scale-to", "1100", "-png", str(output), str(folder / kind)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            previews = []
            for path in sorted(folder.glob(f"{kind}-*.png")):
                with Image.open(path) as img:
                    gray = img.convert("L")
                    ink = ImageChops.difference(gray, Image.new("L", gray.size, 255))
                    if ImageStat.Stat(ink).mean[0] < 0.2:
                        raise AssertionError(f"Blank rendered page: {path.name}")
                    bounds = ink.getbbox()
                    if (
                        bounds is None
                        or bounds[0] <= 1
                        or bounds[1] <= 1
                        or bounds[2] >= gray.width - 1
                        or bounds[3] >= gray.height - 1
                    ):
                        raise AssertionError(f"Content touches page boundary: {path.name}")
                previews.append({"name": path.name, "bytes": path.read_bytes()})
            results.append(
                {
                    "case": kind,
                    "input_notes": counts,
                    "seconds": time.perf_counter() - start,
                    **metadata,
                    "midi": midi,
                    "pdf": output.read_bytes(),
                    "previews": previews,
                }
            )
        malformed = folder / "malformed.mid"
        malformed.write_bytes(b"not a MIDI file")
        try:
            render_score_file(malformed, folder / "invalid.pdf")
        except (RuntimeError, ValueError):
            rejected = True
        else:
            raise AssertionError("Malformed MIDI was not rejected")
    return {
        "engine_version": SCORE_ENGINE_VERSION,
        "cases": results,
        "malformed_rejected": rejected,
    }


@app.local_entrypoint()
def main(deployed: bool = False):
    target = (
        modal.Function.from_name(APP_NAME, "probe_score_engine") if deployed else probe_score_engine
    )
    result = target.remote()
    destination = Path("outputs/pdf-probe-20260906")
    destination.mkdir(parents=True, exist_ok=True)
    for case in result["cases"]:
        (destination / f"{case['case']}.mid").write_bytes(case.pop("midi"))
        (destination / f"{case['case']}.pdf").write_bytes(case.pop("pdf"))
        for preview in case.pop("previews"):
            (destination / preview["name"]).write_bytes(preview["bytes"])
    result["deployed"] = deployed
    (destination / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
