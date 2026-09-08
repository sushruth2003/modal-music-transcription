"""Download three small URMP examples and construct performance-aligned reference MIDI.

The supplied score MIDI is also downloaded, but its score clock is NOT a valid
audio-timing reference. We use the official Notes_* annotations instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly

from music_transcription.benchmarks.quality import Note, program_for_name, write_midi

REPO = "Eredis02/URMP"
REVISION = "58177a0f0f816e621b3d5d304c5fd1a03c035a86"
PIECES = ["08_Spring_fl_vn", "31_Slavonic_tpt_tpt_hn_tbn", "40_Miserere_fl_fl_ob_cl_bn"]
INSTRUMENTS = {
    "fl": "flutes",
    "vn": "violin",
    "tpt": "trumpet",
    "hn": "french_horn",
    "tbn": "trombone",
    "ob": "oboe",
    "cl": "clarinet",
    "bn": "bassoon",
}
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(relative: str, destination: Path) -> dict:
    url = f"{BASE_URL}/{relative}"
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 50 * 1024 * 1024:
                    raise ValueError("Dataset file exceeds the 50 MiB download bound")
                out.write(block)
        partial.replace(destination)
    return {
        "url": url,
        "file": destination.name,
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def prepare(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for piece in PIECES:
        folder = output / piece
        folder.mkdir(exist_ok=True)
        sources = []
        filenames = [f"AuMix_{piece}.wav", f"Sco_{piece}.mid"]
        number, title, *codes = piece.split("_")
        filenames += [f"Notes_{i}_{code}_{number}_{title}.txt" for i, code in enumerate(codes, 1)]
        for filename in filenames:
            sources.append(download(f"{piece}/{filename}", folder / filename))
        # Preserve the original full-resolution WAV; normalize just the fixed excerpt.
        with sf.SoundFile(folder / filenames[0]) as source:
            sr = source.samplerate
            audio = source.read(frames=30 * sr, dtype="float32", always_2d=True).mean(axis=1)
        audio = resample_poly(audio, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000))
        duration = len(audio) / 16000
        sf.write(folder / "excerpt.wav", audio, 16000, subtype="PCM_16")
        notes = []
        for filename, code in zip(filenames[2:], codes, strict=True):
            values = np.loadtxt(folder / filename, ndmin=2)
            for start, hz, length in values:
                if start < duration and length > 0 and hz > 0:
                    notes.append(
                        Note(
                            round(pretty_midi.hz_to_note_number(hz)),
                            float(start),
                            float(min(start + length, duration)),
                            program_for_name(INSTRUMENTS[code]),
                        )
                    )
        write_midi(notes, folder / "reference.mid")
        examples.append(
            {
                "id": piece,
                "audio": f"{piece}/excerpt.wav",
                "reference": f"{piece}/reference.mid",
                "original_score": f"{piece}/Sco_{piece}.mid",
                "instruments": sorted({INSTRUMENTS[code] for code in codes}),
                "audio_seconds": duration,
                "score_start": 2.5,
                "score_end": duration - 2.5,
                "reference_note_count": len(notes),
                "audio_sha256": sha256(folder / "excerpt.wav"),
                "reference_sha256": sha256(folder / "reference.mid"),
                "sources": sources,
            }
        )
        print(f"Prepared {piece}: {duration:.1f}s, {len(notes)} reference notes", flush=True)
    manifest = {
        "dataset": "URMP",
        "mirror": REPO,
        "revision": REVISION,
        "official_url": "https://labsites.rochester.edu/air/projects/URMP.html",
        "annotation_documentation": "https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf",
        "reference_policy": "Official performance Notes_*: onset seconds, pitch Hz, duration seconds; converted to MIDI. Score MIDI is provenance only.",
        "selection": "First 30 seconds of three preselected duet/quartet/quintet recordings. No selection based on model scores.",
        "examples": examples,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/quality-urmp"))
    args = parser.parse_args()
    print(prepare(args.output))
