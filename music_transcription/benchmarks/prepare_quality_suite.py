"""Freeze and download a balanced 32-recording, four-source transcription suite."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly

from music_transcription.benchmarks.prepare_quality_data import INSTRUMENTS, sha256
from music_transcription.benchmarks.quality import PROGRAM_GROUPS, Note, read_midi, write_midi

ROOT = Path("data/quality-suite")
REPO = "J1mmymm/MIMuT_Data_v2"
REV = "bb320faf307f5d24aeced0e60f9445ff0abce205"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
URMP_BASE = "https://huggingface.co/datasets/Eredis02/URMP/resolve/58177a0f0f816e621b3d5d304c5fd1a03c035a86/"
URMP = [
    "01_Jupiter_vn_vc",
    "03_Dance_fl_cl",
    "05_Entertainer_tpt_tpt",
    "11_Maria_ob_vc",
    "13_Hark_vn_vn_va",
    "21_Rejouissance_cl_tbn_tba",
    "35_Rondeau_vn_vn_va_db",
    "44_K515_vn_vn_va_va_vc",
]
CODES = {**INSTRUMENTS, "vc": "cello", "va": "viola", "db": "contrabass", "tba": "tuba"}
SOURCES = {
    "URMP": {
        "url": "https://labsites.rochester.edu/air/projects/URMP.html",
        "kind": "Recorded ensembles",
        "terms": "Upstream research/source terms; local evaluation only",
        "citation": "Bochen Li et al., Creating a multi-track classical music performance dataset, 2018.",
    },
    "Slakh": {
        "url": "https://zenodo.org/records/4599666",
        "kind": "Rendered band mixtures",
        "terms": "CC BY 4.0",
        "citation": "Ethan Manilow et al., Cutting Music Source Separation Some Slakh, 2019.",
    },
    "GuitarSet": {
        "url": "https://guitarset.weebly.com/",
        "kind": "Recorded acoustic guitar",
        "terms": "CC BY 4.0",
        "citation": "Qingyang Xi et al., GuitarSet: A Dataset for Guitar Transcription, 2018.",
    },
    "MAESTRO": {
        "url": "https://magenta.tensorflow.org/datasets/maestro",
        "kind": "Recorded piano",
        "terms": "CC BY-NC-SA 4.0; local research evaluation",
        "citation": "Curtis Hawthorne et al., Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset, 2019.",
    },
}


def fetch(url: str, target: Path, limit: int = 100_000_000) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        for attempt in range(4):
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    target.with_suffix(target.suffix + ".partial").open("wb") as out,
                ):
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Download size bound exceeded")
                        out.write(chunk)
                target.with_suffix(target.suffix + ".partial").replace(target)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    return {
        "url": url,
        "file": str(target.relative_to(ROOT)),
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
    }


def listing(path: str) -> list[dict]:
    # Preserve API pagination rather than silently restricting candidate selection.
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/{path}?limit=1000"
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=90) as response:
            rows.extend(json.load(response))
            link = response.headers.get("Link", "")
            url = next(
                (p.split(">")[0].strip(" <") for p in link.split(",") if 'rel="next"' in p), None
            )
    return rows


def rank(value: str) -> str:
    return hashlib.sha256(("muscriptor-quality-20260905:" + value).encode()).hexdigest()


def freeze() -> list[dict]:
    path = ROOT / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    selected = [
        {
            "dataset": "URMP",
            "id": "urmp_" + p,
            "piece": p,
            "start": 0.0,
            "split": "new exploratory recordings",
        }
        for p in URMP
    ]
    slakh = sorted(
        (r["path"] for r in listing("data/Slakh2100_redux/test") if r["type"] == "directory"),
        key=rank,
    )[:8]
    selected.extend(
        {
            "dataset": "Slakh",
            "id": "slakh_" + p.split("/")[-1],
            "path": p,
            "start": 30.0,
            "split": "official test",
        }
        for p in slakh
    )
    guitar = sorted(
        (
            r["path"]
            for r in listing("data/guitarset/guitarset_yourmt3_16k/annotation")
            if r["path"].endswith(".jams")
        ),
        key=rank,
    )
    used = set()
    styles = set()
    chosen = []
    # First cover every style, then add different lead sheets; never use model output.
    for diversity in (True, False):
        for p in guitar:
            name = Path(p).stem
            sheet = name.split("_")[1]
            style = sheet[:2]
            if sheet in used or (diversity and style in styles):
                continue
            used.add(sheet)
            styles.add(style)
            chosen.append(p)
            if len(chosen) == 8:
                break
        if len(chosen) == 8:
            break
    selected.extend(
        {
            "dataset": "GuitarSet",
            "id": "guitar_" + Path(p).stem,
            "path": p,
            "start": 0.0,
            "split": "no official split; unique lead sheets",
            "style": Path(p).stem.split("_")[1][:2],
        }
        for p in chosen
    )
    meta_path = ROOT / "sources/maestro-v3.0.0.json"
    fetch(
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json",
        meta_path,
    )
    meta = json.loads(meta_path.read_text())
    rows = [{k: v[i] for k, v in meta.items()} for i in meta["split"] if meta["split"][i] == "test"]
    composers = set()
    for r in sorted(rows, key=lambda x: rank(x["audio_filename"])):
        if r["canonical_composer"] in composers:
            continue
        composers.add(r["canonical_composer"])
        selected.append(
            {
                "dataset": "MAESTRO",
                "id": f"piano_{len(composers):02d}",
                "path": "data/maestro/test/" + r["audio_filename"],
                "midi": r["midi_filename"],
                "start": 30.0,
                "split": "official test",
                "title": r["canonical_title"],
                "composer": r["canonical_composer"],
            }
        )
        if len(composers) == 8:
            break
    path.write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def prepare_one(e: dict) -> dict:
    folder = ROOT / e["id"]
    folder.mkdir(exist_ok=True)
    sources = []
    notes = []
    dataset = e["dataset"]

    def get(rel: str, base: str = BASE) -> Path:
        target = folder / Path(rel).name
        sources.append(fetch(base + urllib.parse.quote(rel, safe="/"), target))
        return target

    if dataset == "URMP":
        p = e["piece"]
        num, title, *codes = p.split("_")
        audio = get(f"{p}/AuMix_{p}.wav", URMP_BASE)
        get(f"{p}/Sco_{p}.mid", URMP_BASE)
        for i, code in enumerate(codes, 1):
            annotations = get(f"{p}/Notes_{i}_{code}_{num}_{title}.txt", URMP_BASE)
            for start, hz, length in np.loadtxt(annotations, ndmin=2):
                if length > 0 and hz > 0:
                    notes.append(
                        Note(
                            round(pretty_midi.hz_to_note_number(hz)),
                            float(start),
                            float(start + length),
                            PROGRAM_GROUPS[CODES[code]][0],
                        )
                    )
        policy = "Official performance Notes annotations; score MIDI retained as provenance, never timing reference."
    elif dataset == "GuitarSet":
        annotation = get(e["path"])
        name = annotation.stem
        audio = get(f"data/guitarset/guitarset_yourmt3_16k/audio_mono-mic/{name}_mic.wav")
        for a in json.loads(annotation.read_text())["annotations"]:
            if a["namespace"] == "note_midi":
                for n in a["data"]:
                    if n["duration"] > 0:
                        notes.append(
                            Note(round(n["value"]), n["time"], n["time"] + n["duration"], 24)
                        )
        policy = "Official JAMS per-string note_midi; pitches rounded to semitones. Prepared 16 kHz microphone audio."
    elif dataset == "Slakh":
        import yaml

        audio = get(e["path"] + "/mix.flac")
        metadata = yaml.safe_load(get(e["path"] + "/metadata.yaml").read_text())
        get(e["path"] + "/all_src.mid")
        for stem, info in metadata["stems"].items():
            if info["audio_rendered"] and info["midi_saved"]:
                notes.extend(read_midi(get(e["path"] + "/MIDI/" + stem + ".mid")))
        policy = "Only MIDI stems marked audio_rendered and midi_saved; unsounded source tracks excluded. Key-release offsets, no pedal extension."
    else:
        audio = get(e["path"])
        archive = ROOT / "sources/maestro-v3.0.0-midi.zip"
        with zipfile.ZipFile(archive) as z:
            target = folder / "original.mid"
            target.write_bytes(z.read("maestro-v3.0.0/" + e["midi"]))
        notes = read_midi(target)
        sources.append(
            {
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
                "archive_member": "maestro-v3.0.0/" + e["midi"],
                "file": str(target.relative_to(ROOT)),
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
        )
        policy = "Original aligned MAESTRO v3 performance MIDI, key-release offsets. Pedal extension is not evaluated. Prepared 16 kHz audio mirror."
    with sf.SoundFile(audio) as f:
        sr = f.samplerate
        full = f.frames / sr
        start = min(e["start"], max(0.0, full - 30.0))
        f.seek(round(start * sr))
        x = f.read(frames=30 * sr, dtype="float32", always_2d=True).mean(axis=1)
    x = resample_poly(x, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000))
    duration = len(x) / 16000
    if duration < 10:
        raise ValueError("Unexpected short recording")
    clipped = [
        Note(
            n.pitch, max(0.0, n.start - start), min(duration, n.end - start), n.program, n.velocity
        )
        for n in notes
        if n.end > start and n.start < start + duration
    ]
    sf.write(folder / "excerpt.wav", x, 16000, subtype="PCM_16")
    write_midi(clipped, folder / "reference.mid")
    programs = {n.program for n in clipped}
    names = {ps[0]: name for name, ps in PROGRAM_GROUPS.items()}
    names[128] = "drums"
    if programs - names.keys():
        raise ValueError(f"Unsupported reference programs: {programs - names.keys()}")
    result = {
        **e,
        "audio": f"{e['id']}/excerpt.wav",
        "reference": f"{e['id']}/reference.mid",
        "audio_seconds": duration,
        "excerpt_start_seconds": start,
        "original_seconds": full,
        "score_start": 2.5,
        "score_end": duration - 2.5,
        "instruments": sorted(names[p] for p in programs),
        "reference_note_count": len(clipped),
        "audio_sha256": sha256(folder / "excerpt.wav"),
        "reference_sha256": sha256(folder / "reference.mid"),
        "reference_policy": policy,
        "sources": sources,
    }
    print(
        f"Prepared {e['id']}: {duration:.1f}s, {len(clipped)} notes, {len(programs)} instruments",
        flush=True,
    )
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = freeze()
    archive = ROOT / "sources/maestro-v3.0.0-midi.zip"
    source = fetch(
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        archive,
    )
    if source["sha256"] != "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c":
        raise ValueError("Official MAESTRO MIDI checksum mismatch")
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        examples = list(pool.map(prepare_one, selected))
    manifest = {
        "dataset": "Balanced four-source suite",
        "mirror": REPO,
        "revision": REV,
        "sources": SOURCES,
        "archive_sources": [source],
        "selection": "Frozen before inference: 8 new URMP ensemble recordings; 8 SHA256-ranked Slakh Redux official-test tracks; 8 GuitarSet unique lead sheets covering five styles; 8 MAESTRO official-test works by distinct composers. Fixed first 30s for URMP/guitar, 30–60s for Slakh/piano, shorter clips retained. No filtering by model scores.",
        "selection_sha256": sha256(ROOT / "selection.json"),
        "examples": examples,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(ROOT / "manifest.json")


if __name__ == "__main__":
    main()
