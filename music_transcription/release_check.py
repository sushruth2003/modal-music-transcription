"""Bounded pre-deploy integration checks. GPU budget estimate: under $0.25.

Runs the new graph ephemerally without changing the deployed app. Uses four
previously authorized, frozen evaluation excerpts plus synthetic format/edge cases.
"""

import json
import wave
from pathlib import Path

from music_transcription import beat_grid as _beat_grid  # noqa: F401
from music_transcription import transcribe as _transcribe  # noqa: F401
from music_transcription.preprocess import process_job
from music_transcription.resources import app, artifact_volume, audio_image
from music_transcription.storage import get_job, job_paths, new_job_spec, stage_job_sources


@app.local_entrypoint()
def main():
    import numpy as np

    out = Path("outputs/yourmt3-release-20260906")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("data/quality-suite/manifest.json").read_text())
    selected = [
        next(e for e in manifest["examples"] if e["dataset"] == name)
        for name in ["URMP", "Slakh", "GuitarSet", "MAESTRO"]
    ]
    cases = [
        (e["id"], Path("data/quality-suite") / e["audio"], None, i == 0)
        for i, e in enumerate(selected)
    ]
    for name, seconds in [("silence", 2), ("long-synthetic", 65)]:
        samples = np.zeros(int(16000 * seconds), dtype=np.int16)
        if name != "silence":
            t = np.arange(len(samples)) / 16000
            samples = (6000 * np.sin(2 * np.pi * 440 * t) * (t % 1 < 0.6)).astype(np.int16)
        path = out / f"{name}.wav"
        with wave.open(str(path), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            f.writeframes(samples.tobytes())
        cases.append((name, path, None, False))
    video = out / "filtered-video.mp4"
    video.write_bytes(make_video.remote(cases[0][1].read_bytes()))
    cases.append(("filtered-video", video, ["violin"], True))
    results = []
    for name, path, instruments, score in cases:
        spec = new_job_spec(path.name, instruments, generate_score=score)
        stage_job_sources([path], [spec])
        print(f"CHECK {name}: {spec['job_id']}", flush=True)
        result = process_job.remote(spec)
        assert result["model"]["name"] == "YourMT3+ MoE"
        assert get_job(spec["job_id"])["state"] == "completed"
        if instruments:
            assert set(result["instruments"]) <= set(instruments)
        dest = out / name
        dest.mkdir(exist_ok=True)
        paths = job_paths(spec["job_id"], spec["source_suffix"])
        for key in ["midi", "events", "metrics"] + (["score_pdf"] if score else []):
            payload = b"".join(artifact_volume.read_file(paths[key]))
            (dest / Path(paths[key]).name).write_bytes(payload)
            if key == "score_pdf":
                assert payload.startswith(b"%PDF")
        results.append({"case": name, "spec": spec, "result": result})
        (out / "checks.json").write_text(json.dumps(results, indent=2))
        print(f"PASS {name}: {result['note_count']} notes", flush=True)


@app.function(image=audio_image, timeout=120)
def make_video(audio: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source.wav"
        target = Path(folder) / "video.mp4"
        source.write_bytes(audio)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x120:r=1",
                "-i",
                str(source),
                "-t",
                "8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(target),
            ],
            check=True,
        )
        return target.read_bytes()
