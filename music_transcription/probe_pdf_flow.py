"""Authenticated CPU-only probe of the deployed saved-MIDI -> PDF HTTP flow.

Creates an explicitly labelled synthetic fixture using the owner's Modal account;
never runs transcription or spends public audio-submission quota.

    python -m music_transcription.probe_pdf_flow --prepare-only
    python -m music_transcription.probe_pdf_flow --job-id <printed-id>
"""

import argparse
import hashlib
import io
import json
import math
import time
import wave
from array import array
from pathlib import Path

from music_transcription.note_export import Note, serialize_notes
from music_transcription.probe_score import fixture_midi
from music_transcription.resources import artifact_volume
from music_transcription.storage import job_paths, new_job_spec, stage_job_sources, update_job


def prepare_fixture(destination):
    import mido

    midi, _ = fixture_midi("polyphonic")
    clock = 0.0
    active = {}
    notes = []
    for message in mido.MidiFile(file=io.BytesIO(midi)):
        clock += message.time
        if message.type == "note_on" and message.velocity:
            active[(message.channel, message.note)] = clock
        elif message.type in ("note_on", "note_off"):
            onset = active.pop((message.channel, message.note))
            notes.append(Note(onset, clock, message.note, 0, "acoustic_piano"))
    duration = max(n.end for n in notes) + 0.2
    samples = [0.0] * math.ceil(duration * 16000)
    for note in notes:
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        for i in range(round(note.start * 16000), min(len(samples), round(note.end * 16000))):
            samples[i] += 4000 * math.sin(2 * math.pi * frequency * i / 16000)
    pcm = array("h", (int(max(-32767, min(32767, s))) for s in samples))
    source = destination / "pdf-probe-polyphony.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    spec = new_job_spec(source.name, None)
    stage_job_sources([source], [spec])
    paths = job_paths(spec["job_id"], ".wav")
    events = "\n".join(json.dumps(e) for e in serialize_notes(notes)) + "\n"
    result = {
        "job_id": spec["job_id"],
        "note_count": len(notes),
        "audio_seconds": duration,
        "instruments": ["acoustic_piano"],
        "model": {"name": "PDF probe: synthetic MIDI fixture; no transcription"},
        "artifacts": {k: paths[k] for k in ["midi", "events", "metrics"]},
    }
    with artifact_volume.batch_upload() as upload:
        upload.put_file(io.BytesIO(midi), "/" + paths["midi"])
        upload.put_file(io.BytesIO(events.encode()), "/" + paths["events"])
        upload.put_file(io.BytesIO(json.dumps(result).encode()), "/" + paths["metrics"])
    update_job(spec["job_id"], "completed", result=result)
    return spec["job_id"]


def main():
    import httpx
    from pypdf import PdfReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://sushruthb03--transcribe.modal.run")
    parser.add_argument("--job-id")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = Path("outputs/pdf-probe-20260906")
    out.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or prepare_fixture(out)
    print(f"PDF probe job: {job_id}", flush=True)
    (out / "flow-job.json").write_text(json.dumps({"job_id": job_id}))
    if args.prepare_only:
        return
    base = args.base_url.rstrip("/")
    route = f"{base}/transcriptions/{job_id}"
    with httpx.Client(timeout=90) as client:
        before = client.get(route)
        before.raise_for_status()
        before = before.json()
        midi = client.get(route + "/midi")
        midi.raise_for_status()
        response = client.post(route + "/score")
        response.raise_for_status()
        assert response.status_code in (200, 202)
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            status = client.get(route)
            status.raise_for_status()
            job = status.json()
            assert job["state"] == "completed", "Rendering must not hide the original transcription"
            if job["score_state"] == "completed":
                break
            if job["score_state"] == "failed":
                raise AssertionError(job.get("score_error"))
            time.sleep(2)
        else:
            raise TimeoutError("Score did not finish within 8 minutes")
        pdf = client.get(base + job["links"]["score_pdf"])
        pdf.raise_for_status()
        assert pdf.headers["content-type"] == "application/pdf"
        parsed = PdfReader(io.BytesIO(pdf.content), strict=True)
        assert len(parsed.pages) > 0
        again = client.post(route + "/score")
        assert again.status_code == 200
        after = client.get(route + "/midi")
        after.raise_for_status()
        assert after.content == midi.content
        assert job["result"]["model"] == before["result"]["model"]
        (out / "flow.pdf").write_bytes(pdf.content)
        report = {
            "job_id": job_id,
            "result_url": f"{base}/jobs/{job_id}",
            "engine": job["result"]["score"],
            "midi_unchanged": True,
            "model_unchanged": True,
            "idempotent_status": again.status_code,
            "pdf_sha256": hashlib.sha256(pdf.content).hexdigest(),
        }
        (out / "flow.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
