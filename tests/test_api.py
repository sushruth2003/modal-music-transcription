from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from music_transcription import api
from music_transcription.storage import initial_job_record

JOB_ID = "a" * 32


def completed_record():
    spec = {
        "job_id": JOB_ID,
        "source_name": "demo song.wav",
        "source_suffix": ".wav",
        "instruments": None,
    }
    record = initial_job_record(spec)
    record["state"] = "completed"
    record["result"] = {
        "note_count": 1,
        "audio_seconds": 2.0,
        "inference": {"seconds": 0.5},
        "artifacts": {"events": "internal", "midi": "internal"},
    }
    return record


def test_copy_limited_rejects_empty_and_oversized_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        api.copy_limited(io.BytesIO(b""), tmp_path / "empty.wav", max_bytes=10)

    with pytest.raises(api.UploadTooLargeError, match="limited"):
        api.copy_limited(io.BytesIO(b"12345"), tmp_path / "large.wav", max_bytes=4)


def test_pair_note_events_ignores_unmatched_and_invalid_notes() -> None:
    events = [
        {"type": "note_start", "index": 2, "pitch": 64, "instrument": "piano", "time": 1.0},
        {"type": "note_end", "index": 99, "time": 1.2},
        {"type": "note_end", "index": 2, "time": 1.75},
        {"type": "note_start", "index": 3, "pitch": 67, "instrument": "piano", "time": 2.0},
        {"type": "note_end", "index": 3, "time": 1.0},
    ]

    assert api.pair_note_events(events) == [
        {"index": 2, "pitch": 64, "instrument": "piano", "start": 1.0, "end": 1.75}
    ]


def test_public_job_record_hides_paths_and_labels_cost_as_estimate() -> None:
    payload = api.public_job_record(completed_record())

    assert "paths" not in payload
    assert "artifacts" not in payload["result"]
    assert payload["progress"] == 100
    assert payload["result"]["inference"]["estimated_gpu_cost_usd"] == 0.000111


def test_submit_returns_accepted_job_handle(monkeypatch) -> None:
    spec = {
        "job_id": JOB_ID,
        "source_name": "demo.wav",
        "source_suffix": ".wav",
        "instruments": ["piano"],
    }
    monkeypatch.setattr(api, "stage_uploaded_file", lambda *_args: (spec, 5))
    monkeypatch.setattr(api, "spawn_process_job", lambda _spec: "fc-123")
    client = TestClient(api.create_web_app())

    response = client.post(
        "/transcriptions",
        files={"audio": ("demo.wav", b"audio", "audio/wav")},
        data={"instruments": "piano"},
    )

    assert response.status_code == 202
    assert response.headers["location"] == f"/transcriptions/{JOB_ID}"
    assert response.json() == {
        "job_id": JOB_ID,
        "state": "submitted",
        "source_bytes": 5,
        "function_call_id": "fc-123",
        "status_url": f"/transcriptions/{JOB_ID}",
        "result_url": f"/jobs/{JOB_ID}",
    }


def test_status_and_piano_roll_endpoints(monkeypatch) -> None:
    record = completed_record()
    event_bytes = (
        b'{"type":"note_start","index":1,"pitch":60,"instrument":"piano","time":0.25}\n'
        b'{"type":"note_end","index":1,"time":1.5}\n'
    )
    monkeypatch.setattr(api, "get_job", lambda _job_id: record)
    monkeypatch.setattr(api, "read_artifact_bytes", lambda _path: event_bytes)
    client = TestClient(api.create_web_app())

    status = client.get(f"/transcriptions/{JOB_ID}")
    roll = client.get(f"/transcriptions/{JOB_ID}/piano-roll")

    assert status.status_code == 200
    assert status.json()["state"] == "completed"
    assert roll.status_code == 200
    assert roll.json() == {
        "job_id": JOB_ID,
        "duration": 2.0,
        "instruments": ["piano"],
        "notes": [
            {
                "index": 1,
                "pitch": 60,
                "instrument": "piano",
                "start": 0.25,
                "end": 1.5,
            }
        ],
    }


def test_job_page_and_health_are_served() -> None:
    client = TestClient(api.create_web_app())

    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get(f"/jobs/{JOB_ID}")
    assert page.status_code == 200
    assert "MuScriptor Studio" in page.text
