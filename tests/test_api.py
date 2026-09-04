from __future__ import annotations

import io
import re

import pytest
from fastapi.testclient import TestClient

from music_transcription import api
from music_transcription.storage import initial_job_record

JOB_ID = "a" * 32
ALLOW_SUBMISSION = api.RateLimitDecision(
    allowed=True,
    retry_after_seconds=0,
    ip_remaining=2,
    global_remaining=9,
)


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
        "instruments": ["acoustic_piano"],
    }
    monkeypatch.setattr(api, "stage_uploaded_file", lambda *_args: (spec, 5))
    monkeypatch.setattr(api, "spawn_process_job", lambda _spec: "fc-123")
    monkeypatch.setattr(api, "reserve_submission", lambda _client_ip: ALLOW_SUBMISSION)
    client = TestClient(api.create_web_app())

    response = client.post(
        "/transcriptions",
        files={"media": ("demo.wav", b"audio", "audio/wav")},
        data={"instruments": "acoustic_piano"},
    )

    assert response.status_code == 202
    assert response.headers["location"] == f"/transcriptions/{JOB_ID}"
    assert response.headers["x-ratelimit-ip-remaining"] == "2"
    assert response.headers["x-ratelimit-global-remaining"] == "9"
    assert response.json() == {
        "job_id": JOB_ID,
        "state": "submitted",
        "generate_score": False,
        "source_bytes": 5,
        "function_call_id": "fc-123",
        "status_url": f"/transcriptions/{JOB_ID}",
        "result_url": f"/jobs/{JOB_ID}",
    }


def test_submit_accepts_video_and_score_option(monkeypatch) -> None:
    spec = {
        "job_id": JOB_ID,
        "source_name": "performance.mp4",
        "source_suffix": ".mp4",
        "instruments": None,
        "generate_score": True,
    }
    monkeypatch.setattr(api, "stage_uploaded_file", lambda *_args: (spec, 9))
    monkeypatch.setattr(api, "spawn_process_job", lambda _spec: "fc-video")
    monkeypatch.setattr(api, "reserve_submission", lambda _client_ip: ALLOW_SUBMISSION)
    client = TestClient(api.create_web_app())

    response = client.post(
        "/transcriptions",
        files={"media": ("performance.mp4", b"video", "video/mp4")},
        data={"generate_score": "true"},
    )

    assert response.status_code == 202
    assert response.json()["generate_score"] is True
    assert response.json()["source_bytes"] == 9


def test_submit_rejects_missing_file_and_removed_url_input(monkeypatch) -> None:
    monkeypatch.setattr(api, "reserve_submission", lambda _client_ip: ALLOW_SUBMISSION)
    client = TestClient(api.create_web_app())

    missing = client.post("/transcriptions", data={})
    old_url_input = client.post("/transcriptions", data={"source_url": "https://youtu.be/example"})

    assert missing.status_code == 400
    assert old_url_input.status_code == 400


def test_submit_rejects_unknown_instrument_before_spawning(monkeypatch) -> None:
    spawned = []
    monkeypatch.setattr(api, "spawn_process_job", spawned.append)
    monkeypatch.setattr(api, "reserve_submission", lambda _client_ip: ALLOW_SUBMISSION)
    client = TestClient(api.create_web_app())

    response = client.post(
        "/transcriptions",
        files={"media": ("demo.wav", b"audio", "audio/wav")},
        data={"instruments": "grand_piano"},
    )

    assert response.status_code == 400
    assert "Unsupported instrument selection" in response.json()["detail"]
    assert spawned == []


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


def test_score_artifact_endpoints_are_conditional(monkeypatch) -> None:
    record = completed_record()
    record["generate_score"] = True
    monkeypatch.setattr(api, "get_job", lambda _job_id: record)
    monkeypatch.setattr(
        api,
        "read_artifact_bytes",
        lambda path: b"%PDF" if path.endswith(".pdf") else b"<score-partwise />",
    )
    client = TestClient(api.create_web_app())

    status_response = client.get(f"/transcriptions/{JOB_ID}")
    pdf = client.get(f"/transcriptions/{JOB_ID}/score.pdf")
    musicxml = client.get(f"/transcriptions/{JOB_ID}/musicxml")

    assert status_response.json()["links"]["score_pdf"].endswith("/score.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert "inline" in pdf.headers["content-disposition"]
    assert musicxml.status_code == 200
    assert "musicxml" in musicxml.headers["content-type"]


def test_video_playback_serves_extracted_audio(monkeypatch) -> None:
    record = completed_record()
    record["source_name"] = "performance.mp4"
    record["paths"]["source"] = f"jobs/{JOB_ID}/source.mp4"
    requested_paths = []
    monkeypatch.setattr(api, "get_job", lambda _job_id: record)
    monkeypatch.setattr(
        api,
        "read_artifact_bytes",
        lambda path: requested_paths.append(path) or b"RIFF",
    )
    client = TestClient(api.create_web_app())

    response = client.get(f"/transcriptions/{JOB_ID}/audio")

    assert response.status_code == 200
    assert requested_paths == [record["paths"]["normalized"]]
    assert response.headers["content-type"] == "audio/wav"
    assert "performance.wav" in response.headers["content-disposition"]


def test_job_page_and_health_are_served() -> None:
    client = TestClient(api.create_web_app())

    assert client.get("/api/health").json() == {"status": "ok"}
    instrument_payload = client.get("/api/instruments").json()
    options = [option for group in instrument_payload["groups"] for option in group["options"]]
    values = [option["value"] for option in options]
    assert len(values) == len(set(values)) == 35
    assert {option["value"] for option in options} >= {
        "acoustic_piano",
        "distorted_electric_guitar",
        "drums",
    }
    assert {option["label"] for option in options} >= {"Soprano & Alto Sax"}
    script_response = client.get("/app.js")
    assert script_response.headers["cache-control"] == "no-cache"
    declared_elements = set(
        re.findall(r"^\s+(\w+): document\.querySelector", script_response.text, re.MULTILINE)
    )
    referenced_elements = set(re.findall(r"\bels\.(\w+)\b", script_response.text))
    assert referenced_elements <= declared_elements
    page = client.get(f"/jobs/{JOB_ID}")
    assert page.status_code == 200
    assert "Auto Transcribe" in page.text
    assert "Source audio" in page.text
    assert "Transcription preview" in page.text
    assert "Drop audio or video here" in page.text
    assert "Paste URL" not in page.text
    assert "Auto-detect instruments" in page.text
    assert "MIDI + score" in page.text
    assert "/app.js?v=20260904-5" in page.text
    assert "/styles.css?v=20260904-2" in page.text


def test_rate_limit_allows_three_hourly_submissions_then_rejects() -> None:
    now = 1_800_000_000.0
    state = None
    decisions = []
    for _ in range(4):
        state, decision = api.evaluate_submission_limit(state, "client-a", now)
        decisions.append(decision)
        now += 1

    assert [decision.allowed for decision in decisions] == [True, True, True, False]
    assert decisions[2].ip_remaining == 0
    assert decisions[3].retry_after_seconds > 3_500


def test_rate_limit_enforces_global_day_and_resets_next_day() -> None:
    now = 1_800_000_000.0
    state = None
    for index in range(api.WEB_SUBMISSIONS_GLOBAL_DAY):
        state, decision = api.evaluate_submission_limit(state, f"client-{index}", now)
        assert decision.allowed

    state, denied = api.evaluate_submission_limit(state, "one-more", now)
    state, next_day = api.evaluate_submission_limit(state, "one-more", now + 24 * 60 * 60)

    assert not denied.allowed
    assert denied.global_remaining == 0
    assert next_day.allowed


def test_rate_limit_rejects_before_parsing_upload(monkeypatch) -> None:
    denied = api.RateLimitDecision(
        allowed=False,
        retry_after_seconds=60,
        ip_remaining=0,
        global_remaining=7,
    )
    monkeypatch.setattr(api, "reserve_submission", lambda _client_ip: denied)
    client = TestClient(api.create_web_app())

    response = client.post("/transcriptions")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["retry_after_seconds"] == 60


def test_benchmark_app_bypasses_public_submission_limit(monkeypatch) -> None:
    def unexpected_reservation(_client_ip: str):
        raise AssertionError("benchmark requests must not reserve public quota")

    monkeypatch.setattr(api, "reserve_submission", unexpected_reservation)
    client = TestClient(api.create_web_app(enforce_submission_limits=False))

    response = client.post("/transcriptions")

    assert response.status_code == 400
