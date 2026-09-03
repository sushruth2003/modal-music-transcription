from __future__ import annotations

import pytest

from music_transcription import storage


def test_new_job_spec_and_paths(monkeypatch) -> None:
    class FixedUuid:
        hex = "a" * 32

    monkeypatch.setattr(storage, "uuid4", lambda: FixedUuid())
    spec = storage.new_job_spec("../demo.MP3", ["piano"])

    assert spec == {
        "job_id": "a" * 32,
        "source_name": "demo.MP3",
        "source_suffix": ".mp3",
        "instruments": ["piano"],
    }
    assert storage.job_paths(spec["job_id"], spec["source_suffix"])["source"] == (
        f"jobs/{'a' * 32}/source.mp3"
    )


@pytest.mark.parametrize("job_id", ["short", "A" * 32, "../" + "a" * 29])
def test_validate_job_id_rejects_unsafe_values(job_id: str) -> None:
    with pytest.raises(ValueError, match="job_id"):
        storage.validate_job_id(job_id)


def test_mounted_artifact_path_rejects_escape() -> None:
    with pytest.raises(ValueError, match="safe and relative"):
        storage.mounted_artifact_path("jobs/../outside")


def test_update_job_preserves_record(monkeypatch) -> None:
    job_id = "b" * 32
    spec = {
        "job_id": job_id,
        "source_name": "demo.wav",
        "source_suffix": ".wav",
        "instruments": None,
    }
    fake_dict = {job_id: storage.initial_job_record(spec)}
    monkeypatch.setattr(storage, "job_states", fake_dict)

    updated = storage.update_job(job_id, "preprocessing")

    assert updated["state"] == "preprocessing"
    assert updated["paths"]["midi"].endswith("/transcription.mid")
