from __future__ import annotations

from pathlib import Path

from music_transcription.score import render_command


def test_render_command_uses_musescore_cli_output_mode() -> None:
    assert render_command(Path("input.mid"), Path("score.pdf")) == [
        "xvfb-run",
        "-a",
        "/opt/musescore/AppRun",
        "-o",
        "score.pdf",
        "input.mid",
    ]


def test_empty_transcription_skips_pdf_without_failing_job(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from music_transcription import score

    paths = {k: k for k in ["midi", "score_pdf", "metrics"]}
    (tmp_path / "metrics").write_text(json.dumps({"note_count": 0, "artifacts": {"midi": "midi"}}))
    monkeypatch.setattr(score, "job_paths", lambda *_: paths)
    monkeypatch.setattr(score, "mounted_artifact_path", lambda path: tmp_path / path)
    monkeypatch.setattr(
        score, "artifact_volume", SimpleNamespace(reload=lambda: None, commit=lambda: None)
    )
    monkeypatch.setattr(
        score,
        "render_score_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("Must not render an empty score")),
    )
    result = score.render_score.local("unused", ".wav")
    assert "No notes" in result["score"]["skipped_reason"]
    assert "score_pdf" not in result["artifacts"]


def test_pdf_validation_rejects_corrupt_and_blank_documents(tmp_path):
    import pytest
    from pypdf import PdfWriter
    from pypdf.errors import PdfReadError

    from music_transcription.score import validate_score_pdf

    path = tmp_path / "score.pdf"
    path.write_bytes(b"%PDF definitely not a complete document")
    with pytest.raises(PdfReadError):
        validate_score_pdf(path)
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.write(path)
    with pytest.raises(ValueError, match="blank"):
        validate_score_pdf(path)


def test_render_failure_preserves_existing_pdf(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from music_transcription import score

    source = tmp_path / "input.mid"
    source.write_bytes(b"MThd")
    output = tmp_path / "output.pdf"
    output.write_bytes(b"previous valid artifact")
    monkeypatch.setattr(
        score.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="bad MIDI", stdout=""),
    )
    with pytest.raises(RuntimeError, match="bad MIDI"):
        score.render_score_file(source, output)
    assert output.read_bytes() == b"previous valid artifact"
    assert not output.with_suffix(".pdf.partial").exists()


def test_optional_pdf_failure_keeps_successful_transcription(monkeypatch):
    from types import SimpleNamespace

    from music_transcription import preprocess

    updates = []
    result = {"note_count": 3, "artifacts": {"midi": "saved.mid"}}

    def fail(*_args):
        raise RuntimeError("engraving failed")

    monkeypatch.setattr(preprocess, "render_score", SimpleNamespace(remote=fail))
    monkeypatch.setattr(
        preprocess, "update_job", lambda *args, **kwargs: updates.append((args, kwargs))
    )
    actual = preprocess.finish_job(
        {"job_id": "test", "source_suffix": ".wav", "generate_score": True}, result
    )
    assert actual is result
    assert updates[-1][0] == ("test", "completed")
    assert updates[-1][1]["score_state"] == "failed"
    assert updates[-1][1]["result"]["artifacts"]["midi"] == "saved.mid"
