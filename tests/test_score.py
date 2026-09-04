from __future__ import annotations

from pathlib import Path

from music_transcription.score import render_command


def test_render_command_uses_musescore_cli_output_mode() -> None:
    assert render_command(Path("input.mid"), Path("score.pdf")) == [
        "mscore3",
        "-o",
        "score.pdf",
        "input.mid",
    ]
