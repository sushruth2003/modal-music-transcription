from __future__ import annotations

import pytest

from music_transcription.transcribe import corrected_event_time


def test_corrected_event_time_removes_positive_model_lag() -> None:
    assert corrected_event_time(1.5, 0.025) == pytest.approx(1.475)


def test_corrected_event_time_preserves_signed_correction() -> None:
    assert corrected_event_time(1.5, -0.01) == pytest.approx(1.51)
