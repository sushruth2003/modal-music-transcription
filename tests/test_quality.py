"""Scoring must punish real errors and never learn alignment from the reference."""

import pytest

pytest.importorskip("mir_eval", reason="Install the eval dependency group")
pytest.importorskip("pretty_midi", reason="Install the eval dependency group")

from music_transcription.benchmarks.quality import (
    Note,
    consensus,
    evaluate,
    read_midi,
    write_midi,
)


def score(ref, est):
    return evaluate(ref, est, start=0, end=4, audio_end=4)


def test_identical_midi_scores_perfectly_after_roundtrip(tmp_path):
    ref = [Note(60, 0.25, 1, 40), Note(64, 0.25, 1.25, 71), Note(60, 2, 3, 40)]
    path = tmp_path / "test.mid"
    write_midi(ref, path)
    result = score(ref, read_midi(path))
    for metric in ("onset", "onset_offset", "instrument_onset", "instrument_onset_offset"):
        assert result[metric]["f1"] == 1


def test_duplicate_predictions_are_false_positives():
    note = Note(60, 1, 2)
    result = score([note], [note, note])
    assert result["onset"]["tp"] == 1
    assert result["onset"]["fp"] == 1
    assert result["onset"]["recall"] == 1
    assert result["onset"]["precision"] == 0.5


def test_overlapping_unisons_keep_independent_release_times_in_midi(tmp_path):
    ref = [Note(60, 1, 2, 72), Note(60, 1.5, 3, 72)]
    path = tmp_path / "unisons.mid"
    write_midi(ref, path)
    recovered = read_midi(path)
    assert [(n.start, n.end) for n in recovered] == [(1, 2), (1.5, 3)]


def test_wrong_instrument_is_only_correct_for_instrument_agnostic_metric():
    result = score([Note(60, 1, 2, 40)], [Note(60, 1, 2, 71)])
    assert result["onset"]["f1"] == 1
    assert result["instrument_onset"]["f1"] == 0


def test_timing_and_offset_errors_are_not_automatically_aligned():
    ref = [Note(60, 1, 2)]
    assert score(ref, [Note(60, 1.1, 2.1)])["onset"]["f1"] == 0
    result = score(ref, [Note(60, 1.02, 2.5)])
    assert result["onset"]["f1"] == 1
    assert result["onset_offset"]["f1"] == 0


def test_missing_notes_and_empty_output():
    result = score([Note(60, 1, 2), Note(64, 1, 2)], [])
    assert result["onset"]["fn"] == 2
    assert result["onset"]["f1"] == 0
    assert result["frame"]["f1"] == 0


def test_consensus_does_not_reuse_one_note_for_repeated_attacks():
    first = [Note(60, 1, 1.05), Note(60, 1.1, 1.2), Note(64, 1, 2)]
    second = [Note(60, 1.04, 1.1), Note(67, 1, 2)]
    combined = consensus(first, second)
    assert len(combined) == 1
    assert combined[0].pitch == 60
    assert combined[0].start == pytest.approx(1.02)


def test_frame_sustain_crossing_evaluation_start_is_kept():
    notes = [Note(60, 0.5, 3)]
    result = evaluate(notes, notes, start=1, end=3.5, audio_end=4)
    assert result["reference_notes"] == 0
    assert result["frame"]["f1"] == 1


def test_drums_are_separate_and_do_not_require_matching_duration():
    result = score([Note(36, 1, 1.2, 128)], [Note(36, 1.01, 1.02, 128)])
    assert result["reference_notes"] == 0
    assert result["drum_onset"]["f1"] == 1


def test_invalid_notes_fail_instead_of_silently_disappearing():
    with pytest.raises(ValueError):
        Note(60, -1, 2)
    with pytest.raises(ValueError):
        Note(60, 1, float("nan"))


def test_audio_refinement_does_not_change_silence_or_drums():
    import numpy as np

    from music_transcription.benchmarks.quality_refinement import refine_onsets

    notes = [Note(60, 1, 2), Note(36, 1, 1.1, 128)]
    refined, stats = refine_onsets(np.zeros(48000, dtype=np.float32), 16000, notes)
    assert refined == notes
    assert stats["moved_notes"] == 0


def test_audio_refinement_moves_delayed_attack_within_fixed_bound():
    import numpy as np

    from music_transcription.benchmarks.quality_refinement import refine_onsets

    sr = 16000
    t = np.arange(sr * 3) / sr
    audio = np.sin(2 * np.pi * 440 * t) * ((t >= 1) & (t < 2))
    original = Note(69, 1.07, 1.9)
    refined, stats = refine_onsets(audio, sr, [original])
    assert len(refined) == 1
    assert abs(refined[0].start - 1) < abs(original.start - 1)
    assert stats["max_shift_ms"] <= 80.00001
    assert refined[0].end - refined[0].start == pytest.approx(original.end - original.start)
    assert refined[0].pitch == original.pitch


def test_product_scoring_removes_only_exported_bar_padding(tmp_path):
    import mido

    from music_transcription.benchmarks.report_quality_suite import product_notes

    path = tmp_path / "padded.mid"
    write_midi([Note(60, 1.6, 2.6)], path)
    midi = mido.MidiFile(path)
    midi.tracks[0].insert(0, mido.MetaMessage("marker", text="muscriptor:bar_offset=0.5", time=0))
    midi.save(path)
    notes = product_notes(path, 0.5)
    assert notes[0].start == pytest.approx(1.1)
    # A remaining 100 ms acoustic error must not be silently aligned away.
    assert score([Note(60, 1, 2)], notes)["onset"]["f1"] == 0
    with pytest.raises(ValueError):
        product_notes(path, 0.6)


def test_paired_bootstrap_preserves_constant_recording_differences():
    from music_transcription.benchmarks.report_quality_suite import bootstrap

    baseline = {}
    rows = []
    for i, base in enumerate([0.2, 0.4, 0.6, 0.8]):
        key = str(i)
        baseline[key] = {"metrics": {"onset": {"f1": base}}}
        rows.append(
            {
                "example": key,
                "dataset": "A" if i < 2 else "B",
                "seconds": 1,
                "metrics": {"onset": {"f1": base + 0.1, "tp": 1, "fp": 0, "fn": 0}},
            }
        )
    stats = bootstrap(rows, baseline, "onset", samples=100)
    assert stats["low"] == pytest.approx(0.1)
    assert stats["high"] == pytest.approx(0.1)
    assert stats["wins"] == 4
