import io
from types import SimpleNamespace

import mido
import pretty_midi
import pytest

from music_transcription.api import pair_note_events
from music_transcription.config import INSTRUMENT_NAMES
from music_transcription.note_export import (
    PROGRAM_NAMES,
    Note,
    midi_bytes,
    normalize_notes,
    serialize_notes,
)


def test_export_preserves_fine_programs_unisons_and_performance_clock():
    notes = [
        Note(0.231, 1.117, 60, 40, "violin"),
        Note(0.55, 1.49, 60, 40, "violin"),
        Note(0.4, 0.7, 48, 42, "cello"),
        Note(0.9, 0.91, 36, 128, "drums"),
    ]
    data = midi_bytes(notes, {"bpm": 137.2, "beats_per_bar": 3, "first_downbeat": 0.43})
    parsed = pretty_midi.PrettyMIDI(io.BytesIO(data))
    actual = sorted(
        (128 if i.is_drum else i.program, n.pitch, n.start, n.end)
        for i in parsed.instruments
        for n in i.notes
    )
    expected = sorted((n.program, n.pitch, n.start, n.end) for n in notes)
    for a, e in zip(actual, expected, strict=True):
        assert a == pytest.approx(e, abs=0.0005)
    paired = pair_note_events(serialize_notes(notes))
    assert len(paired) == len(notes)
    assert sorted(n["start"] for n in paired) == sorted(n.start for n in notes)
    assert {i.program for i in parsed.instruments if not i.is_drum} == {40, 42}


def test_filter_is_output_selection_and_padding_is_removed():
    def raw(program=40, start=0.1, end=0.9, drum=False):
        return SimpleNamespace(program=program, onset=start, offset=end, pitch=60, is_drum=drum)

    notes, counts = normalize_notes(
        [
            raw(),
            raw(42),
            raw(100),
            raw(start=2, end=3),
            raw(start=float("nan")),
        ],
        1,
        ["violin"],
    )
    assert [n.instrument for n in notes] == ["violin"]
    assert counts == {"filtered_notes": 2, "discarded_notes": 2}
    assert set(PROGRAM_NAMES.values()) <= set(INSTRUMENT_NAMES)
    all_notes, _ = normalize_notes([raw(100), raw(101), raw(drum=True)], 1)
    assert {n.instrument for n in all_notes} == {"singing_voice", "singing_voice_chorus", "drums"}
    assert next(n for n in all_notes if n.program == 128).end == pytest.approx(0.11)


def test_empty_midi_and_more_than_fifteen_instruments():
    assert len(mido.MidiFile(file=io.BytesIO(midi_bytes([]))).tracks) == 1
    notes = [Note(0.1, 0.5, 60, p, PROGRAM_NAMES[p]) for p in range(20)]
    midi = mido.MidiFile(file=io.BytesIO(midi_bytes(notes)))
    identities = []
    for track in midi.tracks[1:]:
        port = next(m.port for m in track if m.type == "midi_port")
        channel = next(m.channel for m in track if m.type == "program_change")
        assert channel != 9
        identities.append((port, channel))
    assert len(set(identities)) == 20


def test_singing_labels_survive_with_general_midi_voice_patches():
    notes = [
        Note(0.1, 0.5, 60, 100, "singing_voice"),
        Note(0.2, 0.6, 64, 101, "singing_voice_chorus"),
    ]
    midi = mido.MidiFile(file=io.BytesIO(midi_bytes(notes)))
    programs = [m.program for t in midi.tracks for m in t if m.type == "program_change"]
    assert programs == [53, 52]
    tags = [m.text for t in midi.tracks for m in t if m.type == "text"]
    assert tags == ["yourmt3_program=100", "yourmt3_program=101"]
