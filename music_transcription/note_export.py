"""Stable note events and performance-timed MIDI, independent of ML runtimes."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

PROGRAM_GROUPS = {
    "acoustic_piano": [0, 1, 3, 6, 7],
    "electric_piano": [2, 4, 5],
    "chromatic_percussion": list(range(8, 16)),
    "organ": list(range(16, 24)),
    "acoustic_guitar": [24, 25],
    "clean_electric_guitar": [26, 27, 28],
    "distorted_electric_guitar": [29, 30, 31],
    "acoustic_bass": [32, 35],
    "electric_bass": [33, 34, 36, 37, 38, 39],
    "violin": [40],
    "viola": [41],
    "cello": [42],
    "contrabass": [43],
    "orchestral_harp": [46],
    "timpani": [47],
    "string_ensemble": [48, 49, 44, 45],
    "synth_strings": [50, 51],
    "voice": [52, 53, 54],
    "orchestra_hit": [55],
    "trumpet": [56, 59],
    "trombone": [57],
    "tuba": [58],
    "french_horn": [60],
    "brass_section": [61, 62, 63],
    "soprano_and_alto_sax": [64, 65],
    "tenor_sax": [66],
    "baritone_sax": [67],
    "oboe": [68],
    "english_horn": [69],
    "bassoon": [70],
    "clarinet": [71],
    "flutes": list(range(72, 80)),
    "synth_lead": list(range(80, 88)),
    "synth_pad": list(range(88, 96)),
}

PROGRAM_GROUPS.update(singing_voice=[100], singing_voice_chorus=[101], drums=[128])
PROGRAM_NAMES = {program: name for name, programs in PROGRAM_GROUPS.items() for program in programs}


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    program: int
    instrument: str
    velocity: int = 100


def normalize_notes(raw_notes, audio_seconds, instruments=None):
    """Retain predicted labels and timing, excluding padding and invalid events."""
    notes = []
    discarded = filtered = 0
    for raw in raw_notes:
        program = 128 if raw.is_drum else int(raw.program)
        start = float(raw.onset)
        end = start + 0.01 if raw.is_drum else float(raw.offset)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= raw.pitch <= 127
            or program not in PROGRAM_NAMES
            or end <= start
            or end <= 0
            or start >= audio_seconds
        ):
            discarded += 1
            continue
        name = PROGRAM_NAMES[program]
        if instruments and name not in instruments:
            filtered += 1
            continue
        notes.append(Note(max(0.0, start), min(audio_seconds, end), int(raw.pitch), program, name))
    return sorted(notes, key=lambda n: (n.start, n.program, n.pitch, n.end)), {
        "discarded_notes": discarded,
        "filtered_notes": filtered,
    }


def serialize_notes(notes):
    events = []
    for index, note in enumerate(notes):
        events.extend(
            [
                {
                    "type": "note_start",
                    "index": index,
                    "pitch": note.pitch,
                    "instrument": note.instrument,
                    "time": note.start,
                },
                {"type": "note_end", "index": index, "time": note.end},
            ]
        )
    return sorted(events, key=lambda e: (e["time"], e["type"] == "note_start", e["index"]))


def midi_bytes(notes, beat_grid=None):
    """Export tempo/meter without shifting or quantizing performance timestamps.

    A time-signature event starts at the first detected downbeat, so the audio
    clock remains the MIDI clock. Overlapping unisons use independent tracks and
    channels. MIDI ports preserve channel isolation beyond 15 pitched voices.
    """
    import mido

    bpm = float(beat_grid["bpm"]) if beat_grid else 120.0
    tempo = mido.bpm2tempo(bpm)
    midi = mido.MidiFile(type=1, ticks_per_beat=1000)
    header = mido.MidiTrack()
    midi.tracks.append(header)
    header.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    if beat_grid and beat_grid.get("beats_per_bar"):
        first = max(0.0, float(beat_grid["first_downbeat"]))
        header.append(
            mido.MetaMessage(
                "time_signature",
                numerator=int(beat_grid["beats_per_bar"]),
                denominator=4,
                time=round(mido.second2tick(first, midi.ticks_per_beat, tempo)),
            )
        )
    header.append(mido.MetaMessage("end_of_track", time=0))
    channels = [i for i in range(16) if i != 9]
    pitched_track = 0
    for program in sorted({note.program for note in notes}):
        lanes = []
        for note in sorted((n for n in notes if n.program == program), key=lambda n: n.start):
            for lane, active in lanes:
                if active.get(note.pitch, -1) <= note.start:
                    break
            else:
                lane, active = [], {}
                lanes.append((lane, active))
            lane.append(note)
            active[note.pitch] = note.end
        for lane_index, (lane, _) in enumerate(lanes):
            if program == 128:
                channel, port = 9, lane_index
            else:
                channel, port = channels[pitched_track % 15], pitched_track // 15
                pitched_track += 1
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("track_name", name=lane[0].instrument, time=0))
            track.append(mido.MetaMessage("midi_port", port=port, time=0))
            if program in (100, 101):
                track.append(mido.MetaMessage("text", text=f"yourmt3_program={program}", time=0))
            track.append(
                mido.Message(
                    "program_change",
                    program={100: 53, 101: 52}.get(program, program % 128),
                    channel=channel,
                )
            )
            events = []
            for note in lane:
                start = round(mido.second2tick(note.start, midi.ticks_per_beat, tempo))
                end = max(start + 1, round(mido.second2tick(note.end, midi.ticks_per_beat, tempo)))
                events.extend([(start, 1, note), (end, 0, note)])
            previous = 0
            for tick, on, note in sorted(events, key=lambda e: (e[0], e[1], e[2].pitch)):
                track.append(
                    mido.Message(
                        "note_on" if on else "note_off",
                        note=note.pitch,
                        velocity=note.velocity if on else 0,
                        channel=channel,
                        time=tick - previous,
                    )
                )
                previous = tick
            track.append(mido.MetaMessage("end_of_track", time=0))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()
