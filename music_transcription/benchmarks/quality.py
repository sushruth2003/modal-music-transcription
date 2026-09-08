"""Offline, performance-aligned MIDI scoring and conservative two-pass consensus.

Install the eval dependency group. This module never invokes Modal or downloads data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

# MuScriptor 0.3.0 MT3_FULL_PLUS groups; first program is the decoded representative.
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


def canonical_program(program: int) -> int:
    for programs in PROGRAM_GROUPS.values():
        if program in programs:
            return programs[0]
    return program


def program_for_name(name: str) -> int:
    if name == "drums":
        return 128
    if name.startswith("program_"):
        return int(name.removeprefix("program_"))
    return PROGRAM_GROUPS[name][0]


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    program: int = 0
    velocity: int = 80

    def __post_init__(self) -> None:
        if not 0 <= self.pitch <= 127 or not 0 <= self.program <= 128:
            raise ValueError("Invalid MIDI pitch or program")
        if not np.isfinite([self.start, self.end]).all() or not 0 <= self.start < self.end:
            raise ValueError("Note intervals must be finite, positive, and nonnegative")


def read_midi(path: str | Path) -> list[Note]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return sorted(
        [
            Note(
                n.pitch,
                n.start,
                n.end,
                128 if i.is_drum else canonical_program(i.program),
                n.velocity,
            )
            for i in midi.instruments
            for n in i.notes
            if n.end > n.start
        ],
        key=lambda n: (n.start, n.program, n.pitch, n.end),
    )


def write_midi(notes: list[Note], path: str | Path) -> None:
    """Write performance seconds, preserving overlapping unisons on separate tracks.

    Multiple note-ons of the same pitch on one MIDI track cannot reliably retain
    independent release times. Allocate another track when that pitch overlaps.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=1000)
    for program in sorted({n.program for n in notes}):
        tracks = []
        active_until = []
        for note in sorted((n for n in notes if n.program == program), key=lambda n: n.start):
            lane = next(
                (i for i, ends in enumerate(active_until) if ends.get(note.pitch, 0) <= note.start),
                len(tracks),
            )
            if lane == len(tracks):
                tracks.append(pretty_midi.Instrument(program=program % 128, is_drum=program == 128))
                active_until.append({})
            tracks[lane].notes.append(
                pretty_midi.Note(note.velocity, note.pitch, note.start, note.end)
            )
            active_until[lane][note.pitch] = note.end
        midi.instruments.extend(tracks)
    midi.write(str(path))


def arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[n.start, n.end] for n in notes], dtype=float).reshape(-1, 2),
        pretty_midi.note_number_to_hz(np.array([n.pitch for n in notes], dtype=float)),
    )


def matches(
    reference: list[Note],
    estimate: list[Note],
    *,
    offsets: bool = False,
    instruments: bool = False,
    tolerance: float = 0.05,
) -> list[tuple[int, int]]:
    result = []
    groups = sorted({n.program for n in reference + estimate}) if instruments else [None]
    for group in groups:
        ri = [i for i, n in enumerate(reference) if group is None or n.program == group]
        ei = [i for i, n in enumerate(estimate) if group is None or n.program == group]
        if not ri or not ei:
            continue
        r = arrays([reference[i] for i in ri])
        e = arrays([estimate[i] for i in ei])
        pairs = mir_eval.transcription.match_notes(
            *r,
            *e,
            onset_tolerance=tolerance,
            pitch_tolerance=50.0,
            offset_ratio=0.2 if offsets and group != 128 else None,
            offset_min_tolerance=0.05,
            strict=False,
        )
        result.extend((ri[a], ei[b]) for a, b in pairs)
    return sorted(result)


def counts(tp: int, n_reference: int, n_estimate: int) -> dict[str, float | int]:
    precision = tp / n_estimate if n_estimate else 0.0
    recall = tp / n_reference if n_reference else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (n_reference + n_estimate) if n_reference + n_estimate else 0.0,
        "tp": tp,
        "fp": n_estimate - tp,
        "fn": n_reference - tp,
    }


def evaluate(
    reference: list[Note], estimate: list[Note], *, start: float, end: float, audio_end: float
) -> dict:
    """Score onsets inside a fixed window; never align predictions to ground truth.

    Notes crossing the audio end have censored offsets at that end. Onsets before
    the window do not count toward note metrics. Frame metrics retain their sustain.
    Drums are evaluated separately (onset only), not mixed into pitched-note F1.
    """
    if not 0 <= start < end <= audio_end:
        raise ValueError("Invalid evaluation window")
    ref = [
        replace(n, end=min(n.end, audio_end))
        for n in reference
        if start <= n.start < end and n.program != 128
    ]
    est = [
        replace(n, end=min(n.end, audio_end))
        for n in estimate
        if start <= n.start < end and n.program != 128
    ]
    metrics = {}
    for key, offsets, instruments in [
        ("onset", False, False),
        ("onset_offset", True, False),
        ("instrument_onset", False, True),
        ("instrument_onset_offset", True, True),
    ]:
        pairs = matches(ref, est, offsets=offsets, instruments=instruments)
        metrics[key] = counts(len(pairs), len(ref), len(est))
    pairs = matches(ref, est)
    errors = [est[b].start - ref[a].start for a, b in pairs]
    metrics["matched_onset_mae_ms"] = float(np.mean(np.abs(errors)) * 1000) if errors else None
    metrics["matched_onset_signed_ms"] = float(np.mean(errors) * 1000) if errors else None
    metrics["reference_notes"] = len(ref)
    metrics["estimated_notes"] = len(est)
    # Diagnostic only: keep the standard 50 ms score as the headline metric.
    metrics["onset_tolerance_curve"] = {
        str(ms): counts(len(matches(ref, est, tolerance=ms / 1000)), len(ref), len(est))
        for ms in (25, 50, 100, 200)
    }
    confusion = {}
    for a, b in pairs:
        key = f"{ref[a].program}->{est[b].program}"
        confusion[key] = confusion.get(key, 0) + 1
    metrics["instrument_confusion_on_matched_notes"] = confusion
    metrics["per_instrument"] = {
        str(p): counts(
            len(matches([n for n in ref if n.program == p], [n for n in est if n.program == p])),
            sum(n.program == p for n in ref),
            sum(n.program == p for n in est),
        )
        for p in sorted({n.program for n in ref + est})
    }
    # Fixed 10 ms frame grid, instrument-agnostic pitched activity.
    times = np.arange(start, end, 0.01)

    def roll(notes: list[Note]) -> np.ndarray:
        result = np.zeros((len(times), 128), dtype=bool)
        for n in notes:
            if n.program != 128:
                result[(times >= n.start) & (times < n.end), n.pitch] = True
        return result

    rr, er = roll(reference), roll(estimate)
    metrics["frame"] = counts(int((rr & er).sum()), int(rr.sum()), int(er.sum()))
    dr = [n for n in reference if n.program == 128 and start <= n.start < end]
    de = [n for n in estimate if n.program == 128 and start <= n.start < end]
    metrics["drum_onset"] = counts(len(matches(dr, de)), len(dr), len(de))
    return metrics


def consensus(first: list[Note], second: list[Note], tolerance: float = 0.08) -> list[Note]:
    """Keep one-to-one, instrument/pitch/onset-agreeing notes; average timing.

    This intentionally favors precision over recall. No reference annotations are used.
    """
    return [
        replace(
            first[a],
            start=(first[a].start + second[b].start) / 2,
            end=(first[a].end + second[b].end) / 2,
        )
        for a, b in matches(first, second, instruments=True, tolerance=tolerance)
    ]


def write_notes(notes: list[Note], path: Path) -> None:
    path.write_text(json.dumps([asdict(n) for n in notes], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--audio-end", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                read_midi(args.reference),
                read_midi(args.estimate),
                start=args.start,
                end=args.end,
                audio_end=args.audio_end or args.end,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
