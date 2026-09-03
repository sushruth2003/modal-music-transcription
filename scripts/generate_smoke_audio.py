"""Generate a deterministic, rights-safe chord progression for smoke testing."""

from __future__ import annotations

import argparse
import math
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16_000
DURATION_SECONDS = 30
CHORD_SECONDS = 2.5
CHORDS = (
    (48, (60, 64, 67)),  # C
    (43, (59, 62, 67)),  # G
    (45, (57, 60, 64)),  # Am
    (41, (57, 60, 65)),  # F
)


def frequency(midi_pitch: int) -> float:
    return 440.0 * 2 ** ((midi_pitch - 69) / 12)


def synthesize() -> array:
    """Return signed 16-bit mono samples with bass, chords, and a soft pulse."""

    samples = array("h")
    for sample_index in range(SAMPLE_RATE * DURATION_SECONDS):
        time_seconds = sample_index / SAMPLE_RATE
        chord_index = int(time_seconds / CHORD_SECONDS) % len(CHORDS)
        bass_pitch, chord_pitches = CHORDS[chord_index]
        chord_time = time_seconds % CHORD_SECONDS

        attack = min(chord_time / 0.04, 1.0)
        release = min((CHORD_SECONDS - chord_time) / 0.20, 1.0)
        envelope = max(0.0, min(attack, release))

        signal = 0.22 * math.sin(2 * math.pi * frequency(bass_pitch) * time_seconds)
        for pitch in chord_pitches:
            fundamental = math.sin(2 * math.pi * frequency(pitch) * time_seconds)
            harmonic = math.sin(4 * math.pi * frequency(pitch) * time_seconds)
            signal += envelope * (0.12 * fundamental + 0.025 * harmonic)

        pulse_time = time_seconds % 0.5
        pulse = math.exp(-70 * pulse_time) * (
            0.10 * math.sin(2 * math.pi * 90 * time_seconds)
            + 0.025 * math.sin(2 * math.pi * 3_200 * time_seconds)
        )
        signal = max(-0.95, min(0.95, signal + pulse))
        samples.append(round(signal * 32_767))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic-chords-30s.wav"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(args.output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(synthesize().tobytes())

    print(args.output.resolve())


if __name__ == "__main__":
    main()
