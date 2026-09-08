"""Audio-only, bounded onset refinement; no reference labels or fitted parameters."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, stft

from music_transcription.benchmarks.quality import Note


def refine_onsets(audio: np.ndarray, sr: int, notes: list[Note]) -> tuple[list[Note], dict]:
    """Move confident pitched attacks by at most 80 ms, preserving note duration.

    Positive spectral flux around the first three harmonics supplies candidates.
    The constants are fixed before suite scoring. Sustained, weak, and percussive
    notes without a clear pitch-specific attack retain the model's timing.
    """
    if audio.ndim != 1 or sr != 16000:
        raise ValueError("Expected mono 16kHz audio")
    if not notes or np.max(np.abs(audio), initial=0) < 1e-6:
        return list(notes), {"moved_notes": 0, "max_shift_ms": 0.0, "mean_abs_shift_ms": 0.0}
    frequencies, times, spectrum = stft(audio, fs=sr, nperseg=512, noverlap=432, boundary="zeros")
    magnitude = np.abs(spectrum)
    flux = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0)
    curves = {}
    candidates = {}
    for pitch in {n.pitch for n in notes if n.program != 128}:
        hz = 440 * 2 ** ((pitch - 69) / 12)
        curve = np.zeros(len(times))
        for harmonic in (1, 2, 3):
            index = int(np.argmin(abs(frequencies - hz * harmonic)))
            curve += flux[max(0, index - 1) : index + 2].sum(axis=0) / harmonic
        curve = gaussian_filter1d(curve, 0.8)
        median = np.median(curve)
        mad = np.median(abs(curve - median))
        threshold = max(median + 3 * mad, 0.04 * float(curve.max()))
        peaks, _ = find_peaks(curve, height=threshold, distance=6, prominence=threshold / 2)
        curves[pitch] = curve
        candidates[pitch] = peaks
    adjusted = []
    shifts = []
    duration = len(audio) / sr
    for n in notes:
        if n.program == 128:
            adjusted.append(n)
            continue
        peaks = candidates[n.pitch]
        nearby = peaks[abs(times[peaks] - n.start) <= 0.08]
        if len(nearby):
            # Prefer nearby peaks, so a larger neighboring note does not capture this one.
            strength = curves[n.pitch][nearby]
            utility = strength * (1 - 0.5 * abs(times[nearby] - n.start) / 0.08)
            target = float(times[nearby[int(np.argmax(utility))]])
            previous = max(
                (
                    m.start
                    for m in notes
                    if m.program == n.program and m.pitch == n.pitch and m.start < n.start
                ),
                default=-1.0,
            )
            following = min(
                (
                    m.start
                    for m in notes
                    if m.program == n.program and m.pitch == n.pitch and m.start > n.start
                ),
                default=duration + 1.0,
            )
            if previous < target < following and abs(target - n.start) >= 0.01:
                delta = target - n.start
                end = min(duration, n.end + delta)
                if end > target:
                    adjusted.append(replace(n, start=target, end=end))
                    shifts.append(delta)
                    continue
        adjusted.append(n)
    return adjusted, {
        "moved_notes": len(shifts),
        "max_shift_ms": max([abs(s) * 1000 for s in shifts], default=0.0),
        "mean_abs_shift_ms": float(np.mean(np.abs(shifts)) * 1000) if shifts else 0.0,
    }
