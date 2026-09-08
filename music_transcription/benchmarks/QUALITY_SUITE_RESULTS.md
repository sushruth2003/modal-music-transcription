# Broad quality evaluation — measured results

Completed 32 recordings across four sources, with 192 scored model inferences and an audio-only refinement. All models use the same pinned MuScriptor Large checkpoint. No production defaults changed.

## Recommendation

Automatic beam 4 raised average note F1 from 67.0 to 69.0, but took 2.25× the processing time and split evenly: 14 wins, 14 losses, 4 ties. Its paired 95% interval spans −0.1 to +4.2 points. Keep it as a candidate for an optional comparison setting, particularly for ensembles. The study does not support a universal default swap or a quality gain without tradeoffs.

## App MIDI export

| Approach | Note F1 | Δ [95% interval] | Duration F1 | Instrument F1 | Mean seconds | Wins / ties / losses |
|---|---:|---:|---:|---:|---:|---:|
| Automatic · greedy | 67.03 | +0.00 [+0.00, +0.00] | 35.95 | 49.79 | 23.84 | 0 / 32 / 0 |
| Automatic · beam 2 | 66.68 | -0.35 [-2.18, +1.25] | 34.28 | 50.10 | 43.51 | 15 / 4 / 13 |
| Automatic · beam 4 | 69.02 | +1.99 [-0.13, +4.20] | 36.19 | 50.52 | 53.65 | 14 / 4 / 14 |
| Known instruments · greedy | 65.56 | -1.47 [-4.24, +1.22] | 36.59 | 56.45 | 26.40 | 13 / 6 / 13 |
| Known instruments · beam 2 | 65.36 | -1.67 [-4.05, +0.84] | 35.48 | 55.04 | 46.13 | 15 / 1 / 16 |
| Known instruments · beam 4 | 65.84 | -1.19 [-3.86, +1.46] | 35.94 | 55.30 | 59.90 | 15 / 2 / 15 |
| Automatic · audio refinement | 68.02 | +0.99 [-0.37, +2.37] | 34.79 | 50.68 | 23.87 | 13 / 3 / 16 |

F1 values are percentages, averaged equally over 32 recordings. Δ is percentage points relative to automatic greedy. Source-stratified paired bootstrap: 5,000 recording resamples, seed 20260905. Intervals are exploratory and not adjusted for multiple comparisons.

## Source-specific findings

**Beam search helps one category.** Automatic beam 4 gained 7.9 note-F1 points on the eight ensemble clips. Changes on bands, guitar and piano were −0.2, −0.2 and +0.4 points. Most of the average gain comes from the ensemble subset.

**Instrument hints trade notes for labels.** Supplying instruments raised instrument-aware F1 from 49.8 to 56.4, while note F1 fell from 67.0 to 65.6. Keep manual instrument selection available with a preview; a better track label does not guarantee better notes.

**Hold the timing-refinement experiment.** The audio-only adjustment gained 1.0 note-F1 point overall, but reduced note-and-duration F1 by 1.2 points and regressed on 16 recordings. Its current form is not ready for a global default.

## Artifacts and limits

The interactive report is generated at `outputs/quality-suite-20260905/index.html`, with source audio, reference/predicted MIDI, `analysis.json`, `per-recording.csv`, `manifest.json`, and the raw run record alongside it. See [reproduction instructions](QUALITY_SUITE.md).

Input audio: 926.097 seconds. Reference notes in excerpts: 11,317. Completed-result inference: 126.35 GPU minutes, approximately $1.68. An initial interruption logged another approximately $0.11 of completed partial-clip work; cancelled in-flight work and startup/CPU/idle overhead are excluded. These are estimates, not billing.

The app-export view includes the real MIDI serializer and beat processing; only its documented bar padding is removed for scoring. Raw notes are an ablation. Velocity, pedal controls, pitch bends, PDF engraving, real singing, live drums, noisy commercial mixes and long-form consistency are not evaluated. Piano/Slakh offsets use key release without pedal extension. Model training overlap is unverified. No listener panel was conducted.

All 60 tests passed; formatting/lint and browser checks passed. Full source provenance, artifact checksums and interruption accounting remain local with the report.
