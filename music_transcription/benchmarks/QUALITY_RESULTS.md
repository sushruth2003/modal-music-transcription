# Initial quality results — 2026-09-05

Completed 15 MuScriptor Large inference calls on one NVIDIA L4, plus three derived
two-pass consensus outputs. The tested checkpoint is
`8809fdfbed2affa7ade94a7059e746e3880720e7`, using `muscriptor==0.3.0`, fp16, CFG 1,
and prelude forcing. Reproduction instructions are in [QUALITY.md](QUALITY.md).

## Data and protocol

Downloaded real WAV recordings, score MIDIs, and official performance annotations
for URMP 08 Spring (flute/violin), 31 Slavonic (brass quartet), and 40 Miserere
(woodwind quintet). Each input is the first 30 seconds; onsets within 2.5–27.5
seconds are scored, totaling 607 reference notes across the three excerpts.

The score MIDIs use a score clock. The actual reference MIDIs are constructed from
performance-aligned note annotations, preserving overlapping unisons on separate
tracks. No reference pitches or timings are supplied to inference or used to
optimize alignment. Instrument-conditioned variants receive the correct instrument
list from recording metadata.

## Results

F1 percentages are macro averages, weighting each recording equally. The primary
note criterion is correct pitch and onset within 50 ms. Duration adds an offset
tolerance of max(50 ms, 20% reference duration). Instrument metrics additionally
require the correct MT3_FULL_PLUS instrument group.

| Variant | Note F1 | Note+duration F1 | Instrument+note F1 | Instrument+duration F1 | Mean inference seconds |
|---|---:|---:|---:|---:|---:|
| Baseline greedy | 48.05 | 34.98 | 16.99 | 11.72 | 15.97 |
| Known instruments | 49.16 | 31.10 | 35.49 | 24.47 | 15.16 |
| Known instruments + beam 2 | 53.02 | 34.98 | 36.93 | 26.07 | 31.11 |
| Known instruments + beam 4 | 56.33 | 36.22 | 39.06 | 25.93 | 44.33 |
| Known instruments + shifted chunks | 43.10 | 31.26 | 20.65 | 15.16 | 14.20 |
| Instrument-aware two-pass consensus | 25.60 | 19.54 | 17.13 | 13.18 | 29.36 |

## Findings

1. Beam 4 improves average note F1 by 8.28 percentage points and instrument-aware
   note F1 by 22.07 points versus the unconditioned greedy baseline, at about 2.78×
   inference time. Relative to conditioned greedy, its note gain is 7.17 points.
   Beam 2 is a less expensive intermediate point.
2. Correct instrument conditioning improves instrument assignments substantially,
   but is not a strict Pareto improvement: instrument-agnostic duration F1 falls.
   These timings are single observations, not proof of a speed advantage.
3. The woodwind excerpt regresses in note F1: baseline 47.66 versus beam 4 39.32,
   even while instrument-aware note F1 improves from 0 to 26.50. Baseline labels
   the notes as organ; beam 4 produces actual woodwind groups. This illustrates why
   both instrument-agnostic and instrument-aware metrics matter.
4. Shifted chunks are not a dependable upgrade. On the brass quartet the shifted
   pass assigns all notes to trombone, while conditioned greedy uses trumpet and
   horn. Their strict instrument-aware intersection is empty, discarding all notes.
   Do not enable this consensus rule in production.
5. Timing is a substantial source of strict note mismatches. Relaxing the onset
   tolerance from 50 to 100 ms raises baseline macro F1 from 48.05 to 78.84 and
   beam 4 from 56.33 to 84.19. This is diagnostic only; headline scores stay at
   50 ms. Audio-grounded onset refinement is a reasonable next experiment.

No production defaults were changed. The evidence supports further evaluation of
optional beam search with known instrumentation, not a universal automatic upgrade.

## Verification and artifacts

- 56 tests pass, including 10 evaluator/consensus/MIDI correctness tests.
- Formatting, lint, and diff whitespace checks pass.
- All 15 GPU calls completed without reported model warnings.
- Total synchronized inference time: 362.30 seconds; model load: 4.63 seconds.
- Estimated inference-only cost: $0.0804 at the historical project rate. This is
  not the actual Modal bill and excludes startup, idle, CPU, and image-build costs.
- [Modal run](https://modal.com/apps/sushruthb03/main/ap-GxgxADCSRvGGa19GDHm6tc).
- Local ignored outputs: `outputs/quality-urmp-20260905/`, containing `REPORT.md`,
  `results.json`, `per-recording.csv`, `manifest.json`, and per-recording WAV,
  reference MIDI, every predicted MIDI, and raw note JSON.
- Initial scoring snapshots are retained. The final scoring corrects a MIDI
  serialization issue with overlapping unisons; raw inference predictions did
  not change. Final scores above use the corrected references.

## Limits

This is a small exploratory classical ensemble set, with unknown overlap with
MuScriptor training data. It does not establish generalization to produced music,
drums, vocals, or distorted guitars. Ground-truth instrument lists are oracle
metadata. Every variant excludes the app's beat-grid onset correction and MIDI bar
shift to isolate decoding; the results do not measure full web-app output. Notes
crossing the excerpt end have censored offsets. Runtime measurements are one run
per example/configuration, and the first call can include kernel warm-up effects.

## Dataset attribution

Bochen Li, Xinzhao Liu, Karthik Dinesh, Zhiyao Duan, and Gaurav Sharma, “Creating a
multi-track classical music performance dataset for multi-modal music analysis:
Challenges, insights, and applications,” IEEE Transactions on Multimedia, 2018.

- [Official URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html)
- [Performance annotation format](https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf)
- [Pinned public mirror](https://huggingface.co/datasets/Eredis02/URMP/tree/58177a0f0f816e621b3d5d304c5fd1a03c035a86)

Exact source URLs and SHA-256 checksums are retained in the run manifest. Downloaded
audio, score MIDI, and annotations remain outside version control.
