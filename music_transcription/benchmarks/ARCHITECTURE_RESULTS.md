# Architecture results — 2026-09-06

Recommend integrating **YourMT3+ MoE as the general backend candidate** and **TransKun
for confirmed piano**, retaining MuScriptor as a fallback during integration. The
benefit is substantially larger than the previous beam-search experiments. Production
settings remain unchanged.

## Official-test subset: 8 Slakh + 8 MAESTRO

| Model | Note F1 | Note + duration F1 | Seconds / clip |
| --- | ---: | ---: | ---: |
| MuScriptor raw | 67.78 | 30.19 | 29.53 |
| MuScriptor app export | 67.68 | 30.52 | 32.50 |
| MuScriptor beam 4 raw | 67.85 | 29.43 | 70.82 |
| YourMT3+ T5 | 83.12 | 45.23 | 4.28 |
| YourMT3+ MoE, preserved labels | **86.83** | **49.05** | **4.13** |
| Basic Pitch, CPU | 55.85 | 13.02 | 0.64 |

MoE gains **19.05 percentage points** in note F1; paired 95% interval **[12.67, 27.77]**.
It wins on note F1 in all 16 clips, and improves instrument-aware F1 from 52.28 to
84.62. Measured processing is about 7.2 times faster than the raw baseline. These are
loaded-model timings, not production end-to-end latency.

On Slakh alone, MoE raises note F1 from 56.10 to 76.67, duration F1 from 21.57 to
52.61, and instrument-aware note F1 from 25.10 to 72.26. All eight clips improve in
these measures and in sounding-pitch F1.

## Piano: 8 MAESTRO test excerpts

| Model | Note F1 | Note + duration F1 | Seconds / clip |
| --- | ---: | ---: | ---: |
| MuScriptor raw | 79.45 | 38.81 | 26.71 |
| YourMT3+ MoE | 96.99 | 45.49 | 4.49 |
| TransKun v2 | **98.94** | **87.75** | **2.51** |

TransKun improves note F1 and duration F1 on all eight clips. Its duration gain over
MuScriptor is 48.95 points, paired 95% interval [39.49, 57.65]. It matches the reference
key-release convention. MoE's duration score regresses on four piano clips despite
better note attacks, so a universal per-recording Pareto improvement is not established.

## Export finding and scope

The YourMT3 demo collapses instruments for playback. Preserving the MoE model's
predicted labels raises instrument F1 on Slakh from 59.41 to 72.26 and on exploratory
URMP from 16.45 to 78.22. Both exports use the same decoded notes; all 64 repeated demo
MIDI files exactly match the initial runs. The T5 checkpoint's coarser predictions
cannot recover fine labels simply by changing the writer.

All-32 note F1: MuScriptor 67.39, T5 85.59, MoE 86.96, Basic Pitch 64.66. MoE wins
note F1 on all 32. These are exploratory aggregate numbers: six of eight URMP clips
overlap YourMT3's published training split, and GuitarSet split membership is unverified.

MoE improves all six available aggregate quality measures and time on the official
test subset. This is measured sample-mean dominance, not proof across every song,
instrument or production constraint. Before default replacement, validate long clips
and chunk joins, vocals and real commercial mixtures, beat/export integration, and
interactive product behavior. An explicit piano mode is justified; automatic routing
requires its own evaluation. See `ARCHITECTURES.md` for reproduction and limitations.
