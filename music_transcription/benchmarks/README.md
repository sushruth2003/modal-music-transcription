# M2 benchmark runner

This directory contains development tooling, not code used by the deployed web,
CPU-processing, or GPU-inference functions.

`m2_load.py` submits synchronized groups of audio jobs, polls every accepted job
to a terminal state, and saves both per-job observations and aggregate latency
statistics as JSON. It never sends traffic unless `--execute` is present.

Preview the default 1/5/10/25-client, three-repeat plan:

```bash
uv run python -m music_transcription.benchmarks.m2_load \
  --base-url https://example--music-transcription-benchmark.modal.run
```

The protected benchmark endpoint is registered by `deployment.py`. It uses the
same web settings and processing/GPU functions as the public endpoint, but skips
the public demo quota because Modal rejects requests at its proxy unless they
carry a valid proxy token.

Run the benchmark against that protected endpoint:

```bash
export BENCH_MODAL_KEY='...'
export BENCH_MODAL_SECRET='...'

uv run python -m music_transcription.benchmarks.m2_load \
  --base-url https://example--music-transcription-benchmark.modal.run \
  --header-env Modal-Key=BENCH_MODAL_KEY \
  --header-env Modal-Secret=BENCH_MODAL_SECRET \
  --concurrency 1,5,10,25 \
  --repeats 3 \
  --execute
```

Header values are read from environment variables and are never written into the
report. `--preflight` checks `/api/health` first, but it also deliberately warms
the web container; omit it when measuring a cold web request. `--round-delay 130`
can separate rounds long enough for the current GPU scale-down window, although
that makes the run substantially slower.

The full matrix creates 123 transcription jobs. Do not point it at the public
demo: that deployment intentionally allows only three submissions per IP per hour
and ten submissions globally per day. Create a protected benchmark deployment
with a deliberate cost cap first.

Reports are written under `music_transcription/benchmarks/results/` and ignored by
Git. Each report includes:

- submit status and latency;
- end-to-end completion latency;
- terminal state and polling count;
- server creation and update timestamps;
- GPU type and container identity, including distinct-container counts per level;
- preprocessing, model-load, and inference measurements returned by the app;
- estimated inference-only L4 cost;
- p50, p95, and maximum latency grouped by requested concurrency.
