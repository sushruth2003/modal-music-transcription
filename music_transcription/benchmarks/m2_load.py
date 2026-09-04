"""Run controlled M2 submission and completion benchmarks against a deployed API."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

TERMINAL_STATES = frozenset({"completed", "failed"})
DEFAULT_AUDIO = Path("data/synthetic-chords-30s.wav")
DEFAULT_RESULTS_DIRECTORY = Path(__file__).parent / "results"


def parse_concurrency(value: str) -> list[int]:
    """Parse a comma-separated, positive, de-duplicated concurrency matrix."""

    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("concurrency must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("concurrency values must be positive integers")
    return list(dict.fromkeys(values))


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for a small benchmark sample."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _metric_values(records: list[dict[str, Any]], name: str) -> list[float]:
    values = []
    for record in records:
        value = record.get("metrics", {}).get(name)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw job observations by requested concurrency."""

    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["concurrency"])].append(record)

    summaries = []
    for concurrency, group in sorted(grouped.items()):
        submit = _metric_values(group, "submit_seconds")
        end_to_end = _metric_values(group, "end_to_end_seconds")
        inference = _metric_values(group, "inference_seconds")
        estimated_cost = _metric_values(group, "estimated_gpu_cost_usd")
        states = Counter(str(record.get("terminal_state", "not_accepted")) for record in group)
        statuses = Counter(str(record.get("submit_status", "transport_error")) for record in group)
        gpu_container_ids = sorted(
            {
                str(record["worker"]["container_id"])
                for record in group
                if isinstance(record.get("worker"), dict) and record["worker"].get("container_id")
            }
        )
        summaries.append(
            {
                "concurrency": concurrency,
                "attempted": len(group),
                "accepted": sum(record.get("submit_status") == 202 for record in group),
                "completed": states["completed"],
                "failed": states["failed"],
                "timed_out": states["timed_out"],
                "submit_statuses": dict(sorted(statuses.items())),
                "submit_seconds": {
                    "p50": _rounded(percentile(submit, 0.50)),
                    "p95": _rounded(percentile(submit, 0.95)),
                    "max": _rounded(max(submit) if submit else None),
                },
                "end_to_end_seconds": {
                    "p50": _rounded(percentile(end_to_end, 0.50)),
                    "p95": _rounded(percentile(end_to_end, 0.95)),
                    "max": _rounded(max(end_to_end) if end_to_end else None),
                },
                "inference_seconds_p50": _rounded(percentile(inference, 0.50)),
                "estimated_inference_gpu_cost_usd": round(sum(estimated_cost), 6),
                "distinct_gpu_containers": len(gpu_container_ids),
                "gpu_container_ids": gpu_container_ids,
            }
        )
    return summaries


def load_header_environment(bindings: list[str]) -> tuple[dict[str, str], list[str]]:
    """Resolve HEADER=ENV_VAR bindings without putting secret values in arguments or output."""

    headers: dict[str, str] = {}
    for binding in bindings:
        header, separator, environment_name = binding.partition("=")
        if not separator or not header.strip() or not environment_name.strip():
            raise ValueError("--header-env must use HEADER=ENV_VAR")
        value = os.environ.get(environment_name.strip())
        if not value:
            raise ValueError(f"environment variable {environment_name.strip()!r} is not set")
        headers[header.strip()] = value
    return headers, sorted(headers)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {"detail": response.text[:1_000]}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _numeric(mapping: object, *path: str) -> float | None:
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _result_metrics(payload: dict[str, Any], end_to_end_seconds: float) -> dict[str, float]:
    metrics = {"end_to_end_seconds": end_to_end_seconds}
    paths = {
        "audio_seconds": ("result", "audio_seconds"),
        "preprocessing_seconds": ("result", "preprocessing", "seconds"),
        "model_load_seconds": ("result", "model", "load_seconds"),
        "inference_seconds": ("result", "inference", "seconds"),
        "real_time_factor": ("result", "inference", "real_time_factor"),
        "estimated_gpu_cost_usd": ("result", "inference", "estimated_gpu_cost_usd"),
    }
    for name, path in paths.items():
        value = _numeric(payload, *path)
        if value is not None:
            metrics[name] = value
    return metrics


async def run_job(
    client: httpx.AsyncClient,
    gate: asyncio.Event,
    *,
    base_url: str,
    audio_name: str,
    audio_bytes: bytes,
    media_type: str,
    instruments: str | None,
    concurrency: int,
    repeat: int,
    client_index: int,
    poll_interval: float,
    job_timeout: float,
) -> dict[str, Any]:
    """Submit one job at the shared start signal and observe it to a terminal state."""

    record: dict[str, Any] = {
        "concurrency": concurrency,
        "repeat": repeat,
        "client_index": client_index,
        "started_at": None,
        "submit_status": None,
        "terminal_state": "not_accepted",
        "metrics": {},
    }
    await gate.wait()
    record["started_at"] = _utc_now()
    started = time.perf_counter()
    try:
        response = await client.post(
            urljoin(f"{base_url}/", "transcriptions"),
            files={"audio": (audio_name, audio_bytes, media_type)},
            data={} if instruments is None else {"instruments": instruments},
        )
    except httpx.HTTPError as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["metrics"]["submit_seconds"] = time.perf_counter() - started
        return record

    record["metrics"]["submit_seconds"] = time.perf_counter() - started
    record["submit_status"] = response.status_code
    record["rate_limit"] = {
        "ip_remaining": response.headers.get("x-ratelimit-ip-remaining"),
        "global_remaining": response.headers.get("x-ratelimit-global-remaining"),
        "retry_after": response.headers.get("retry-after"),
    }
    submission = _safe_json(response)
    if response.status_code != 202:
        record["submission_response"] = submission
        return record

    job_id = submission.get("job_id")
    status_path = submission.get("status_url")
    if not isinstance(job_id, str) or not isinstance(status_path, str):
        record["error"] = "202 response did not include job_id and status_url"
        return record
    record["job_id"] = job_id
    record["function_call_id"] = submission.get("function_call_id")

    status_url = urljoin(f"{base_url}/", status_path.lstrip("/"))
    deadline = started + job_timeout
    polls = 0
    while time.perf_counter() < deadline:
        await asyncio.sleep(poll_interval)
        try:
            status_response = await client.get(status_url)
        except httpx.HTTPError as error:
            record["error"] = f"status {type(error).__name__}: {error}"
            continue
        polls += 1
        if status_response.status_code != 200:
            record["error"] = f"status endpoint returned {status_response.status_code}"
            continue
        payload = _safe_json(status_response)
        state = payload.get("state")
        if state not in TERMINAL_STATES:
            continue

        elapsed = time.perf_counter() - started
        record["terminal_state"] = state
        record["finished_at"] = _utc_now()
        record["polls"] = polls
        record["server_timestamps"] = {
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        result = payload.get("result")
        model = result.get("model") if isinstance(result, dict) else None
        if isinstance(model, dict):
            record["worker"] = {
                "container_id": model.get("container_id"),
                "gpu_name": model.get("gpu_name"),
            }
        record["metrics"].update(_result_metrics(payload, elapsed))
        if state == "failed":
            record["error"] = payload.get("error", "job failed without an error message")
        return record

    record["terminal_state"] = "timed_out"
    record["finished_at"] = _utc_now()
    record["polls"] = polls
    record["metrics"]["end_to_end_seconds"] = time.perf_counter() - started
    record["error"] = f"job did not finish within {job_timeout:g} seconds"
    return record


async def execute_benchmark(
    args: argparse.Namespace,
    headers: dict[str, str],
    audio_bytes: bytes,
) -> list[dict[str, Any]]:
    """Execute the requested matrix, one synchronized round at a time."""

    limits = httpx.Limits(
        max_connections=max(args.concurrency) + 5,
        max_keepalive_connections=max(args.concurrency) + 5,
    )
    timeout = httpx.Timeout(args.request_timeout)
    records: list[dict[str, Any]] = []
    media_type = mimetypes.guess_type(args.audio.name)[0] or "application/octet-stream"

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        if args.preflight:
            response = await client.get(urljoin(f"{args.base_url}/", "api/health"))
            response.raise_for_status()

        for concurrency in args.concurrency:
            for repeat in range(1, args.repeats + 1):
                print(f"starting concurrency={concurrency} repeat={repeat}", flush=True)
                gate = asyncio.Event()
                tasks = [
                    asyncio.create_task(
                        run_job(
                            client,
                            gate,
                            base_url=args.base_url,
                            audio_name=args.audio.name,
                            audio_bytes=audio_bytes,
                            media_type=media_type,
                            instruments=args.instruments,
                            concurrency=concurrency,
                            repeat=repeat,
                            client_index=index,
                            poll_interval=args.poll_interval,
                            job_timeout=args.job_timeout,
                        )
                    )
                    for index in range(1, concurrency + 1)
                ]
                await asyncio.sleep(0)
                gate.set()
                records.extend(await asyncio.gather(*tasks))
                if args.round_delay and (concurrency, repeat) != (
                    args.concurrency[-1],
                    args.repeats,
                ):
                    await asyncio.sleep(args.round_delay)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark M2 HTTP submission, job completion, and reported inference metrics."
    )
    parser.add_argument("--base-url", required=True, help="Deployed benchmark API URL")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument(
        "--concurrency", type=parse_concurrency, default=parse_concurrency("1,5,10,25")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--job-timeout", type=float, default=15 * 60.0)
    parser.add_argument("--round-delay", type=float, default=0.0)
    parser.add_argument("--instruments", help="Comma-separated instrument hints")
    parser.add_argument("--label", default="m2-load")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--header-env",
        action="append",
        default=[],
        metavar="HEADER=ENV_VAR",
        help="Read a request header value from an environment variable; repeat as needed",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check /api/health before the run; this deliberately warms the web container",
    )
    parser.add_argument("--max-jobs", type=int, default=200)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit billable jobs; without this flag the command only prints its plan",
    )
    return parser


def validate_args(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.poll_interval <= 0 or args.request_timeout <= 0 or args.job_timeout <= 0:
        raise ValueError("timeouts and polling interval must be positive")
    if args.round_delay < 0:
        raise ValueError("--round-delay cannot be negative")
    planned_jobs = sum(args.concurrency) * args.repeats
    if planned_jobs > args.max_jobs:
        raise ValueError(f"plan contains {planned_jobs} jobs, exceeding --max-jobs={args.max_jobs}")
    return planned_jobs


def _default_output(label: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in label
    )
    return DEFAULT_RESULTS_DIRECTORY / f"{safe_label}-{timestamp}.json"


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nconcurrency  accepted  completed  failed  submit p50  submit p95  e2e p50")
    for item in summary:
        print(
            f"{item['concurrency']:>11}  {item['accepted']:>8}  {item['completed']:>9}  "
            f"{item['failed']:>6}  {item['submit_seconds']['p50']!s:>10}  "
            f"{item['submit_seconds']['p95']!s:>10}  "
            f"{item['end_to_end_seconds']['p50']!s:>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    try:
        planned_jobs = validate_args(args)
        headers, header_names = load_header_environment(args.header_env)
    except ValueError as error:
        parser.error(str(error))

    plan = {
        "base_url": args.base_url,
        "audio": str(args.audio),
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "planned_jobs": planned_jobs,
        "poll_interval_seconds": args.poll_interval,
        "job_timeout_seconds": args.job_timeout,
        "round_delay_seconds": args.round_delay,
        "preflight_warms_web_container": args.preflight,
        "request_header_names": header_names,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("\nDry run only. Add --execute to submit billable jobs.")
        return 0

    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    audio_bytes = args.audio.read_bytes()
    if not audio_bytes:
        parser.error(f"audio file is empty: {args.audio}")

    started_at = _utc_now()
    started = time.perf_counter()
    try:
        records = asyncio.run(execute_benchmark(args, headers, audio_bytes))
    except (httpx.HTTPError, OSError) as error:
        print(f"benchmark stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    summary = summarize(records)
    output = args.output or _default_output(args.label)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "label": args.label,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "configuration": plan,
        "summary": summary,
        "jobs": records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"\nRaw report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
