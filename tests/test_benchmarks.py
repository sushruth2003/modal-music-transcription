from __future__ import annotations

import argparse

import pytest

from music_transcription.benchmarks.m2_load import parse_concurrency, percentile, summarize


def test_parse_concurrency_preserves_order_and_removes_duplicates() -> None:
    assert parse_concurrency("1, 5,5, 10") == [1, 5, 10]

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_concurrency("1,0")


def test_percentile_interpolates_small_samples() -> None:
    assert percentile([], 0.95) is None
    assert percentile([2.0], 0.95) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_summarize_groups_status_latency_and_cost() -> None:
    records = [
        {
            "concurrency": 2,
            "submit_status": 202,
            "terminal_state": "completed",
            "metrics": {
                "submit_seconds": 0.1,
                "end_to_end_seconds": 10.0,
                "inference_seconds": 4.0,
                "estimated_gpu_cost_usd": 0.001,
            },
        },
        {
            "concurrency": 2,
            "submit_status": 202,
            "terminal_state": "failed",
            "metrics": {"submit_seconds": 0.3, "end_to_end_seconds": 12.0},
        },
    ]

    assert summarize(records) == [
        {
            "concurrency": 2,
            "attempted": 2,
            "accepted": 2,
            "completed": 1,
            "failed": 1,
            "timed_out": 0,
            "submit_statuses": {"202": 2},
            "submit_seconds": {"p50": 0.2, "p95": 0.29, "max": 0.3},
            "end_to_end_seconds": {"p50": 11.0, "p95": 11.9, "max": 12.0},
            "inference_seconds_p50": 4.0,
            "estimated_inference_gpu_cost_usd": 0.001,
        }
    ]
