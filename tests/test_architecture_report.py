from music_transcription.benchmarks.report_architectures import aggregate_architectures
from music_transcription.benchmarks.report_quality_suite import METRICS


def row(example, dataset, model, score=1.0):
    return {
        "example": example,
        "dataset": dataset,
        "model": model,
        "seconds": 1.0,
        "metrics": {m: {"f1": score, "tp": 1, "fp": 0, "fn": 0} for m in METRICS},
    }


def test_specialist_cannot_rank_against_a_different_recording_set():
    examples = [{"id": "p", "dataset": "MAESTRO"}, {"id": "s", "dataset": "Slakh"}]
    rows = [
        row(e["id"], e["dataset"], m)
        for e in examples
        for m in ["muscriptor_raw", "muscriptor_product"]
    ]
    rows.append(row("p", "MAESTRO", "transkun"))
    result = aggregate_architectures(rows, examples)["muscriptor_raw"]
    assert "transkun" not in result["Official test 16"]
    assert "transkun" not in result["All 32"]
    assert result["MAESTRO"]["transkun"]["onset"]["n"] == 1
    assert result["URMP"] == {}


def test_pitch_only_instrument_metric_stays_unavailable():
    examples = [{"id": "p", "dataset": "MAESTRO"}]
    rows = [row("p", "MAESTRO", m) for m in ["muscriptor_raw", "muscriptor_product", "basic_pitch"]]
    rows[-1]["metrics"]["instrument_onset"] = None
    result = aggregate_architectures(rows, examples)["muscriptor_raw"]["MAESTRO"]["basic_pitch"]
    assert result["onset"]["mean"] == 1.0
    assert result["instrument_onset"] is None


def test_duplicate_clip_cannot_stand_in_for_a_missing_clip():
    examples = [{"id": "p", "dataset": "MAESTRO"}, {"id": "s", "dataset": "Slakh"}]
    rows = [
        row(e["id"], e["dataset"], m)
        for e in examples
        for m in ["muscriptor_raw", "muscriptor_product"]
    ]
    rows += [row("p", "MAESTRO", "yourmt3_moe")] * 2
    result = aggregate_architectures(rows, examples)["muscriptor_raw"]
    assert "yourmt3_moe" not in result["Official test 16"]
