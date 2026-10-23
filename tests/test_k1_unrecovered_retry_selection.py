import json

from tools.booster_k1_retarget.retry_unrecovered_references import (
    collect_targets,
    score_review,
)


def fixture(tmp_path):
    folder = tmp_path / "batch_original_0000"
    review = folder / "attempt_000/review"
    (review / "80_short_or_source_review").mkdir(parents=True)
    rows = [
        {
            "motion_key": k,
            "frames": 200,
            "candidate_frames": 0,
            "source_error": None,
            "native_leg_direction_p95_deg": 20,
            "reason_intervals": {},
        }
        for k in ["retry", "short", "missing", "whole"]
    ]
    rows[2]["source_error"] = "missing human source"
    rows[3]["candidate_frames"] = 200
    report = review / "report.json"
    report.write_text(json.dumps({"clips": rows}))
    short = {
        "length": 60,
        "metadata": {
            "parent_motion_key": "short",
            "parent_frame_start": 20,
            "parent_frame_end": 80,
        },
    }
    (review / "80_short_or_source_review/manifest.json").write_text(
        json.dumps({"clips": {"short": short}})
    )
    (folder / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "attempt": str(folder / "attempt_000"),
                "report": str(report),
            }
        )
    )
    return report, rows, short


def test_retry_only_zero_recovery_with_available_source(tmp_path):
    fixture(tmp_path)
    targets = collect_targets(tmp_path)
    assert [key for key, _, _ in targets] == ["retry"]


def test_score_counts_only_safe_short_intervals(tmp_path):
    path, rows, _ = fixture(tmp_path)
    row = rows[1]
    row["candidate_frames"] = 100
    path.write_text(json.dumps({"clips": [row]}))
    score, short = score_review(path)
    assert score == (160, -20.0) and short == 60
    row["reason_intervals"] = {"collision": [[50, 60]]}
    path.write_text(json.dumps({"clips": [row]}))
    score, short = score_review(path)
    assert score == (100, -20.0) and short == 0
