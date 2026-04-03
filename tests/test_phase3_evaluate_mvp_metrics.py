import importlib
import json


def test_compute_postprocess_metric_summary_reports_grounding_and_scv_lite():
    module = importlib.import_module("oglans.utils.json_parser")

    diagnostics_rows = [
        {
            "scv_lite_triggered": False,
            "argument_diagnostics": [
                {"grounding_status": "exact"},
                {"grounding_status": "normalized_exact"},
            ],
        },
        {
            "scv_lite_triggered": True,
            "argument_diagnostics": [
                {"grounding_status": "ungrounded"},
            ],
        },
    ]

    metrics = module.compute_postprocess_metric_summary(
        diagnostics_rows,
        scv_call_count=0,
        scv_total_seconds=0.0,
        total_runtime_seconds=12.5,
    )

    assert metrics["grounding_rate"] == 0.6667
    assert metrics["ungrounded_argument_rate"] == 0.3333
    assert metrics["scv_lite_trigger_count"] == 1
    assert metrics["scv_lite_triggered_samples"] == 1
    assert metrics["scv_call_count"] == 0
    assert metrics["scv_wall_clock_ratio"] == "NA"


def test_write_postprocess_diagnostics_sidecar_outputs_jsonl(tmp_path):
    module = importlib.import_module("oglans.utils.json_parser")

    sidecar_path = tmp_path / "eval_results_dev_seed3047_diagnostics.jsonl"
    rows = [
        {
            "id": "sample-1",
            "scv_lite_triggered": True,
            "argument_diagnostics": [{"grounding_status": "ungrounded"}],
        }
    ]

    written_path = module.write_postprocess_diagnostics_sidecar(sidecar_path, rows)

    assert str(sidecar_path) == written_path
    payload = sidecar_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload) == 1
    assert json.loads(payload[0])["id"] == "sample-1"
