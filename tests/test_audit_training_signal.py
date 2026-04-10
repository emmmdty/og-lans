import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_signal.py"
spec = importlib.util.spec_from_file_location("audit_training_signal", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_audit_training_signal_explains_filtered_row_count_gap(tmp_path: Path):
    dataset_root = tmp_path / "logs" / "DuEE-Fin"
    checkpoint_dir = dataset_root / "checkpoints" / "exp1"
    samples_dir = dataset_root / "samples" / "exp1"
    checkpoint_dir.mkdir(parents=True)
    samples_dir.mkdir(parents=True)

    manifest = {
        "meta": {
            "exp_name": "exp1",
            "training_mode": "preference",
            "stage_mode": "single_pass",
            "prompt_variant": "zeroshot",
            "configured_train_count": 10,
            "effective_train_count": 8,
            "effective_lans_enabled": True,
            "effective_scv_enabled": True,
        },
        "runtime": {
            "wall_clock_seconds": 123.0,
        },
    }
    (checkpoint_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = {
        "total_generated": 4,
        "scv_filter_event_count": 3,
        "scv_filtered_count": 2,
        "scv_filter_event_rate": 0.75,
        "scv_filter_rate": 0.5,
        "retry_exhausted_count": 0,
        "strategy_distribution": {"EASY": 1, "MEDIUM": 2, "HARD": 1},
        "post_scv_strategy_distribution": {"EASY": 2, "MEDIUM": 2, "HARD": 0},
    }
    (samples_dir / "lans_sampling_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _write_jsonl(samples_dir / "lans_generated_samples.jsonl", [{"sample_id": i} for i in range(4)])
    _write_jsonl(
        samples_dir / "scv_filtered_samples.jsonl",
        [
            {"sample_id": 1, "strategy": "MEDIUM", "attempt": 0, "reason": "SCV detected false negative"},
            {"sample_id": 1, "strategy": "EASY", "attempt": 1, "reason": "SCV detected false negative"},
            {"sample_id": 2, "strategy": "HARD", "attempt": 0, "reason": "SCV detected false negative"},
        ],
    )

    audit = mod.audit_training_signal(checkpoint_dir / "run_manifest.json")

    assert audit["generated_rows"] == 4
    assert audit["filtered_event_audit"]["row_count"] == 3
    assert audit["filtered_event_audit"]["unique_sample_ids"] == 2
    assert audit["filtered_event_audit"]["avg_filter_events_per_filtered_sample"] == 1.5
    assert audit["explanations"]
