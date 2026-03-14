import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "run_manifest.py"
spec = importlib.util.spec_from_file_location("run_manifest", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load run_manifest from {MODULE_PATH}")
run_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_manifest)


build_run_manifest = run_manifest.build_run_manifest
compute_file_sha256 = run_manifest.compute_file_sha256
compute_json_sha256 = run_manifest.compute_json_sha256
save_json = run_manifest.save_json


def test_compute_json_sha256_is_stable_for_same_payload():
    payload = {"b": 2, "a": 1}
    h1 = compute_json_sha256(payload)
    h2 = compute_json_sha256({"a": 1, "b": 2})
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_save_json_and_compute_file_sha256(tmp_path):
    target = tmp_path / "manifest.json"
    payload = {"task": "eval_local", "status": "completed"}
    save_json(target, payload)

    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload

    digest = compute_file_sha256(target)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_build_run_manifest_shape():
    payload = build_run_manifest(
        task="train",
        status="completed",
        meta={"run_id": "abc"},
        artifacts={"output_dir": "/tmp/out"},
        runtime={"wall_clock_seconds": 1.23},
        runtime_manifest={"python": {"version": "3.10"}},
    )
    assert payload["task"] == "train"
    assert payload["status"] == "completed"
    assert payload["meta"]["run_id"] == "abc"
    assert payload["artifacts"]["output_dir"] == "/tmp/out"
    assert payload["runtime"]["wall_clock_seconds"] == 1.23
    assert payload["runtime_manifest"]["python"]["version"] == "3.10"
