import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_model_store.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("migrate_model_store", str(SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_model_dir(root: Path, identifier: str) -> Path:
    model_dir = root / identifier
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    return model_dir


def test_migrate_model_store_dry_run_reports_plan_without_writing(tmp_path):
    source_root = tmp_path / "legacy"
    _make_model_dir(source_root, "Qwen/Qwen3-4B-Instruct-2507")
    mod = _load_module()

    summary = mod.migrate_model_store(
        project_root=tmp_path,
        source_roots=[source_root],
        mode="copy",
        dry_run=True,
    )

    assert summary["models_root"] == str((tmp_path / "models").resolve())
    assert len(summary["migrations"]) == 1
    assert summary["migrations"][0]["identifier"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert not (tmp_path / "models").exists()


def test_migrate_model_store_copy_mode_copies_model_tree(tmp_path):
    source_root = tmp_path / "legacy"
    src_model = _make_model_dir(source_root, "Qwen/Qwen3-4B-Instruct-2507")
    mod = _load_module()

    summary = mod.migrate_model_store(
        project_root=tmp_path,
        source_roots=[source_root],
        mode="copy",
        dry_run=False,
    )

    dst_model = tmp_path / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"
    assert dst_model.exists()
    assert (dst_model / "config.json").read_text(encoding="utf-8") == "{}"
    assert src_model.exists()
    assert summary["migrated"] == 1


def test_migrate_model_store_rejects_conflicting_existing_target(tmp_path):
    source_root = tmp_path / "legacy"
    _make_model_dir(source_root, "Qwen/Qwen3-4B-Instruct-2507")
    target_root = tmp_path / "models" / "Qwen" / "Qwen3-4B-Instruct-2507"
    target_root.mkdir(parents=True)
    (target_root / "config.json").write_text('{"existing": true}', encoding="utf-8")
    mod = _load_module()

    with pytest.raises(FileExistsError, match="already exists"):
        mod.migrate_model_store(
            project_root=tmp_path,
            source_roots=[source_root],
            mode="copy",
            dry_run=False,
        )


def test_migrate_model_store_move_mode_can_link_legacy_source(tmp_path):
    source_root = tmp_path / "legacy"
    src_model = _make_model_dir(source_root, "Fengshenbang/Erlangshen-MegatronBert-1___3B-NLI")
    mod = _load_module()

    summary = mod.migrate_model_store(
        project_root=tmp_path,
        source_roots=[source_root],
        mode="move",
        dry_run=False,
        link_legacy=True,
    )

    dst_model = tmp_path / "models" / "Fengshenbang" / "Erlangshen-MegatronBert-1___3B-NLI"
    assert dst_model.exists()
    assert src_model.is_symlink()
    assert src_model.resolve() == dst_model.resolve()
    assert summary["linked_legacy"] == 1
