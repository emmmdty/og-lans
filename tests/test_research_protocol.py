import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "oglans"
    / "utils"
    / "research_protocol.py"
)
spec = importlib.util.spec_from_file_location("research_protocol", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load research_protocol from {MODULE_PATH}")
research_protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research_protocol)


def _sample(sample_id: str, event_types: list[str]):
    return SimpleNamespace(id=sample_id, event_types=event_types, text=f"text-{sample_id}")


def test_split_research_samples_is_deterministic_and_disjoint():
    samples = [
        _sample("a", ["企业收购"]),
        _sample("b", ["企业收购"]),
        _sample("c", ["中标"]),
        _sample("d", ["中标"]),
        _sample("e", ["股份回购"]),
        _sample("f", ["股份回购"]),
        _sample("g", ["被约谈"]),
        _sample("h", ["被约谈"]),
    ]

    fit_a, tune_a, manifest_a = research_protocol.split_research_samples(
        samples,
        tune_ratio=0.25,
        seed=3407,
    )
    fit_b, tune_b, manifest_b = research_protocol.split_research_samples(
        samples,
        tune_ratio=0.25,
        seed=3407,
    )

    fit_ids_a = {sample.id for sample in fit_a}
    tune_ids_a = {sample.id for sample in tune_a}
    fit_ids_b = {sample.id for sample in fit_b}
    tune_ids_b = {sample.id for sample in tune_b}

    assert fit_ids_a == fit_ids_b
    assert tune_ids_a == tune_ids_b
    assert fit_ids_a.isdisjoint(tune_ids_a)
    assert manifest_a == manifest_b
    assert manifest_a["fit_count"] == len(fit_ids_a)
    assert manifest_a["tune_count"] == len(tune_ids_a)


def test_select_fewshot_pool_samples_defaults_to_train_fit():
    samples = [
        _sample("a", ["企业收购"]),
        _sample("b", ["企业收购"]),
        _sample("c", ["中标"]),
        _sample("d", ["中标"]),
    ]

    selected, manifest = research_protocol.select_fewshot_pool_samples(
        samples,
        pool_split="train_fit",
        tune_ratio=0.25,
        seed=3407,
    )

    selected_ids = {sample.id for sample in selected}
    tune_ids = set(manifest["tune_ids"])

    assert selected_ids
    assert selected_ids.isdisjoint(tune_ids)
    assert manifest["pool_split"] == "train_fit"


def test_select_fewshot_pool_samples_can_use_frozen_manifest(tmp_path: Path):
    samples = [
        _sample("a", ["企业收购"]),
        _sample("b", ["企业收购"]),
        _sample("c", ["中标"]),
        _sample("d", ["中标"]),
    ]
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": 3407,
                "tune_ratio": 0.25,
                "fit_ids": ["a", "c"],
                "tune_ids": ["b", "d"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected, manifest = research_protocol.select_fewshot_pool_samples(
        samples,
        pool_split="train_fit",
        split_manifest=manifest_path,
    )

    assert {sample.id for sample in selected} == {"a", "c"}
    assert manifest["fit_count"] == 2
    assert manifest["manifest_path"] == str(manifest_path)


def test_restrict_schema_to_event_types_uses_full_schema_on_empty_predictions():
    schema = {
        "企业收购": ["收购方", "被收购方"],
        "中标": ["中标公司", "中标金额"],
    }

    resolved_schema, selected_event_types = research_protocol.restrict_schema_to_event_types(
        schema,
        predicted_event_types=[],
    )

    assert resolved_schema == schema
    assert selected_event_types == ["企业收购", "中标"]


def test_restrict_schema_to_event_types_filters_and_preserves_order():
    schema = {
        "企业收购": ["收购方", "被收购方"],
        "中标": ["中标公司", "中标金额"],
        "股份回购": ["回购方"],
    }

    resolved_schema, selected_event_types = research_protocol.restrict_schema_to_event_types(
        schema,
        predicted_event_types=["股份回购", "企业收购", "未知类型"],
    )

    assert list(resolved_schema.keys()) == ["企业收购", "股份回购"]
    assert selected_event_types == ["企业收购", "股份回购"]
