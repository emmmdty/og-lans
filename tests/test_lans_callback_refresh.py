import copy
from types import SimpleNamespace

import pytest
import torch


try:
    from oglans.trainer.unsloth_trainer import (  # type: ignore
        CGADPOTrainer,
        LANSCallback,
        LANSDataCollator,
        LANSIterableDataset,
        LANSNegativeSampler,
        build_explicit_dpo_record,
        derive_online_iterable_max_steps,
    )
    import oglans.trainer.unsloth_trainer as unsloth_trainer_module  # type: ignore

    _LANS_CALLBACK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 仅在缺少训练依赖时触发
    LANSCallback = None  # type: ignore
    LANSDataCollator = None  # type: ignore
    LANSIterableDataset = None  # type: ignore
    LANSNegativeSampler = None  # type: ignore
    CGADPOTrainer = None  # type: ignore
    build_explicit_dpo_record = None  # type: ignore
    derive_online_iterable_max_steps = None  # type: ignore
    unsloth_trainer_module = None  # type: ignore
    _LANS_CALLBACK_IMPORT_ERROR = exc


class _DummyScheduler:
    def __init__(self):
        self.competence = 0.05
        self.current_threshold = 3.85
        self._recent_strategies = []
        self.last_loss = None

    def update_competence(self, loss):
        self.last_loss = loss
        self.competence = min(0.95, max(0.05, float(loss)))
        return self.competence

    def get_statistics(self):
        return {
            "competence": self.competence,
            "threshold": self.current_threshold,
            "recent_avg_loss": self.last_loss or 0.0,
            "strategy_distribution": {"EASY": 1.0, "MEDIUM": 0.0, "HARD": 0.0},
        }


class _DummySampler:
    def __init__(self):
        self.epoch = None
        self.generate_calls = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def generate_rejected(self, sample):
        self.generate_calls += 1
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": f"neg_{self.generate_calls}",
        }


class _DummyDSCNS:
    def __init__(self):
        self._use_lans = False

    def generate_negative_json(self, chosen, strategy, current_step, total_steps):
        return '[{"event_type":"质押","arguments":[{"role":"质押方","argument":"张三"}]}]'


class _DummySCV:
    def __init__(self, false_negative=False):
        self.false_negative = bool(false_negative)
        self.calls = 0

    def is_false_negative(self, text, neg_json):
        self.calls += 1
        return self.false_negative


class _DummyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        token_ids = [((ord(ch) - 31) % 97) + 1 for ch in text]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }


class _DummyPaddingCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = {}
        pad_keys = (
            "prompt_input_ids",
            "prompt_attention_mask",
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_labels",
        )

        for key in pad_keys:
            if key not in features[0]:
                continue
            pad_value = -100 if key.endswith("labels") else 0
            max_len = max(len(feature[key]) for feature in features)
            batch[key] = torch.tensor(
                [
                    feature[key] + [pad_value] * (max_len - len(feature[key]))
                    for feature in features
                ],
                dtype=torch.long,
            )

        for key in features[0]:
            if key not in batch:
                batch[key] = [feature[key] for feature in features]
        return batch


@pytest.mark.skipif(LANSCallback is None, reason="LANSCallback 依赖缺失")
def test_refresh_starts_from_configured_epoch():
    base_samples = [
        {"prompt": "p1", "chosen": "c1", "text": "t1", "event_types": []},
        {"prompt": "p2", "chosen": "c2", "text": "t2", "event_types": []},
    ]
    scheduler = _DummyScheduler()
    sampler = _DummySampler()
    trainer_ref = SimpleNamespace(train_dataset="original", _train_dataloader="cached", accelerator=None)

    cb = LANSCallback(
        lans_scheduler=scheduler,
        lans_sampler=sampler,
        trainer_ref=trainer_ref,
        base_samples=base_samples,
        refresh_start_epoch=1,
        refresh_log_interval=1,
    )

    # epoch 0: 跳过刷新
    cb.on_epoch_begin(
        args=SimpleNamespace(output_dir="."),
        state=SimpleNamespace(epoch=0),
        control=SimpleNamespace(),
    )
    assert sampler.generate_calls == 0
    assert trainer_ref.train_dataset == "original"

    # epoch 1: 开始刷新
    cb.on_epoch_begin(
        args=SimpleNamespace(output_dir="."),
        state=SimpleNamespace(epoch=1),
        control=SimpleNamespace(),
    )
    assert sampler.generate_calls == len(base_samples)
    assert trainer_ref.train_dataset != "original"


@pytest.mark.skipif(LANSNegativeSampler is None, reason="LANSNegativeSampler 依赖缺失")
def test_negative_sampler_scv_cache_reduces_repeated_calls():
    ds = _DummyDSCNS()
    scv = _DummySCV(false_negative=False)
    sampler = LANSNegativeSampler(
        ds_cns=ds,
        scv=scv,
        scv_cache_enabled=True,
        scv_cache_max_entries=16,
        scv_max_retries=0,
    )

    example = {"prompt": "p", "chosen": "[]", "text": "t", "event_types": []}
    sampler.generate_rejected(example)
    sampler.generate_rejected(example)

    assert scv.calls == 1, "相同文本+负样本应命中缓存，避免重复 SCV 调用"
    stats = sampler.get_statistics()
    assert stats["scv_cache_hits"] >= 1
    assert stats["scv_cache_misses"] == 1


@pytest.mark.skipif(LANSNegativeSampler is None, reason="LANSNegativeSampler 依赖缺失")
def test_negative_sampler_scv_rates_are_sample_bounded_even_with_retries():
    ds = _DummyDSCNS()
    scv = _DummySCV(false_negative=True)
    sampler = LANSNegativeSampler(
        ds_cns=ds,
        scv=scv,
        scv_cache_enabled=False,
        scv_max_retries=1,
    )

    example = {"prompt": "p", "chosen": "[]", "text": "t", "event_types": []}
    sampler.generate_rejected(example)

    stats = sampler.get_statistics()
    assert stats["total_generated"] == 1
    assert stats["scv_rejected_sample_count"] == 1
    assert stats["scv_filter_event_count"] == 2
    assert stats["scv_filter_rate"] == pytest.approx(1.0)
    assert stats["scv_reject_rate"] == pytest.approx(1.0)
    assert stats["scv_filter_event_rate"] == pytest.approx(2.0)


@pytest.mark.skipif(CGADPOTrainer is None, reason="CGADPOTrainer 依赖缺失")
def test_rpo_weight_warmup_schedule():
    trainer = object.__new__(CGADPOTrainer)
    trainer.rpo_alpha = 0.2
    trainer.rpo_warmup_steps = 100
    trainer.state = SimpleNamespace(global_step=0)
    assert trainer._compute_rpo_weight() == pytest.approx(0.0)

    trainer.state.global_step = 50
    assert trainer._compute_rpo_weight() == pytest.approx(0.1)

    trainer.state.global_step = 200
    assert trainer._compute_rpo_weight() == pytest.approx(0.2)


@pytest.mark.skipif(LANSCallback is None, reason="LANSCallback 依赖缺失")
def test_on_step_end_uses_fresh_aux_loss_snapshot():
    scheduler = _DummyScheduler()
    trainer_ref = SimpleNamespace(
        get_aux_metrics_snapshot=lambda: {"pref_loss_raw": 0.123},
    )
    cb = LANSCallback(
        lans_scheduler=scheduler,
        trainer_ref=trainer_ref,
        refresh_log_interval=1,
    )

    cb.on_step_end(
        args=SimpleNamespace(output_dir="."),
        state=SimpleNamespace(max_steps=10, global_step=1, log_history=[{"loss": 9.9}]),
        control=SimpleNamespace(),
    )

    assert scheduler.last_loss == pytest.approx(0.123)


@pytest.mark.skipif(LANSCallback is None, reason="LANSCallback 依赖缺失")
def test_online_iterable_callback_skips_dataset_refresh():
    base_samples = [
        {"prompt": "p1", "chosen": "c1", "text": "t1", "event_types": []},
        {"prompt": "p2", "chosen": "c2", "text": "t2", "event_types": []},
    ]
    scheduler = _DummyScheduler()
    sampler = _DummySampler()
    original_dataset = object()
    trainer_ref = SimpleNamespace(
        train_dataset=original_dataset,
        _train_dataloader="cached",
        accelerator=None,
    )

    cb = LANSCallback(
        lans_scheduler=scheduler,
        lans_sampler=sampler,
        trainer_ref=trainer_ref,
        base_samples=base_samples,
        runtime_mode="online_iterable",
        refresh_log_interval=1,
    )

    cb.on_epoch_begin(
        args=SimpleNamespace(output_dir="."),
        state=SimpleNamespace(epoch=1),
        control=SimpleNamespace(),
    )

    assert sampler.epoch == 1
    assert sampler.generate_calls == 0
    assert trainer_ref.train_dataset is original_dataset
    assert trainer_ref._train_dataloader == "cached"


@pytest.mark.skipif(LANSIterableDataset is None, reason="LANSIterableDataset 依赖缺失")
def test_online_iterable_dataset_reports_length():
    dataset = LANSIterableDataset(
        base_samples=[
            {"prompt": "p1", "chosen": "c1", "text": "t1", "event_types": []},
            {"prompt": "p2", "chosen": "c2", "text": "t2", "event_types": []},
        ],
        lans_sampler=_DummySampler(),
        sample_builder=lambda sample, result: result,
        seed=3407,
    )

    assert len(dataset) == 2


@pytest.mark.skipif(
    derive_online_iterable_max_steps is None,
    reason="derive_online_iterable_max_steps 依赖缺失",
)
def test_online_iterable_max_steps_is_derived_from_length():
    derived = derive_online_iterable_max_steps(
        dataset_length=65,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
    )

    assert derived == 9


@pytest.mark.skipif(build_explicit_dpo_record is None, reason="显式 DPO record 构造依赖缺失")
def test_explicit_dpo_record_separates_prompt_and_completion():
    tokenizer = _DummyTokenizer()
    record = build_explicit_dpo_record(
        tokenizer=tokenizer,
        prompt="系统提示：",
        chosen='{"event_list":[]}',
        rejected='{"event_list":[1]}',
        max_prompt_length=64,
        max_length=128,
    )

    expected_prompt = tokenizer("系统提示：", add_special_tokens=False)["input_ids"]
    expected_chosen = tokenizer('{"event_list":[]}' + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    expected_rejected = tokenizer('{"event_list":[1]}' + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    assert record["prompt_input_ids"] == expected_prompt
    assert record["chosen_input_ids"] == expected_chosen
    assert record["rejected_input_ids"] == expected_rejected
    assert record["chosen_attention_mask"] == [1] * len(expected_chosen)
    assert record["rejected_attention_mask"] == [1] * len(expected_rejected)
    assert record["chosen_labels"] == expected_chosen
    assert record["rejected_labels"] == expected_rejected


@pytest.mark.skipif(LANSDataCollator is None, reason="LANSDataCollator 依赖缺失")
def test_online_iterable_collator_regenerates_rejected_and_preserves_chosen(monkeypatch):
    monkeypatch.setattr(
        unsloth_trainer_module,
        "DPODataCollatorWithPadding",
        _DummyPaddingCollator,
    )

    tokenizer = _DummyTokenizer()
    sampler = _DummySampler()
    collator = LANSDataCollator(
        tokenizer=tokenizer,
        lans_sampler=sampler,
        max_length=128,
    )

    base_feature = build_explicit_dpo_record(
        tokenizer=tokenizer,
        prompt="系统提示：",
        chosen='{"event_list":[]}',
        rejected="占位符",
        max_prompt_length=64,
        max_length=128,
    )
    base_feature.update(
        {
            "text": "原始文本",
            "event_types": ["质押"],
        }
    )
    chosen_input_ids = list(base_feature["chosen_input_ids"])
    chosen_labels = list(base_feature["chosen_labels"])

    batch_one = collator([copy.deepcopy(base_feature)])
    batch_two = collator([copy.deepcopy(base_feature)])

    assert sampler.generate_calls == 2
    assert torch.equal(batch_one["chosen_input_ids"][0], torch.tensor(chosen_input_ids))
    assert torch.equal(batch_two["chosen_input_ids"][0], torch.tensor(chosen_input_ids))
    assert torch.equal(batch_one["chosen_labels"][0], torch.tensor(chosen_labels))
    assert torch.equal(batch_two["chosen_labels"][0], torch.tensor(chosen_labels))
    assert not torch.equal(batch_one["rejected_input_ids"][0], batch_two["rejected_input_ids"][0])
    assert torch.equal(batch_one["rejected_input_ids"][0], batch_one["rejected_labels"][0])
    assert torch.equal(batch_two["rejected_input_ids"][0], batch_two["rejected_labels"][0])


@pytest.mark.skipif(CGADPOTrainer is None, reason="CGADPOTrainer 依赖缺失")
def test_compute_chosen_sft_loss_reconstructs_prompt_and_completion():
    class _SpyModel:
        def __init__(self):
            self.input_ids = None
            self.attention_mask = None
            self.labels = None

        def __call__(self, input_ids, attention_mask, labels, use_cache=False, return_dict=True):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
            return SimpleNamespace(loss=torch.tensor(0.5))

    model = _SpyModel()
    inputs = {
        "prompt_input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "chosen_input_ids": torch.tensor([[21, 22, 0]], dtype=torch.long),
        "chosen_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        "chosen_labels": torch.tensor([[21, 22, -100]], dtype=torch.long),
    }

    loss = CGADPOTrainer._compute_chosen_sft_loss(model, inputs)

    assert loss is not None
    assert torch.equal(model.input_ids, torch.tensor([[10, 11, 12, 21, 22, 0]], dtype=torch.long))
    assert torch.equal(model.attention_mask, torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long))
    assert torch.equal(model.labels, torch.tensor([[-100, -100, -100, 21, 22, -100]], dtype=torch.long))


@pytest.mark.skipif(CGADPOTrainer is None, reason="CGADPOTrainer 依赖缺失")
def test_rpo_missing_labels_fail_fast(monkeypatch):
    import trl

    monkeypatch.setattr(
        trl.DPOTrainer,
        "compute_loss",
        lambda self, model, inputs, return_outputs=False, num_items_in_batch=None: torch.tensor(0.2),
    )

    trainer = object.__new__(CGADPOTrainer)
    trainer.lans_scheduler = None
    trainer.rpo_alpha = 0.1
    trainer.rpo_warmup_steps = 0
    trainer.rpo_require_valid_labels = True
    trainer.preference_mode = "ipo"
    trainer.odpo_offset_source = "margin_bucket"
    trainer.odpo_offset_static = 0.15
    trainer.odpo_offset_clip = (0.0, 1.0)
    trainer.aux_log_interval = 50
    trainer._cga_applied_count = 0
    trainer._rpo_warning_emitted = False
    trainer._rpo_label_warning_emitted = False
    trainer._rpo_steps = 0
    trainer._rpo_nonzero_steps = 0
    trainer._rpo_missing_label_steps = 0
    trainer._recent_odpo_offsets = []
    trainer._last_aux_metrics = {}
    trainer.state = SimpleNamespace(global_step=1)

    with pytest.raises(ValueError, match="RPO enabled but chosen_labels has no valid completion tokens"):
        trainer.compute_loss(
            model=SimpleNamespace(),
            inputs={
                "chosen_input_ids": torch.ones((1, 4), dtype=torch.long),
                "chosen_attention_mask": torch.ones((1, 4), dtype=torch.long),
                "chosen_labels": torch.full((1, 4), -100, dtype=torch.long),
            },
        )
