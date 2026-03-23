from types import SimpleNamespace

import pytest
import torch


try:
    from oglans.trainer.unsloth_trainer import (  # type: ignore
        CGADPOTrainer,
        LANSCallback,
        LANSIterableDataset,
        LANSNegativeSampler,
        derive_online_iterable_max_steps,
    )

    _LANS_CALLBACK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 仅在缺少训练依赖时触发
    LANSCallback = None  # type: ignore
    LANSIterableDataset = None  # type: ignore
    LANSNegativeSampler = None  # type: ignore
    CGADPOTrainer = None  # type: ignore
    derive_online_iterable_max_steps = None  # type: ignore
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
