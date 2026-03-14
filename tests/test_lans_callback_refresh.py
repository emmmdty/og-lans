from types import SimpleNamespace

import pytest


try:
    from oglans.trainer.unsloth_trainer import (  # type: ignore
        CGADPOTrainer,
        LANSCallback,
        LANSNegativeSampler,
    )

    _LANS_CALLBACK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 仅在缺少训练依赖时触发
    LANSCallback = None  # type: ignore
    LANSNegativeSampler = None  # type: ignore
    CGADPOTrainer = None  # type: ignore
    _LANS_CALLBACK_IMPORT_ERROR = exc


class _DummyScheduler:
    def __init__(self):
        self.competence = 0.05
        self.current_threshold = 3.85
        self._recent_strategies = []

    def get_statistics(self):
        return {"strategy_distribution": {"EASY": 1.0, "MEDIUM": 0.0, "HARD": 0.0}}


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
