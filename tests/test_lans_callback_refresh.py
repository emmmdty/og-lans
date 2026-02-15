from types import SimpleNamespace

import pytest


try:
    from oglans.trainer.unsloth_trainer import LANSCallback  # type: ignore

    _LANS_CALLBACK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 仅在缺少训练依赖时触发
    LANSCallback = None  # type: ignore
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
