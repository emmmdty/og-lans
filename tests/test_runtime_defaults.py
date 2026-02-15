import yaml

from oglans.config import ConfigManager


def test_runtime_defaults_are_applied(tmp_path):
    config_path = tmp_path / "config_minimal.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 3407},
                "algorithms": {
                    "lans": {"enabled": True},
                    "scv": {"enabled": True},
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = ConfigManager.load_config(str(config_path))

    assert loaded["algorithms"]["lans"]["refresh_start_epoch"] == 1
    assert loaded["algorithms"]["lans"]["refresh_log_interval"] == 200
    assert loaded["algorithms"]["lans"]["refresh_log_seconds"] == 30
    assert loaded["algorithms"]["scv"]["progress_log_interval"] == 200
    assert loaded["algorithms"]["scv"]["progress_log_seconds"] == 30
    assert loaded["training"]["fast_io"] is False
    assert loaded["training"]["dataloader_num_workers"] == 0
    assert loaded["training"]["dataloader_pin_memory"] is False
