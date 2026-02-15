import pytest
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "config.py"
spec = importlib.util.spec_from_file_location("oglans_config", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load config module from {MODULE_PATH}")
oglans_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oglans_config)
ConfigManager = oglans_config.ConfigManager

class TestConfig:
    def test_load_default_config(self):
        """测试加载默认配置文件"""
        manager = ConfigManager()
        config = manager.load_config("configs/config.yaml")
        assert config['project']['name'] == "OG-LANS-DPO"
        assert config['algorithms']['lans']['enabled'] is True

    def test_singleton_pattern(self):
        """测试 ConfigManager 单例模式"""
        m1 = ConfigManager()
        m2 = ConfigManager()
        assert m1 is m2

    def test_cli_overrides(self):
        """测试命令行参数覆盖"""
        manager = ConfigManager()
        overrides = ["--training.max_steps", "100", "--algorithms.lans.enabled", "false"]
        config = manager.load_config("configs/config.yaml", overrides)

        assert config['training']['max_steps'] == 100
        assert config['algorithms']['lans']['enabled'] is False

    def test_get_config_without_load(self):
        """测试未加载直接获取配置应抛出异常"""
        # Reset singleton manually for test (hacky but needed for unit test isolation if running sequentially)
        ConfigManager._instance = None
        ConfigManager._config = None

        manager = ConfigManager()
        with pytest.raises(RuntimeError):
            manager.get_config()

    def test_lans_loss_baseline_exists_in_main_config(self):
        """主配置必须使用 loss_baseline 作为 LANS 损失基准键"""
        manager = ConfigManager()
        config = manager.load_config("configs/config.yaml")
        lans_cfg = config["algorithms"]["lans"]
        assert "loss_baseline" in lans_cfg
        assert isinstance(lans_cfg["loss_baseline"], (int, float))

    def test_lans_loss_baseline_exists_in_debug_config(self):
        """调试配置应与主配置保持一致，避免键名漂移"""
        manager = ConfigManager()
        config = manager.load_config("configs/config_debug.yaml")
        lans_cfg = config["algorithms"]["lans"]
        assert "loss_baseline" in lans_cfg
