import importlib


def test_import_oglans_without_training_dependencies():
    """
    目标：即使训练依赖（如 unsloth）不可用，也应能导入非训练模块。
    """
    pkg = importlib.import_module("oglans")
    assert hasattr(pkg, "DuEEFinAdapter")
    try:
        _ = pkg.setup_logger
    except ImportError:
        # 允许在缺少可选依赖时延迟报错；关键是 import oglans 不应直接崩溃
        pass


def test_import_oglans_trainer_module_is_lazy():
    """
    目标：CPU-only 环境下导入 oglans.trainer 不应因为 Unsloth/GPU 缺失而立刻崩溃。
    """
    trainer_pkg = importlib.import_module("oglans.trainer")
    assert trainer_pkg is not None
    assert hasattr(trainer_pkg, "__all__")
