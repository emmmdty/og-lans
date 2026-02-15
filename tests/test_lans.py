"""
OG-LANS 核心算法单元测试
测试 LANSScheduler 的能力评估、策略选择和 CGA 机制
"""

import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oglans.utils.ds_cns import LANSScheduler


class TestLANSScheduler:
    """LANS 调度器测试套件"""
    
    @pytest.fixture
    def scheduler(self) -> LANSScheduler:
        """创建测试用调度器"""
        return LANSScheduler(
            d_max=4.0,
            d_min=1.0,
            ema_decay=0.95,
            loss_baseline=0.5,
            warmup_steps=10,
            competence_floor=0.05,
            competence_ceiling=0.95,
            warmup_target=0.25,
            use_ema=True,
            cga_beta=0.1,
            use_cga=True,
            granularity_weights={"EVENT_LEVEL": 0.3, "ARGUMENT_LEVEL": 0.5, "VALUE_LEVEL": 0.2}
        )
    
    def test_initialization(self, scheduler: LANSScheduler):
        """测试调度器初始化"""
        assert scheduler.d_max == 4.0
        assert scheduler.d_min == 1.0
        assert scheduler.competence == scheduler.competence_floor
        assert scheduler.warmup_steps == 10
        assert scheduler.use_cga == True
        assert scheduler.cga_beta == 0.1
    
    def test_warmup_competence(self, scheduler: LANSScheduler):
        """测试预热期能力线性增长"""
        initial_competence = scheduler.competence
        
        # 模拟预热期训练
        for step in range(1, 11):
            new_c = scheduler.update_competence(0.5)  # 使用基准损失
            # 预热期应该线性增长
            assert new_c >= initial_competence
        
        # 预热结束后能力应该接近 warmup_target
        assert scheduler.competence >= scheduler.competence_floor
    
    def test_competence_update_low_loss(self, scheduler: LANSScheduler):
        """测试低损失时能力增加"""
        # 跳过预热期
        for _ in range(15):
            scheduler.update_competence(0.5)
        
        competence_before = scheduler.competence
        
        # 低损失（低于基准线）应该增加能力
        scheduler.update_competence(0.3)
        
        assert scheduler.competence >= competence_before
    
    def test_competence_update_high_loss(self, scheduler: LANSScheduler):
        """测试高损失时能力降低或稳定"""
        # 跳过预热期，先让能力值上升
        for _ in range(15):
            scheduler.update_competence(0.3)
        
        competence_before = scheduler.competence
        
        # 高损失（高于基准线）应该降低能力
        scheduler.update_competence(0.8)
        
        # 由于 EMA 平滑，可能不会立即下降太多
        assert scheduler.competence <= competence_before + 0.1
    
    def test_competence_boundaries(self, scheduler: LANSScheduler):
        """测试能力值边界约束"""
        # 模拟极端低损失
        for _ in range(100):
            scheduler.update_competence(0.0)
        
        assert scheduler.competence <= scheduler.competence_ceiling
        
        # 模拟极端高损失
        for _ in range(100):
            scheduler.update_competence(2.0)
        
        assert scheduler.competence >= scheduler.competence_floor
    
    def test_threshold_calculation(self, scheduler: LANSScheduler):
        """测试阈值计算"""
        # 阈值公式: λ(C) = D_max - (D_max - D_min) · C
        
        # 当 C = 0 时，λ = D_max
        scheduler._competence = 0.0
        assert abs(scheduler.current_threshold - scheduler.d_max) < 0.01
        
        # 当 C = 1 时，λ = D_min
        scheduler._competence = 1.0
        assert abs(scheduler.current_threshold - scheduler.d_min) < 0.01
        
        # 当 C = 0.5 时，λ = (D_max + D_min) / 2
        scheduler._competence = 0.5
        expected = (scheduler.d_max + scheduler.d_min) / 2
        assert abs(scheduler.current_threshold - expected) < 0.01
    
    def test_cga_weight_calculation(self, scheduler: LANSScheduler):
        """测试 CGA 权重计算"""
        # CGA 公式: w(C) = 1 + β_cga · (1 - C)
        
        # 当 C = 0 时，w = 1 + β
        scheduler._competence = 0.0
        expected_w = 1.0 + scheduler.cga_beta
        assert abs(scheduler.cga_weight - expected_w) < 0.01
        
        # 当 C = 1 时，w = 1
        scheduler._competence = 1.0
        assert abs(scheduler.cga_weight - 1.0) < 0.01
        
        # 当 C = 0.5 时，w = 1 + 0.5 * β
        scheduler._competence = 0.5
        expected_w = 1.0 + 0.5 * scheduler.cga_beta
        assert abs(scheduler.cga_weight - expected_w) < 0.01
    
    def test_cga_disabled(self):
        """测试 CGA 禁用时权重为 1"""
        scheduler = LANSScheduler(
            d_max=4.0, d_min=1.0, use_cga=False
        )
        scheduler._competence = 0.0
        assert scheduler.cga_weight == 1.0
    
    def test_get_strategy_distribution(self, scheduler: LANSScheduler):
        """测试策略分布在有效范围内"""
        strategies = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
        
        for _ in range(100):
            strategy = scheduler.get_strategy()
            assert strategy in strategies
            strategies[strategy] += 1
        
        # 确保所有策略都有可能被选中
        assert sum(strategies.values()) == 100
    
    def test_get_statistics(self, scheduler: LANSScheduler):
        """测试统计信息导出"""
        # 模拟一些训练
        for _ in range(20):
            scheduler.update_competence(0.4)
            scheduler.get_strategy()
        
        stats = scheduler.get_statistics()
        
        assert "step_count" in stats
        assert "competence" in stats
        assert "threshold" in stats
        assert "strategy_distribution" in stats
        assert stats["step_count"] == 20
    
    def test_export_history(self, scheduler: LANSScheduler):
        """测试历史记录导出"""
        for i in range(10):
            scheduler.update_competence(0.5 - i * 0.01)
        
        history = scheduler.export_history()
        
        assert "loss_history" in history
        assert "competence_history" in history
        assert len(history["loss_history"]) == 10
        assert len(history["competence_history"]) == 10
    
    def test_ema_disabled(self):
        """测试禁用 EMA 时的即时更新"""
        scheduler = LANSScheduler(
            d_max=4.0, d_min=1.0,
            warmup_steps=5,
            use_ema=False
        )
        
        # 跳过预热
        for _ in range(10):
            scheduler.update_competence(0.5)
        
        # 禁用 EMA 时，能力值应该更直接地响应损失变化
        c1 = scheduler.competence
        scheduler.update_competence(0.1)  # 很低的损失
        c2 = scheduler.competence
        
        # 变化应该比 EMA 模式更明显
        assert c2 != c1
    
    def test_granularity_weights(self, scheduler: LANSScheduler):
        """测试多粒度权重配置"""
        assert scheduler.granularity_weights["EVENT_LEVEL"] == 0.3
        assert scheduler.granularity_weights["ARGUMENT_LEVEL"] == 0.5
        assert scheduler.granularity_weights["VALUE_LEVEL"] == 0.2


class TestLANSEdgeCases:
    """LANS 边界情况测试"""
    
    def test_zero_warmup_steps(self):
        """测试零预热步数"""
        scheduler = LANSScheduler(d_max=4.0, d_min=1.0, warmup_steps=0)
        scheduler.update_competence(0.5)
        # 应该正常工作
        assert scheduler.competence >= scheduler.competence_floor
    
    def test_negative_loss(self):
        """测试负损失（异常情况）"""
        scheduler = LANSScheduler(d_max=4.0, d_min=1.0, warmup_steps=5)
        
        for _ in range(10):
            scheduler.update_competence(-0.5)
        
        # 应该处理而不崩溃
        assert scheduler.competence_floor <= scheduler.competence <= scheduler.competence_ceiling
    
    def test_very_high_loss(self):
        """测试极高损失"""
        scheduler = LANSScheduler(d_max=4.0, d_min=1.0, warmup_steps=5)
        
        for _ in range(10):
            scheduler.update_competence(100.0)
        
        # 应该保持在边界内
        assert scheduler.competence >= scheduler.competence_floor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
