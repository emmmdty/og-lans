import json
import random
import pytest

pytest.importorskip("networkx")

from oglans.utils.ds_cns import DSCNSampler, LANSScheduler

class TestLANSDeterminism:
    """测试 LANS 组件的确定性 (Reproducibility)"""

    @pytest.fixture
    def schema_path(self, tmp_path):
        """创建一个临时的 Schema 文件"""
        schema_file = tmp_path / "test_schema.json"
        schema_data = [
            {
                "event_type": "Win",
                "role_list": [{"role": "Winner"}, {"role": "Loser"}, {"role": "Prize"}]
            },
            {
                "event_type": "Lose",
                "role_list": [{"role": "Loser"}, {"role": "Winner"}]
            }
        ]
        with open(schema_file, 'w', encoding='utf-8') as f:
            for item in schema_data:
                f.write(json.dumps(item) + "\n")
        return str(schema_file)

    def test_sampler_determinism(self, schema_path):
        """测试 DSCNSampler 生成负样本的确定性"""

        input_json = json.dumps([{
            "event_type": "Win",
            "arguments": [{"role": "Winner", "argument": "Alice"}, {"role": "Prize", "argument": "100"}]
        }])

        # Run 1
        random.seed(42)
        sampler1 = DSCNSampler(schema_path, c0=0.1)
        neg1 = sampler1.generate_negative_json(input_json, "EASY", 0, 100)

        # Run 2
        random.seed(42)
        sampler2 = DSCNSampler(schema_path, c0=0.1)
        neg2 = sampler2.generate_negative_json(input_json, "EASY", 0, 100)

        assert neg1 == neg2, "Same seed should produce identical negative samples"

        # Run 3 (Different seed)
        random.seed(43)
        sampler3 = DSCNSampler(schema_path, c0=0.1)
        neg3 = sampler3.generate_negative_json(input_json, "EASY", 0, 100)

        # Note: In EASY mode with few candidates, it might still pick the same one by chance,
        # but with enough complexity or checking internal state, it should ideally differ or at least RNG state differs.
        # Here we just ensure neg1 == neg2.

    def test_scheduler_determinism(self):
        """测试 LANSScheduler 能力更新的确定性"""
        scheduler1 = LANSScheduler(d_max=4, d_min=1, use_ema=True)
        scheduler2 = LANSScheduler(d_max=4, d_min=1, use_ema=True)

        losses = [0.5, 0.4, 0.6, 0.3]

        c1_history = []
        for l in losses:
            c1_history.append(scheduler1.update_competence(l))

        c2_history = []
        for l in losses:
            c2_history.append(scheduler2.update_competence(l))

        assert c1_history == c2_history, "Scheduler updates should be deterministic given same loss sequence"
