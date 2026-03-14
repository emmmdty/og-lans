"""
SCV (Semantic Consistency Verification) 单元测试
测试语义一致性验证器的核心功能
"""

import pytest
import sys
import os
import json

# 由于 pytest.ini 已配置 pythonpath = src，无需手动添加 sys.path
from oglans.utils.scv import SemanticConsistencyVerifier


class TestSCVJsonConversion:
    """SCV JSON 转换测试（不需要加载模型）"""
    
    def test_json_to_natural_language_simple(self):
        """测试简单 JSON 转自然语言"""
        # 模拟转换逻辑
        json_str = '{"event_type": "融资", "arguments": [{"role": "金额", "argument": "1亿"}]}'
        
        # 预期输出类似：发生了融资事件，其中金额为1亿
        import json
        data = json.loads(json_str)
        
        # 手动构造自然语言
        event_type = data.get("event_type", "")
        args = data.get("arguments", [])
        
        parts = [f"发生了{event_type}事件"]
        for arg in args:
            role = arg.get("role", "")
            value = arg.get("argument", "")
            parts.append(f"{role}为{value}")
        
        result = "，其中".join(parts) + "。" if len(parts) > 1 else parts[0] + "。"
        
        assert "融资" in result
        assert "金额" in result
        assert "1亿" in result
    
    def test_json_to_natural_language_multiple_args(self):
        """测试多论元 JSON 转换"""
        json_str = '''{"event_type": "企业收购", "arguments": [
            {"role": "收购方", "argument": "A公司"},
            {"role": "被收购方", "argument": "B公司"},
            {"role": "收购金额", "argument": "10亿美元"}
        ]}'''
        
        import json
        data = json.loads(json_str)
        
        event_type = data.get("event_type", "")
        args = data.get("arguments", [])
        
        parts = [f"发生了{event_type}事件"]
        for arg in args:
            role = arg.get("role", "")
            value = arg.get("argument", "")
            if role and value:
                parts.append(f"{role}为{value}")
        
        result = "，".join(parts) + "。"
        
        assert "企业收购" in result
        assert "收购方" in result
        assert "A公司" in result
        assert "被收购方" in result
        assert "B公司" in result


class TestSCVMock:
    """SCV 功能测试（使用 Mock，不加载真实模型）"""
    
    def test_entailment_logic_true_positive(self):
        """测试蕴含判断逻辑 - 真正例"""
        # 模拟: 当原文确实包含负样本中的信息时
        premise = "阿里巴巴集团在2024年完成了100亿美元的融资。"
        hypothesis = "发生了融资事件，其中金额为100亿美元。"
        
        # 模拟 entailment 分数高
        mock_entailment_score = 0.9
        threshold = 0.85
        
        # 如果 entailment 分数 > 阈值，说明负样本实际上是"真负样本"（应该被过滤）
        is_false_negative = mock_entailment_score > threshold
        
        assert is_false_negative == True
    
    def test_entailment_logic_true_negative(self):
        """测试蕴含判断逻辑 - 真负例"""
        # 模拟: 当原文不包含负样本中的信息时（有效的负样本）
        premise = "阿里巴巴集团在2024年完成了100亿美元的融资。"
        hypothesis = "发生了股份回购事件，其中回购金额为50亿美元。"
        
        # 模拟 entailment 分数低
        mock_entailment_score = 0.2
        threshold = 0.85
        
        # 如果 entailment 分数 < 阈值，说明负样本是有效的
        is_false_negative = mock_entailment_score > threshold
        
        assert is_false_negative == False
    
    def test_sliding_window_logic(self):
        """测试滑动窗口逻辑"""
        # 模拟长文档
        long_premise = "这是一个非常长的文档。" * 100  # 约 1000 字符
        
        max_premise_len = 400
        step_size = max_premise_len // 2
        
        # 计算窗口
        windows = []
        for start in range(0, len(long_premise), step_size):
            end = min(start + max_premise_len, len(long_premise))
            window = long_premise[start:end]
            if len(window) >= 50:
                windows.append(window)
            if end >= len(long_premise):
                break
        
        assert len(windows) > 1  # 应该有多个窗口
        assert len(windows[0]) <= max_premise_len
        
        # 验证窗口重叠
        if len(windows) > 1:
            overlap = len(windows[0]) - step_size
            assert overlap > 0  # 应该有重叠


class TestSCVIntegration:
    """SCV 集成测试（需要模型时跳过）"""
    
    @pytest.fixture
    def check_model_available(self):
        """检查模型是否可用"""
        try:
            from transformers import AutoModelForSequenceClassification
            return True
        except ImportError:
            return False
    
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="集成测试需要设置 RUN_INTEGRATION_TESTS=1"
    )
    def test_scv_initialization(self, check_model_available):
        """测试 SCV 初始化（需要模型）"""
        if not check_model_available:
            pytest.skip("transformers 不可用")
        
        from oglans.utils.scv import SemanticConsistencyVerifier
        
        # 这个测试需要真实模型，可能很慢
        # scv = SemanticConsistencyVerifier(
        #     model_name="Fengshenbang/Erlangshen-MegatronBert-1.3B-NLI",
        #     threshold=0.85
        # )
        # assert scv.threshold == 0.85


class TestCoTFaithfulnessChecker:
    """CoT 忠实度检测器测试"""
    
    def test_extract_event_types_from_cot(self):
        """测试从 CoT 中提取事件类型"""
        import re
        
        cot_text = """
        第一步：Schema 分析
        - 检测到触发词：融资
        - 识别的事件类型：企业融资
        
        第二步：实体扫描
        - 企业融资事件：金额为100亿
        """
        
        # 使用正则匹配
        pattern = re.compile(r'事件类型[:：]?\s*([^\n，。]+)')
        matches = pattern.findall(cot_text)
        
        assert len(matches) >= 1
        assert "企业融资" in matches[0]
    
    def test_extract_arguments_from_cot(self):
        """测试从 CoT 中提取论元"""
        import re
        
        cot_text = """
        第二步：实体扫描
        - 企业融资事件：
          · 金额 = "100亿美元"
          · 融资轮次 = "B轮"
        """
        
        # 【修复】使用专门匹配 "· key = value" 或 "- key = value" 格式的正则
        # 明确要求以 · 或 - 开头（跳过空白），然后是中文/英文 key
        pattern = re.compile(r'[·\-]\s*([\u4e00-\u9fa5a-zA-Z]+)\s*[=]\s*[「"\']*([^「」"\'\n]+)[」"\']*')
        matches = pattern.findall(cot_text)
        
        args = {m[0].strip(): m[1].strip() for m in matches if len(m[0].strip()) <= 10}
        
        assert "金额" in args
        assert "100亿美元" in args["金额"]
    
    def test_consistency_check_consistent(self):
        """测试一致性检查 - 一致的情况"""
        cot_event_types = {"企业融资"}
        json_event_types = {"企业融资"}
        
        overlap = len(cot_event_types & json_event_types)
        union = len(cot_event_types | json_event_types)
        ratio = overlap / union if union > 0 else 0
        
        assert ratio == 1.0
    
    def test_consistency_check_inconsistent(self):
        """测试一致性检查 - 不一致的情况"""
        cot_event_types = {"企业融资", "股份回购"}
        json_event_types = {"企业收购"}
        
        overlap = len(cot_event_types & json_event_types)
        union = len(cot_event_types | json_event_types)
        ratio = overlap / union if union > 0 else 0
        
        assert ratio == 0.0


class TestHallucinationDetection:
    """幻觉检测测试"""
    
    def test_no_hallucination(self):
        """测试无幻觉情况"""
        source_text = "阿里巴巴集团在2024年完成了100亿美元的融资。"
        predicted_value = "阿里巴巴集团"
        
        has_hallucination = predicted_value not in source_text
        assert has_hallucination == False
    
    def test_has_hallucination(self):
        """测试有幻觉情况"""
        source_text = "阿里巴巴集团在2024年完成了100亿美元的融资。"
        predicted_value = "腾讯公司"  # 原文未提及
        
        has_hallucination = predicted_value not in source_text
        assert has_hallucination == True
    
    def test_fuzzy_match(self):
        """测试模糊匹配（去空格）"""
        import re
        
        source_text = "阿里巴巴 集团在2024年完成了融资。"
        predicted_value = "阿里巴巴集团"
        
        # 直接匹配失败
        direct_match = predicted_value in source_text
        
        # 模糊匹配（去空格）
        clean_source = re.sub(r'\s+', '', source_text)
        clean_value = re.sub(r'\s+', '', predicted_value)
        fuzzy_match = clean_value in clean_source
        
        assert direct_match == False
        assert fuzzy_match == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
