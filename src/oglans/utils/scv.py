# src/utils/scv.py
"""
SCV: Semantic Consistency Verification (2026 Enhanced Version)
语义一致性验证模块

============================================================================
学术创新点 (2026年增强):
============================================================================
1. 假负样本过滤: 基于 NLI 模型识别并丢弃"实际正确但被标记为负"的样本
2. 【2026 新增】CoT-JSON 一致性检测: 验证推理过程与最终输出的一致性
3. 【2026 新增】幻觉检测辅助: 识别输出中不存在于原文的实体

核心组件:
- SemanticConsistencyVerifier: NLI 驱动的假负样本过滤器
- CoTFaithfulnessChecker: 推理-输出一致性检测器 (新增)
============================================================================
"""
import torch
import json
import re
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from .hub_runtime import (
    configure_model_download_runtime,
    get_model_download_runtime_snapshot,
    resolve_model_name_or_path,
)

logger = logging.getLogger("OGLANS")


@dataclass
class SCVLiteDecision:
    mode: str
    triggered: bool
    reasons: List[str]
    call_count: int
    wall_clock_seconds: float
    verification_attempted: bool = False
    verification_result: Optional[bool] = None


def evaluate_scv_lite(
    postprocess_diagnostics: Optional[Dict[str, Any]],
    *,
    mode: str = "off",
    verifier: Optional["SemanticConsistencyVerifier"] = None,
    source_text: str = "",
    pred_events: Optional[List[Dict[str, Any]]] = None,
) -> SCVLiteDecision:
    """
    Lightweight SCV gate for Phase 3.

    off:
        Ignore diagnostics entirely.
    trigger_only:
        Record trigger reasons without invoking the heavy verifier.
    verify_triggered:
        Attempt verifier only when triggers exist and a verifier instance is provided.
    """
    normalized_mode = str(mode or "off")
    reasons = list((postprocess_diagnostics or {}).get("scv_lite_reasons", []) or [])
    triggered = bool((postprocess_diagnostics or {}).get("scv_lite_triggered", False) or reasons)

    if normalized_mode == "off":
        return SCVLiteDecision(
            mode="off",
            triggered=False,
            reasons=[],
            call_count=0,
            wall_clock_seconds=0.0,
        )

    if normalized_mode == "trigger_only":
        return SCVLiteDecision(
            mode="trigger_only",
            triggered=triggered,
            reasons=reasons,
            call_count=0,
            wall_clock_seconds=0.0,
        )

    if normalized_mode != "verify_triggered":
        raise ValueError(f"Unsupported SCV-Lite mode: {normalized_mode}")

    if not triggered or verifier is None:
        return SCVLiteDecision(
            mode="verify_triggered",
            triggered=triggered,
            reasons=reasons,
            call_count=0,
            wall_clock_seconds=0.0,
            verification_attempted=False,
            verification_result=None,
        )

    hypothesis = json.dumps(pred_events or [], ensure_ascii=False)
    started = time.perf_counter()
    verification_result = verifier.is_false_negative(source_text, hypothesis)
    elapsed = time.perf_counter() - started
    return SCVLiteDecision(
        mode="verify_triggered",
        triggered=triggered,
        reasons=reasons,
        call_count=1,
        wall_clock_seconds=elapsed,
        verification_attempted=True,
        verification_result=bool(verification_result),
    )

class SemanticConsistencyVerifier:
    """
    SCV: Semantic Consistency Verification
    基于 NLI 模型过滤 '假负样本' (False Negatives)。
    当前模型: Fengshenbang/Erlangshen-MegatronBert-1.3B-NLI
    说明：已移除 pipeline，改用原生模型推理以避免串行警告。
    """
    def __init__(
        self,
        model_name: str,
        threshold: float = 0.8,
        source: str = "modelscope",
        entailment_idx: Optional[int] = None,
        progress_log_interval: int = 200,
        progress_log_seconds: float = 30.0,
    ):
        self.threshold = threshold
        self.source = str(source or "modelscope").lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_log_interval = max(1, int(progress_log_interval))
        self.progress_log_seconds = max(5.0, float(progress_log_seconds))
        self._calls = 0
        self._total_windows = 0
        self._total_time_seconds = 0.0
        self._last_progress_log_ts = time.perf_counter()
        
        logger.info(f"Loading SCV model: {model_name}...")
        project_root = Path(__file__).resolve().parents[3]
        configure_model_download_runtime(project_root, source=self.source)
        model_path = resolve_model_name_or_path(
            model_name,
            source=self.source,
            logger=logger,
            project_root=project_root,
        )

        # 2. 加载 Tokenizer 和 Model (替代 Pipeline)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval() # 开启评估模式
        except Exception as e:
            model_runtime = get_model_download_runtime_snapshot(source=self.source)
            logger.error(
                "Failed to load SCV model: "
                f"model_name={model_name}, resolved_path={model_path}, "
                f"runtime={model_runtime}, "
                f"error={e}"
            )
            raise e
        
        # 3. 验证并确定 entailment 标签索引
        self.entailment_idx = self._detect_entailment_index(explicit_idx=entailment_idx)
        
        # 4. 【关键修复】断言测试 - 用简单样本验证标签检测是否正确
        self._validate_entailment_detection()

    def _validate_entailment_detection(self):
        """
        使用简单测试样本验证 entailment 标签索引是否正确
        这是学术严谨性的关键保障，防止标签反转导致 SCV 逻辑完全失效
        """
        # 测试用例：明显的蕴含关系
        test_premise = "张三是一名医生，在北京工作。"
        test_hypothesis_entail = "张三是医生。"
        test_hypothesis_contradict = "张三是一名教师。"
        
        try:
            # 测试蕴含关系
            inputs_entail = self.tokenizer(
                test_premise, test_hypothesis_entail,
                return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            
            inputs_contradict = self.tokenizer(
                test_premise, test_hypothesis_contradict,
                return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                logits_entail = self.model(**inputs_entail).logits[0]
                logits_contradict = self.model(**inputs_contradict).logits[0]
                
                probs_entail = torch.softmax(logits_entail, dim=-1)
                probs_contradict = torch.softmax(logits_contradict, dim=-1)
            
            entail_score_correct = probs_entail[self.entailment_idx].item()
            entail_score_wrong = probs_contradict[self.entailment_idx].item()
            
            logger.info(f"🧪 SCV 断言测试:")
            logger.info(f"   蕴含样本的 entailment 分数: {entail_score_correct:.3f}")
            logger.info(f"   矛盾样本的 entailment 分数: {entail_score_wrong:.3f}")
            
            # 验证逻辑：蕴含样本的分数应该明显高于矛盾样本
            if entail_score_correct <= entail_score_wrong:
                logger.error(
                    f"❌ SCV 断言失败！标签索引可能错误。"
                    f"蕴含样本分数({entail_score_correct:.3f}) <= 矛盾样本分数({entail_score_wrong:.3f})"
                )
                logger.error("   建议：检查 NLI 模型的 label2id 配置，或手动指定 entailment_idx")
                raise ValueError(
                    f"SCV entailment 标签检测验证失败！"
                    f"当前 entailment_idx={self.entailment_idx} 可能不正确。"
                )
            else:
                logger.info(f"✅ SCV 断言测试通过，entailment_idx={self.entailment_idx} 验证正确")
                
        except Exception as e:
            if "断言失败" in str(e) or "验证失败" in str(e):
                raise  # 重新抛出验证失败的异常
            logger.warning(f"⚠️ SCV 断言测试执行异常: {e}，跳过验证")

    def _detect_entailment_index(self, explicit_idx: Optional[int] = None) -> int:
        """
        检测 NLI 模型的 entailment 标签索引
        不同模型可能使用不同的标签顺序：
        - 常见顺序1: [entailment, neutral, contradiction] -> entailment=0
        - 常见顺序2: [contradiction, neutral, entailment] -> entailment=2
        """
        labels = getattr(self.model.config, "label2id", None)
        id2label = getattr(self.model.config, "id2label", None)
        
        # 打印标签映射用于调试
        logger.info(f"📋 NLI 模型标签配置:")
        logger.info(f"   label2id: {labels}")
        logger.info(f"   id2label: {id2label}")

        num_labels = getattr(self.model.config, "num_labels", None)
        if explicit_idx is not None:
            entailment_idx = int(explicit_idx)
            if entailment_idx < 0 or (num_labels is not None and entailment_idx >= int(num_labels)):
                raise ValueError(
                    f"Explicit SCV entailment_idx out of range: {entailment_idx} (num_labels={num_labels})"
                )
            logger.info(f"✅ 使用显式配置的 entailment_idx={entailment_idx}")
            return entailment_idx

        if labels is not None:
            # 尝试多种可能的 entailment 标签名称
            for key in ['entailment', 'ENTAILMENT', 'Entailment', '蕴含']:
                if key in labels:
                    entailment_idx = int(labels[key])
                    logger.info(f"✅ 检测到 entailment 标签: '{key}' -> index={entailment_idx}")
                    return entailment_idx
        
        # 如果 label2id 中没有明确的 entailment，尝试从 id2label 推断
        if id2label is not None:
            for idx, label_name in id2label.items():
                if 'entail' in str(label_name).lower():
                    entailment_idx = int(idx)
                    logger.info(f"✅ 从 id2label 推断 entailment 标签: '{label_name}' -> index={entailment_idx}")
                    return entailment_idx
        
        raise ValueError(
            "Could not detect SCV entailment label index from model config. "
            "Set algorithms.scv.entailment_idx explicitly."
        )
    
    def _json_to_natural_language(self, json_str: str) -> str:
        """
        将 JSON 格式的事件信息转换为自然语言陈述
        解决 NLI 模型对 JSON 格式输入的 OOD 问题
        
        示例:
        输入: '{"event_type": "融资", "arguments": [{"role": "金额", "argument": "1亿"}]}'
        输出: "发生了融资事件，金额为1亿"
        """
        try:
            # 尝试解析 JSON
            if isinstance(json_str, str):
                data = json.loads(json_str)
            else:
                data = json_str
            
            # 处理单个事件或事件列表
            if isinstance(data, list):
                events = data
            else:
                events = [data]
            
            sentences = []
            for event in events:
                event_type = event.get('event_type', '未知事件')
                arguments = event.get('arguments', [])
                
                if arguments:
                    arg_parts = []
                    for arg in arguments:
                        role = arg.get('role', '')
                        value = arg.get('argument', '')
                        if role and value:
                            arg_parts.append(f"{role}为{value}")
                    
                    if arg_parts:
                        arg_str = "，".join(arg_parts)
                        sentences.append(f"发生了{event_type}事件，其中{arg_str}")
                    else:
                        sentences.append(f"发生了{event_type}事件")
                else:
                    sentences.append(f"发生了{event_type}事件")
            
            return "。".join(sentences) + "。" if sentences else json_str
            
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            # JSON 解析失败时，返回原始字符串的简化描述
            logger.debug(f"JSON 转换失败，使用原始格式: {e}")
            return f"这段文本包含了以下事件信息：{json_str}"

    def is_false_negative(self, premise: str, hypothesis_json: str) -> bool:
        """
        判断生成的负样本是否在语义上其实是正确的（蕴含关系）。
        
        【关键修复】实现滑窗机制处理长文档：
        - 对于超过 NLI 模型 max_length 的长文档，使用滑动窗口
        - 取所有窗口中 entailment 分数的最大值
        - 这确保即使事件论元出现在文档后半部分也能被正确校验
        """
        call_start_ts = time.perf_counter()

        # 兜底：如果输入为空，直接跳过校验（视为非假负例，保留样本）
        if not premise or not hypothesis_json:
            return False

        # 【关键修复】将 JSON 转换为自然语言陈述，提高 NLI 准确性
        hypothesis = self._json_to_natural_language(hypothesis_json)
        
        # 计算 hypothesis 的 token 长度（粗略估计）
        hypothesis_len = len(hypothesis)
        max_premise_len = 512 - hypothesis_len - 20
        if max_premise_len < 100:
            max_premise_len = 100
        
        # 【关键修复】滑窗机制处理长文档
        if len(premise) <= max_premise_len:
            # 短文档：直接处理
            windows = [premise]
        else:
            # 长文档：使用滑动窗口
            # 窗口大小 = max_premise_len, 步长 = max_premise_len // 2 (50% 重叠)
            window_size = max_premise_len
            step_size = max(window_size // 2, 100)
            windows = []
            
            for start in range(0, len(premise), step_size):
                end = min(start + window_size, len(premise))
                window = premise[start:end]
                if len(window) >= 50:  # 忽略过短的尾部片段
                    windows.append(window)
                if end >= len(premise):
                    break
            
            # 确保至少有一个窗口
            if not windows:
                windows = [premise[:max_premise_len]]
            
            logger.debug(f"SCV 滑窗: 文档长度 {len(premise)}, 窗口数 {len(windows)}")
        
        # 对每个窗口计算 entailment 分数，取最大值
        max_entailment_score = 0.0
        
        try:
            for window in windows:
                inputs = self.tokenizer(
                    window, 
                    hypothesis, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                
                entailment_score = probs[self.entailment_idx].item()
                max_entailment_score = max(max_entailment_score, entailment_score)
                
                # 提前终止：如果已经超过阈值，无需继续
                if max_entailment_score > self.threshold:
                    break
            
            # 逻辑：如果蕴含分数过高，说明该负样本其实是对的 -> 判为假负样本 -> 丢弃
            if max_entailment_score > self.threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"SCV check failed: {str(e)}")
            return False
        finally:
            self._calls += 1
            self._total_windows += len(windows)
            elapsed = time.perf_counter() - call_start_ts
            self._total_time_seconds += elapsed

            now = time.perf_counter()
            should_log = (
                self._calls % self.progress_log_interval == 0
                or (now - self._last_progress_log_ts) >= self.progress_log_seconds
            )
            if should_log:
                avg_windows = self._total_windows / max(self._calls, 1)
                avg_time_ms = self._total_time_seconds * 1000.0 / max(self._calls, 1)
                throughput = self._calls / max(self._total_time_seconds, 1e-6)
                logger.info(
                    "🔎 SCV 心跳: "
                    f"calls={self._calls}, avg_windows={avg_windows:.2f}, "
                    f"avg_time={avg_time_ms:.1f}ms, throughput={throughput:.2f} checks/s"
                )
                self._last_progress_log_ts = now


# =========================================================================
# 以下类目前未被集成使用，保留供未来扩展
# 当前 evaluate.py 使用内联实现 (AcademicEventEvaluator.check_hallucination)
# 如需使用完整版本，可导入这些类替换内联实现
# =========================================================================

class CoTFaithfulnessChecker:
    """
    【2026 新增】【暂未集成】CoT-JSON 一致性检测器
    
    注意: 当前未被使用。evaluate.py 中有简化的内联实现。
    此类提供更完整的检测功能，包括事件类型、论元、数值一致性检查。
    
    检测 Chain-of-Thought 推理过程与最终 JSON 输出之间的一致性。
    这是 2026 年论文的关键评估指标之一，用于发现"推理与抽取不一致"的幻觉问题。
    
    检测维度:
    1. 事件类型一致性: CoT 提到的事件类型是否与 JSON 输出匹配
    2. 论元一致性: CoT 中提取的论元是否出现在 JSON 输出中
    3. 数值一致性: 数值是否在 CoT 和 JSON 中保持一致
    """
    
    # 事件类型正则模式
    EVENT_TYPE_PATTERN = re.compile(r'(?:事件类型|检测到|触发|发生了)[:：]?\s*[「"\']*([^「」"\'，。\n]+)[」"\']*')
    
    # 论元提取模式
    ARGUMENT_PATTERN = re.compile(r'([^\s=:：]+)\s*[=:：]\s*[「"\']*([^「」"\',，。\n]+)[」"\']*')
    
    # 数值模式
    NUMBER_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:亿|万|元|%|股|份)?')
    
    def __init__(self):
        self.stats = {
            "total_checked": 0,
            "type_consistent": 0,
            "argument_consistent": 0,
            "value_consistent": 0,
            "fully_consistent": 0
        }
    
    def extract_cot_info(self, cot_text: str) -> Dict:
        """从 CoT 文本中提取关键信息"""
        # 提取事件类型
        event_types = set()
        for match in self.EVENT_TYPE_PATTERN.finditer(cot_text):
            event_types.add(match.group(1).strip())
        
        # 提取论元
        arguments = {}
        for match in self.ARGUMENT_PATTERN.finditer(cot_text):
            role = match.group(1).strip()
            value = match.group(2).strip()
            if len(role) <= 10 and len(value) <= 50:  # 过滤噪声
                arguments[role] = value
        
        # 提取数值
        numbers = set()
        for match in self.NUMBER_PATTERN.finditer(cot_text):
            numbers.add(match.group(1))
        
        return {
            "event_types": event_types,
            "arguments": arguments,
            "numbers": numbers
        }
    
    def extract_json_info(self, json_str: str) -> Dict:
        """从 JSON 输出中提取关键信息"""
        event_types = set()
        arguments = {}
        numbers = set()
        
        try:
            # 清理 JSON 字符串
            json_start = json_str.find("```json")
            json_end = json_str.rfind("```")
            
            if json_start != -1 and json_end > json_start:
                content = json_str[json_start + 7:json_end].strip()
            else:
                content = json_str.strip()
            
            events = json.loads(content)
            
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        etype = event.get("event_type", "")
                        if etype:
                            event_types.add(etype)
                        
                        for arg in event.get("arguments", []):
                            if isinstance(arg, dict):
                                role = arg.get("role", "")
                                value = str(arg.get("argument", ""))
                                if role and value:
                                    arguments[role] = value
                                    # 提取数值
                                    for match in self.NUMBER_PATTERN.finditer(value):
                                        numbers.add(match.group(1))
        except:
            pass
        
        return {
            "event_types": event_types,
            "arguments": arguments,
            "numbers": numbers
        }
    
    def check_faithfulness(self, full_response: str) -> Dict:
        """
        检查 CoT 与 JSON 的一致性
        
        Args:
            full_response: 包含 <thought> 和 JSON 的完整响应
        
        Returns:
            一致性检查结果
        """
        self.stats["total_checked"] += 1
        
        result = {
            "is_type_consistent": False,
            "is_argument_consistent": False,
            "is_value_consistent": False,
            "is_fully_consistent": False,
            "type_overlap_ratio": 0.0,
            "argument_overlap_ratio": 0.0,
            "value_overlap_ratio": 0.0,
            "details": {}
        }
        
        # 分离 CoT 和 JSON
        thought_match = re.search(r'<thought>(.*?)</thought>', full_response, re.DOTALL)
        if not thought_match:
            # 没有 thought 标签，尝试查找 ```json 之前的内容
            json_start = full_response.find("```json")
            if json_start > 0:
                cot_text = full_response[:json_start]
            else:
                cot_text = ""
        else:
            cot_text = thought_match.group(1)
        
        # 提取信息
        cot_info = self.extract_cot_info(cot_text)
        json_info = self.extract_json_info(full_response)
        
        # 检查事件类型一致性
        if cot_info["event_types"] and json_info["event_types"]:
            overlap = len(cot_info["event_types"] & json_info["event_types"])
            union = len(cot_info["event_types"] | json_info["event_types"])
            result["type_overlap_ratio"] = overlap / union if union > 0 else 0.0
            result["is_type_consistent"] = result["type_overlap_ratio"] >= 0.5
        elif not cot_info["event_types"] and not json_info["event_types"]:
            result["is_type_consistent"] = True
            result["type_overlap_ratio"] = 1.0
        
        # 检查论元一致性
        if cot_info["arguments"] and json_info["arguments"]:
            cot_values = set(cot_info["arguments"].values())
            json_values = set(json_info["arguments"].values())
            overlap = len(cot_values & json_values)
            union = len(cot_values | json_values)
            result["argument_overlap_ratio"] = overlap / union if union > 0 else 0.0
            result["is_argument_consistent"] = result["argument_overlap_ratio"] >= 0.3
        elif not cot_info["arguments"] and not json_info["arguments"]:
            result["is_argument_consistent"] = True
            result["argument_overlap_ratio"] = 1.0
        
        # 检查数值一致性
        if cot_info["numbers"] and json_info["numbers"]:
            overlap = len(cot_info["numbers"] & json_info["numbers"])
            union = len(cot_info["numbers"] | json_info["numbers"])
            result["value_overlap_ratio"] = overlap / union if union > 0 else 0.0
            result["is_value_consistent"] = result["value_overlap_ratio"] >= 0.5
        elif not cot_info["numbers"] and not json_info["numbers"]:
            result["is_value_consistent"] = True
            result["value_overlap_ratio"] = 1.0
        
        # 综合判断
        result["is_fully_consistent"] = (
            result["is_type_consistent"] and 
            result["is_argument_consistent"] and 
            result["is_value_consistent"]
        )
        
        # 更新统计
        if result["is_type_consistent"]:
            self.stats["type_consistent"] += 1
        if result["is_argument_consistent"]:
            self.stats["argument_consistent"] += 1
        if result["is_value_consistent"]:
            self.stats["value_consistent"] += 1
        if result["is_fully_consistent"]:
            self.stats["fully_consistent"] += 1
        
        result["details"] = {
            "cot_event_types": list(cot_info["event_types"]),
            "json_event_types": list(json_info["event_types"]),
            "cot_argument_count": len(cot_info["arguments"]),
            "json_argument_count": len(json_info["arguments"])
        }
        
        return result
    
    def get_faithfulness_score(self) -> float:
        """获取整体 CoT 忠实度分数"""
        if self.stats["total_checked"] == 0:
            return 1.0
        return self.stats["fully_consistent"] / self.stats["total_checked"]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = self.stats["total_checked"] or 1
        return {
            "total_checked": self.stats["total_checked"],
            "type_consistency_rate": self.stats["type_consistent"] / total,
            "argument_consistency_rate": self.stats["argument_consistent"] / total,
            "value_consistency_rate": self.stats["value_consistent"] / total,
            "full_consistency_rate": self.stats["fully_consistent"] / total
        }


class HallucinationDetector:
    """
    【2026 新增】【暂未集成】幻觉检测器
    
    注意: 当前未被使用。evaluate.py 中有简化的内联实现。
    此类提供更完整的检测功能，包括模糊匹配和详细统计。
    
    检测模型输出中是否包含原文未提及的实体（幻觉）。
    这是 2026 年论文的关键评估指标之一。
    """
    
    def __init__(self):
        self.stats = {
            "total_checked": 0,
            "hallucination_count": 0,
            "hallucinated_entities": []
        }
    
    def detect_hallucination(
        self, 
        source_text: str, 
        model_output: str,
        check_entities: bool = True,
        check_numbers: bool = True
    ) -> Dict:
        """
        检测幻觉
        
        Args:
            source_text: 原始输入文本
            model_output: 模型输出（JSON 格式）
            check_entities: 是否检查实体幻觉
            check_numbers: 是否检查数值幻觉
        
        Returns:
            幻觉检测结果
        """
        self.stats["total_checked"] += 1
        
        result = {
            "has_hallucination": False,
            "hallucinated_items": [],
            "total_items": 0,
            "hallucination_rate": 0.0
        }
        
        # 提取 JSON 中的所有论元值
        try:
            json_start = model_output.find("```json")
            json_end = model_output.rfind("```")
            
            if json_start != -1 and json_end > json_start:
                content = model_output[json_start + 7:json_end].strip()
            else:
                content = model_output.strip()
            
            events = json.loads(content)
            
            all_values = []
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        for arg in event.get("arguments", []):
                            if isinstance(arg, dict):
                                value = str(arg.get("argument", ""))
                                if value:
                                    all_values.append(value)
            
            result["total_items"] = len(all_values)
            
            # 检查每个值是否出现在原文中
            for value in all_values:
                # 清理值
                clean_value = value.strip()
                if len(clean_value) < 2:
                    continue
                
                # 检查是否在原文中
                if clean_value not in source_text:
                    # 尝试模糊匹配（去掉空格和标点）
                    clean_source = re.sub(r'\s+', '', source_text)
                    clean_check = re.sub(r'\s+', '', clean_value)
                    
                    if clean_check not in clean_source:
                        result["hallucinated_items"].append(value)
                        result["has_hallucination"] = True
            
            if result["total_items"] > 0:
                result["hallucination_rate"] = len(result["hallucinated_items"]) / result["total_items"]
            
            if result["has_hallucination"]:
                self.stats["hallucination_count"] += 1
                self.stats["hallucinated_entities"].extend(result["hallucinated_items"][:3])  # 只保留前3个
            
        except Exception as e:
            logger.debug(f"Hallucination detection error: {e}")
        
        return result
    
    def get_hallucination_rate(self) -> float:
        """获取整体幻觉率"""
        if self.stats["total_checked"] == 0:
            return 0.0
        return self.stats["hallucination_count"] / self.stats["total_checked"]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_checked": self.stats["total_checked"],
            "hallucination_count": self.stats["hallucination_count"],
            "hallucination_rate": self.get_hallucination_rate(),
            "sample_hallucinated_entities": self.stats["hallucinated_entities"][-10:]
        }
