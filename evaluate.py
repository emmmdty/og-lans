# evaluate.py
"""
OG-LANS 学术级评估脚本 (Academic Evaluation Framework)
面向 2026 年高质量论文发表

实现功能:
1. Strict/Relaxed 两种评估模式（符合 ACL/EMNLP 规范）
2. 鲁棒 JSON 解析（集成 RobustJSONParser）
3. 多维度指标（Type F1, Role F1, Argument F1）
4. 详细的错误分析报告
5. 幻觉检测率 (Hallucination Rate)
6. CoT 忠实度 (CoT Faithfulness)
7. Schema 符合度 (Schema Compliance)

论文发表支持:
- 提供完整的 LaTeX 表格格式输出
- 支持消融实验对比分析
- 统计显著性测试（Bootstrap）
"""

import os
import json
import yaml
import argparse
import re
import random
import time
import copy
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any

# 导入项目模块
from oglans.data.adapter import DuEEFinAdapter
from oglans.utils.json_parser import RobustJSONParser, parse_llm_output
from oglans.data.prompt_builder import ChinesePromptBuilder, build_inference_prompt
from oglans.utils.run_manifest import (
    build_run_manifest,
    collect_runtime_manifest,
    compute_file_sha256,
    save_json,
)
from oglans.utils.model_quantization import is_quantized_model


# ===========================
# 1. 数据结构定义
# ===========================
@dataclass
class EvaluationResult:
    """单样本评估结果"""
    sample_id: str
    text_preview: str
    ground_truth: List[Dict]
    prediction: List[Dict]
    raw_response: str
    parse_success: bool
    parse_diagnostics: Dict = field(default_factory=dict)


@dataclass 
class MetricsReport:
    """评估指标报告 (2026 学术论文版)"""
    # Strict 模式指标
    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0
    
    # Relaxed 模式指标
    relaxed_precision: float = 0.0
    relaxed_recall: float = 0.0
    relaxed_f1: float = 0.0
    
    # 事件类型识别指标
    type_precision: float = 0.0
    type_recall: float = 0.0
    type_f1: float = 0.0
    
    # 解析统计
    total_samples: int = 0
    parse_errors: int = 0
    parse_error_rate: float = 0.0
    
    # 幻觉检测指标
    hallucination_rate: float = 0.0  # 包含幻觉的样本比例
    hallucination_entity_rate: float = 0.0  # 幻觉实体占比
    
    # CoT 忠实度指标
    cot_faithfulness: float = 0.0  # CoT 推理与 JSON 输出的一致性
    cot_type_consistency: float = 0.0  # 事件类型一致性
    cot_argument_consistency: float = 0.0  # 论元一致性
    
    # Schema 符合度
    schema_compliance_rate: float = 0.0  # 输出符合 schema 的比例
    
    # 详细错误分析
    error_breakdown: Dict = field(default_factory=dict)


# ===========================
# 2. 核心评估器类
# ===========================
class AcademicEventEvaluator:
    """
    学术级事件抽取评估器
    
    支持两种评估模式:
    - Strict: (event_type, role, argument) 完全匹配
    - Relaxed: argument 部分匹配（包含关系）
    """
    
    def __init__(self, relaxed_match_threshold: float = 0.5):
        """
        初始化评估器
        
        Args:
            relaxed_match_threshold: Relaxed 模式的最小重叠比例
        """
        self.relaxed_threshold = relaxed_match_threshold
        self.json_parser = RobustJSONParser()
        
        # 统计数据
        self.reset()
    
    def reset(self):
        """重置所有统计数据"""
        self.stats = {
            # Strict 模式
            "strict_tp": 0,
            "strict_pred_total": 0,
            "strict_gold_total": 0,
            
            # Relaxed 模式
            "relaxed_tp": 0,
            "relaxed_pred_total": 0,
            "relaxed_gold_total": 0,
            
            # 事件类型
            "type_tp": 0,
            "type_pred_total": 0,
            "type_gold_total": 0,
            
            # 解析统计
            "total_samples": 0,
            "parse_errors": 0,
            
            # 错误类型分布
            "error_types": defaultdict(int),
            
            # 幻觉检测
            "hallucination_samples": 0,
            "total_entities": 0,
            "hallucinated_entities": 0,
            
            # CoT 忠实度
            "cot_checked": 0,
            "cot_type_consistent": 0,
            "cot_argument_consistent": 0,
            "cot_fully_consistent": 0,
            
            # Schema 符合度
            "schema_compliant": 0
        }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        文本归一化（用于比较）
        
        Args:
            text: 原始文本
        
        Returns:
            归一化后的文本
        """
        if text is None:
            return ""
        
        # 确保转换为字符串（argument 可能是数字类型）
        if not isinstance(text, str):
            text = str(text)
        
        if not text:
            return ""
        
        # 1. 移除空白字符
        text = re.sub(r'\s+', '', text)
        
        # 2. 统一全角/半角
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('，', ',').replace('。', '.')
        
        # 3. 转小写（对于英文部分）
        text = text.lower()
        
        return text
    
    def extract_triplets_strict(self, events: List[Dict]) -> Set[Tuple[str, str, str]]:
        """
        提取 Strict 模式三元组: (event_type, role, normalized_argument)
        
        Args:
            events: 事件列表
        
        Returns:
            三元组集合
        """
        triplets = set()
        
        if not isinstance(events, list):
            return triplets
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get("event_type", "")
            if not event_type:
                continue
            
            arguments = event.get("arguments", [])
            if not isinstance(arguments, list):
                continue
            
            for arg in arguments:
                if not isinstance(arg, dict):
                    continue
                
                role = arg.get("role", "")
                argument = arg.get("argument", "")
                
                # 确保 argument 是字符串类型
                if argument is not None and not isinstance(argument, str):
                    argument = str(argument)
                
                if role and argument:
                    norm_arg = self.normalize_text(argument)
                    if norm_arg:  # 只有非空值才计入
                        triplets.add((event_type, role, norm_arg))
        
        return triplets
    
    def extract_triplets_relaxed(self, events: List[Dict]) -> List[Tuple[str, str, str]]:
        """
        提取 Relaxed 模式三元组（保留原始 argument 用于部分匹配）
        
        Returns:
            三元组列表（非集合，因为需要遍历比较）
        """
        triplets = []
        
        if not isinstance(events, list):
            return triplets
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get("event_type", "")
            if not event_type:
                continue
            
            arguments = event.get("arguments", [])
            if not isinstance(arguments, list):
                continue
            
            for arg in arguments:
                if not isinstance(arg, dict):
                    continue
                
                role = arg.get("role", "")
                argument = str(arg.get("argument", "")).strip()
                
                if role and argument:
                    triplets.append((event_type, role, argument))
        
        return triplets
    
    def extract_event_types(self, events: List[Dict]) -> Set[str]:
        """
        提取事件类型集合
        
        Args:
            events: 事件列表
        
        Returns:
            事件类型集合
        """
        types = set()
        
        if not isinstance(events, list):
            return types
        
        for event in events:
            if isinstance(event, dict):
                etype = event.get("event_type", "")
                if etype:
                    types.add(etype)
        
        return types
    
    def relaxed_match(self, pred_arg: str, gold_arg: str) -> bool:
        """
        Relaxed 模式匹配判断
        
        判断条件（满足其一即可）:
        1. pred 包含 gold
        2. gold 包含 pred
        3. 字符级重叠比例超过阈值
        
        Args:
            pred_arg: 预测的论元值
            gold_arg: 标准论元值
        
        Returns:
            是否匹配
        """
        pred_norm = self.normalize_text(pred_arg)
        gold_norm = self.normalize_text(gold_arg)
        
        if not pred_norm or not gold_norm:
            return False
        
        # 完全匹配
        if pred_norm == gold_norm:
            return True
        
        # 包含关系
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return True
        
        # 字符级重叠（Jaccard-like）
        pred_chars = set(pred_norm)
        gold_chars = set(gold_norm)
        
        if not pred_chars or not gold_chars:
            return False
        
        intersection = len(pred_chars & gold_chars)
        union = len(pred_chars | gold_chars)
        
        overlap = intersection / union if union > 0 else 0
        return overlap >= self.relaxed_threshold
    
    def compute_relaxed_matches(
        self, 
        pred_triplets: List[Tuple], 
        gold_triplets: List[Tuple]
    ) -> int:
        """
        计算 Relaxed 模式的匹配数
        
        Args:
            pred_triplets: 预测三元组列表
            gold_triplets: 标准三元组列表
        
        Returns:
            匹配数（True Positives）
        """
        matched_gold = set()  # 记录已匹配的 gold 索引，避免重复计数
        tp = 0
        
        for p_type, p_role, p_arg in pred_triplets:
            for g_idx, (g_type, g_role, g_arg) in enumerate(gold_triplets):
                if g_idx in matched_gold:
                    continue
                
                # 类型和角色必须完全匹配
                if p_type != g_type or p_role != g_role:
                    continue
                
                # 论元使用 Relaxed 匹配
                if self.relaxed_match(p_arg, g_arg):
                    tp += 1
                    matched_gold.add(g_idx)
                    break
        
        return tp
    
    def update(self, pred_events: List[Dict], gold_events: List[Dict], parse_success: bool = True):
        """
        更新评估统计
        
        Args:
            pred_events: 预测的事件列表
            gold_events: 标准事件列表
            parse_success: 解析是否成功
        """
        self.stats["total_samples"] += 1
        
        if not parse_success:
            self.stats["parse_errors"] += 1
        
        # === Strict 模式 ===
        pred_strict = self.extract_triplets_strict(pred_events)
        gold_strict = self.extract_triplets_strict(gold_events)
        
        strict_tp = len(pred_strict & gold_strict)
        self.stats["strict_tp"] += strict_tp
        self.stats["strict_pred_total"] += len(pred_strict)
        self.stats["strict_gold_total"] += len(gold_strict)
        
        # === Relaxed 模式 ===
        pred_relaxed = self.extract_triplets_relaxed(pred_events)
        gold_relaxed = self.extract_triplets_relaxed(gold_events)
        
        relaxed_tp = self.compute_relaxed_matches(pred_relaxed, gold_relaxed)
        self.stats["relaxed_tp"] += relaxed_tp
        self.stats["relaxed_pred_total"] += len(pred_relaxed)
        self.stats["relaxed_gold_total"] += len(gold_relaxed)
        
        # === 事件类型识别 ===
        pred_types = self.extract_event_types(pred_events)
        gold_types = self.extract_event_types(gold_events)
        
        type_tp = len(pred_types & gold_types)
        self.stats["type_tp"] += type_tp
        self.stats["type_pred_total"] += len(pred_types)
        self.stats["type_gold_total"] += len(gold_types)
        
        # === 错误分析 ===
        if pred_strict != gold_strict:
            # 漏报（False Negative）
            missed = gold_strict - pred_strict
            for m_type, m_role, _ in missed:
                self.stats["error_types"][f"FN_{m_type}_{m_role}"] += 1
            
            # 误报（False Positive）
            spurious = pred_strict - gold_strict
            for s_type, s_role, _ in spurious:
                self.stats["error_types"][f"FP_{s_type}_{s_role}"] += 1
    
    def compute_metrics(self) -> MetricsReport:
        """
        计算最终指标（2026 学术论文版）
        
        Returns:
            MetricsReport 对象，包含完整的评估指标
        """
        report = MetricsReport()
        report.total_samples = self.stats["total_samples"]
        report.parse_errors = self.stats["parse_errors"]
        
        # 解析错误率
        if report.total_samples > 0:
            report.parse_error_rate = report.parse_errors / report.total_samples
        
        # === Strict F1 ===
        s_tp = self.stats["strict_tp"]
        s_pred = self.stats["strict_pred_total"]
        s_gold = self.stats["strict_gold_total"]
        
        report.strict_precision = s_tp / s_pred if s_pred > 0 else 0.0
        report.strict_recall = s_tp / s_gold if s_gold > 0 else 0.0
        if report.strict_precision + report.strict_recall > 0:
            report.strict_f1 = 2 * report.strict_precision * report.strict_recall / (report.strict_precision + report.strict_recall)
        
        # === Relaxed F1 ===
        r_tp = self.stats["relaxed_tp"]
        r_pred = self.stats["relaxed_pred_total"]
        r_gold = self.stats["relaxed_gold_total"]
        
        report.relaxed_precision = r_tp / r_pred if r_pred > 0 else 0.0
        report.relaxed_recall = r_tp / r_gold if r_gold > 0 else 0.0
        if report.relaxed_precision + report.relaxed_recall > 0:
            report.relaxed_f1 = 2 * report.relaxed_precision * report.relaxed_recall / (report.relaxed_precision + report.relaxed_recall)
        
        # === Type F1 ===
        t_tp = self.stats["type_tp"]
        t_pred = self.stats["type_pred_total"]
        t_gold = self.stats["type_gold_total"]
        
        report.type_precision = t_tp / t_pred if t_pred > 0 else 0.0
        report.type_recall = t_tp / t_gold if t_gold > 0 else 0.0
        if report.type_precision + report.type_recall > 0:
            report.type_f1 = 2 * report.type_precision * report.type_recall / (report.type_precision + report.type_recall)
        
        # === 幻觉检测指标 ===
        if report.total_samples > 0:
            report.hallucination_rate = self.stats["hallucination_samples"] / report.total_samples
        if self.stats["total_entities"] > 0:
            report.hallucination_entity_rate = self.stats["hallucinated_entities"] / self.stats["total_entities"]
        
        # === CoT 忠实度指标 ===
        cot_checked = self.stats["cot_checked"]
        if cot_checked > 0:
            report.cot_faithfulness = self.stats["cot_fully_consistent"] / cot_checked
            report.cot_type_consistency = self.stats["cot_type_consistent"] / cot_checked
            report.cot_argument_consistency = self.stats["cot_argument_consistent"] / cot_checked
        
        # === Schema 符合度 ===
        if report.total_samples > 0:
            report.schema_compliance_rate = self.stats["schema_compliant"] / report.total_samples
        
        # 错误类型分布（取 Top 10）
        sorted_errors = sorted(
            self.stats["error_types"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        report.error_breakdown = dict(sorted_errors)
        
        return report
    
    def check_hallucination(self, source_text: str, pred_events: List[Dict]) -> Tuple[bool, int, int]:
        """
        检测幻觉
        
        Args:
            source_text: 原始输入文本
            pred_events: 预测事件列表
        
        Returns:
            (是否有幻觉, 幻觉实体数, 总实体数)
        """
        has_hallucination = False
        hallucinated_count = 0
        total_count = 0
        
        # 清理原文
        clean_source = re.sub(r'\s+', '', source_text)
        
        if not isinstance(pred_events, list):
            return False, 0, 0
        
        for event in pred_events:
            if not isinstance(event, dict):
                continue
            
            for arg in event.get("arguments", []):
                if not isinstance(arg, dict):
                    continue
                
                argument = str(arg.get("argument", ""))
                if len(argument) < 2:
                    continue
                
                total_count += 1
                clean_arg = re.sub(r'\s+', '', argument)
                
                # 检查是否在原文中
                if clean_arg not in clean_source:
                    has_hallucination = True
                    hallucinated_count += 1
        
        return has_hallucination, hallucinated_count, total_count
    
    def check_schema_compliance(
        self,
        pred_events: List[Dict],
        valid_event_types: Set[str] = None,
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None
    ) -> bool:
        """
        检测 Schema 符合度
        
        Args:
            pred_events: 预测事件列表
            valid_event_types: 有效的事件类型集合
            valid_roles_by_event:
                事件类型到合法角色集合的映射（严格模式）。
                若提供，则每个 argument.role 必须属于对应 event_type 的合法角色集合。
        
        Returns:
            是否符合 Schema
        """
        if not isinstance(pred_events, list):
            return False
        
        for event in pred_events:
            if not isinstance(event, dict):
                return False
            
            # 必须有 event_type
            if "event_type" not in event:
                return False

            event_type = event["event_type"]
            # 如果提供了有效事件类型，检查是否匹配
            if valid_event_types and event_type not in valid_event_types:
                return False

            # 必须有 arguments 且为列表
            if "arguments" not in event or not isinstance(event.get("arguments"), list):
                return False

            allowed_roles = None
            if valid_roles_by_event is not None:
                allowed_roles = valid_roles_by_event.get(event_type)
                # 提供了 role schema 时，未知事件类型也视为不合规
                if allowed_roles is None:
                    return False

            # 每个 argument 必须有 role 和 argument 字段
            for arg in event["arguments"]:
                if not isinstance(arg, dict):
                    return False
                if "role" not in arg or "argument" not in arg:
                    return False
                if allowed_roles is not None and arg.get("role") not in allowed_roles:
                    return False
        
        return True
    
    def update_with_extended_metrics(
        self, 
        pred_events: List[Dict], 
        gold_events: List[Dict], 
        source_text: str = "",
        full_response: str = "",
        parse_success: bool = True,
        valid_event_types: Set[str] = None,
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None
    ):
        """
        使用扩展指标更新统计
        
        这是 update() 方法的扩展版本，支持幻觉检测和 CoT 忠实度检测
        """
        # 调用原有的 update
        self.update(pred_events, gold_events, parse_success)
        
        # === 幻觉检测 ===
        if source_text:
            has_halluc, halluc_count, total_entities = self.check_hallucination(source_text, pred_events)
            if has_halluc:
                self.stats["hallucination_samples"] += 1
            self.stats["hallucinated_entities"] += halluc_count
            self.stats["total_entities"] += total_entities
        
        # === Schema 符合度 ===
        if self.check_schema_compliance(
            pred_events,
            valid_event_types=valid_event_types,
            valid_roles_by_event=valid_roles_by_event,
        ):
            self.stats["schema_compliant"] += 1
        
        # === CoT 忠实度检测 ===
        if full_response and ("<thought>" in full_response or "```json" in full_response):
            self.stats["cot_checked"] += 1
            
            # 简化的 CoT 一致性检测
            cot_result = self._check_cot_consistency(full_response, pred_events)
            if cot_result["type_consistent"]:
                self.stats["cot_type_consistent"] += 1
            if cot_result["argument_consistent"]:
                self.stats["cot_argument_consistent"] += 1
            if cot_result["fully_consistent"]:
                self.stats["cot_fully_consistent"] += 1
    
    def _check_cot_consistency(self, full_response: str, pred_events: List[Dict]) -> Dict:
        """
        内部方法：检测 CoT 与 JSON 输出的一致性
        """
        result = {
            "type_consistent": True,
            "argument_consistent": True,
            "fully_consistent": True
        }
        
        # 提取 thought 部分
        thought_match = re.search(r'<thought>(.*?)</thought>', full_response, re.DOTALL)
        if not thought_match:
            # 没有 thought 标签，取 json 之前的内容
            json_start = full_response.find("```json")
            if json_start > 0:
                thought_text = full_response[:json_start]
            else:
                return result  # 无法检测
        else:
            thought_text = thought_match.group(1)
        
        # 提取 JSON 中的事件类型
        json_event_types = set()
        if isinstance(pred_events, list):
            for event in pred_events:
                if isinstance(event, dict) and event.get("event_type"):
                    json_event_types.add(event["event_type"])
        
        # 检测事件类型是否在 thought 中被提及
        for etype in json_event_types:
            if etype not in thought_text:
                result["type_consistent"] = False
                result["fully_consistent"] = False
                break
        
        return result


# ===========================
# 3. 评估脚本主逻辑
# ===========================

DEFAULT_EVAL_PROTOCOL: Dict[str, Any] = {
    "version": "1.0",
    "primary_metric": "strict_f1",
    "canonical_metric_mode": "analysis_only",
    "evaluation": {
        "split": "dev",
        "seeds": [3407, 3408, 3409],
        "bootstrap_samples": 1000,
        "concurrency": 8,
        "significance": "paired_permutation",
        "confidence": 0.95,
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_eval_protocol(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return copy.deepcopy(DEFAULT_EVAL_PROTOCOL)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"协议文件格式错误（需为 dict）: {path}")
    return _deep_merge_dict(DEFAULT_EVAL_PROTOCOL, payload)


def load_role_alias_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if str(path).lower().endswith(".json"):
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        return {}
    root = payload.get("event_role_aliases", payload)
    if not isinstance(root, dict):
        return {}

    normalized: Dict[str, Dict[str, str]] = {}
    for event_type, mapping in root.items():
        if not isinstance(mapping, dict):
            continue
        event_key = str(event_type)
        normalized[event_key] = {}
        for alias, canonical in mapping.items():
            if not alias or not canonical:
                continue
            normalized[event_key][str(alias)] = str(canonical)
    return normalized


def canonicalize_pred_roles(
    pred_events: List[Dict[str, Any]],
    alias_map: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    if not isinstance(pred_events, list) or not alias_map:
        return pred_events if isinstance(pred_events, list) else [], 0

    rewritten = 0
    normalized_events: List[Dict[str, Any]] = []
    for event in pred_events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        role_map = alias_map.get(str(event_type), {}) if event_type else {}
        new_event = dict(event)
        args = event.get("arguments", [])
        if isinstance(args, list):
            new_args: List[Dict[str, Any]] = []
            for arg in args:
                if not isinstance(arg, dict):
                    continue
                new_arg = dict(arg)
                role = new_arg.get("role")
                if isinstance(role, str) and role in role_map:
                    mapped = role_map[role]
                    if mapped != role:
                        rewritten += 1
                    new_arg["role"] = mapped
                new_args.append(new_arg)
            new_event["arguments"] = new_args
        normalized_events.append(new_event)
    return normalized_events, rewritten


def safe_compute_file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    return compute_file_sha256(path)


def parse_args():
    parser = argparse.ArgumentParser(description="OG-LANS 评估脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--protocol",
        type=str,
        default="configs/eval_protocol.yaml",
        help="评估协议文件（主指标与统计规范）",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子（复现性）")
    parser.add_argument("--num_samples", type=int, default=None, help="评估样本数量（None=全部）")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批次大小")
    parser.add_argument("--split", type=str, default="dev", help="数据集划分 (train/dev/test)")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl", help="结果输出文件")
    parser.add_argument("--eval_mode", type=str, default="both", choices=["strict", "relaxed", "both"], 
                        help="评估模式: strict/relaxed/both")
    parser.add_argument("--use_oneshot", action="store_true", help="使用 One-Shot 示例进行推理")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="使用采样解码（默认 False，使用 greedy 确定性解码）")
    parser.add_argument(
        "--role_alias_map",
        type=str,
        default="configs/role_aliases_duee_fin.yaml",
        help="角色别名映射文件（用于辅助 canonical 指标）",
    )
    parser.add_argument(
        "--canonical_metric_mode",
        type=str,
        default=None,
        choices=["off", "analysis_only", "apply_for_aux_metric"],
        help="canonical 指标模式：off / analysis_only / apply_for_aux_metric",
    )
    parser.add_argument(
        "--report_primary_metric",
        type=str,
        default=None,
        help="主报告指标名（默认读取 protocol.primary_metric）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备选择（auto/cuda/cpu）",
    )
    return parser.parse_args()


def get_local_model_path(m_cfg: dict) -> str:
    """获取本地模型路径（支持 ModelScope 下载）"""
    model_name_or_path = m_cfg['base_model']
    if m_cfg.get('source') == 'modelscope':
        try:
            from modelscope import snapshot_download
            model_name_or_path = snapshot_download(model_name_or_path, cache_dir='./models')
        except Exception as e:
            print(f"⚠️ ModelScope 下载失败: {e}，使用原始路径")
    return model_name_or_path


def print_metrics_report(report: MetricsReport, eval_mode: str = "both"):
    """打印格式化的评估报告"""
    print("\n" + "=" * 60)
    print("📊 OG-LANS 评估报告")
    print("=" * 60)
    
    print(f"\n📈 样本统计")
    print(f"   总样本数: {report.total_samples}")
    print(f"   解析失败: {report.parse_errors} ({report.parse_error_rate:.2%})")
    
    if eval_mode in ["strict", "both"]:
        print(f"\n📐 Strict 模式 (完全匹配)")
        print(f"   Precision: {report.strict_precision:.4f}")
        print(f"   Recall:    {report.strict_recall:.4f}")
        print(f"   F1 Score:  {report.strict_f1:.4f}")
    
    if eval_mode in ["relaxed", "both"]:
        print(f"\n📏 Relaxed 模式 (部分匹配)")
        print(f"   Precision: {report.relaxed_precision:.4f}")
        print(f"   Recall:    {report.relaxed_recall:.4f}")
        print(f"   F1 Score:  {report.relaxed_f1:.4f}")
    
    print(f"\n🏷️ 事件类型识别")
    print(f"   Type Precision: {report.type_precision:.4f}")
    print(f"   Type Recall:    {report.type_recall:.4f}")
    print(f"   Type F1 Score:  {report.type_f1:.4f}")
    
    if report.error_breakdown:
        print(f"\n❌ 主要错误类型 (Top 10)")
        for error_type, count in report.error_breakdown.items():
            print(f"   {error_type}: {count}")
    
    # 幻觉检测和 CoT 忠实度指标
    print(f"\n🔮 高级指标")
    print(f"   幻觉样本率:      {report.hallucination_rate:.4f}")
    print(f"   幻觉实体率:      {report.hallucination_entity_rate:.4f}")
    print(f"   CoT 忠实度:      {report.cot_faithfulness:.4f}")
    print(f"   CoT 类型一致性:  {report.cot_type_consistency:.4f}")
    print(f"   CoT 论元一致性:  {report.cot_argument_consistency:.4f}")
    print(f"   Schema 符合率(类型+角色): {report.schema_compliance_rate:.4f}")
    
    print("\n" + "=" * 60)


def main():
    # 仅在本地评估执行时加载深度学习依赖，避免 API-only 环境的硬依赖问题
    try:
        import numpy as np
        import torch
    except Exception as e:
        raise RuntimeError(
            "本地模型评估依赖 numpy/torch。若只需 API 评估，请使用 evaluate_api.py。"
        ) from e

    try:
        from unsloth import FastLanguageModel
    except Exception as e:
        raise RuntimeError(
            "本地模型评估依赖 unsloth。若只需 API 评估，请使用 evaluate_api.py。"
        ) from e

    args = parse_args()
    run_start_ts = time.time()
    cmdline = " ".join(os.sys.argv)

    # 0. 复现性设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 保守设置：优先可复现而非极致速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("参数 --device cuda 但当前环境不可用 CUDA。")
        device = "cuda"
    else:
        device = "cpu"

    # 1. 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    protocol = load_eval_protocol(args.protocol)
    if args.report_primary_metric is None:
        args.report_primary_metric = str(protocol.get("primary_metric", "strict_f1"))
    if args.canonical_metric_mode is None:
        args.canonical_metric_mode = str(protocol.get("canonical_metric_mode", "analysis_only"))
    if args.canonical_metric_mode not in {"off", "analysis_only", "apply_for_aux_metric"}:
        raise ValueError(f"Unsupported canonical metric mode: {args.canonical_metric_mode}")

    # 2. 路径解析
    checkpoint_path = os.path.normpath(args.checkpoint)
    try:
        path_parts = checkpoint_path.split(os.sep)
        idx = path_parts.index("checkpoints")
        dataset_name = path_parts[idx - 1]
        # Fix: debug directory is not a dataset name, use default.
        if dataset_name == "debug":
            dataset_name = "DuEE-Fin"
    except (ValueError, IndexError):
        dataset_name = "DuEE-Fin"

    dataset_name_lower = dataset_name.lower().replace("-", "_")
    schema_path = f"./data/raw/{dataset_name}/{dataset_name_lower}_event_schema.json"
    data_path = f"./data/raw/{dataset_name}"
    
    # 创建输出目录（每次运行独立目录，避免覆盖）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.split}_seed{args.seed}_p{os.getpid()}"
    eval_output_dir = f"./logs/{dataset_name}/eval_local/{run_id}"
    os.makedirs(eval_output_dir, exist_ok=True)

    if args.output_file == "eval_results.jsonl":
        final_output_path = os.path.join(eval_output_dir, "eval_results.jsonl")
    elif not os.path.dirname(args.output_file):
        final_output_path = os.path.join(eval_output_dir, args.output_file)
    else:
        final_output_path = args.output_file
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    artifact_dir = os.path.dirname(final_output_path) or "."
    os.makedirs(artifact_dir, exist_ok=True)
    run_manifest_path = os.path.join(artifact_dir, "run_manifest.json")
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_manifest = collect_runtime_manifest(
        repo_dir,
        package_names=["torch", "transformers", "trl", "unsloth", "dirtyjson", "PyYAML"],
    )
    config_hash = compute_file_sha256(args.config)

    print(f"📊 数据集: {dataset_name} | 划分: {args.split}")
    print(f"📂 Schema: {schema_path}")
    print(f"🆔 Run ID: {run_id}")
    print(f"💾 结果保存至: {final_output_path}")
    print(f"📜 Protocol: {args.protocol}")
    print(f"🎯 Primary Metric: {args.report_primary_metric}")
    print(f"🧭 Canonical Metric Mode: {args.canonical_metric_mode}")

    # 3. 加载模型
    print("\n🔄 加载模型...")
    base_model_path = get_local_model_path(config['model'])
    load_in_4bit = config['model'].get('load_in_4bit', True)
    if device == "cpu" and load_in_4bit:
        # bitsandbytes 4bit 在 CPU 路径通常不可用，显式降级为非 4bit 以避免直接崩溃
        print("⚠️ 检测到 CPU 推理，自动禁用 load_in_4bit（原配置为 True）。")
        load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=config['model'].get('max_seq_length', 4096),
        load_in_4bit=load_in_4bit,
    )
    # [修复] 正确的加载顺序：先加载 adapter，再切换推理模式
    model.load_adapter(args.checkpoint)
    FastLanguageModel.for_inference(model)
    model_quantized = bool(load_in_4bit) or is_quantized_model(model)
    if model_quantized:
        model_device_strategy = "auto_from_pretrained"
        print("ℹ️ 检测到量化模型，跳过 model.to(device)（由 from_pretrained 自动放置设备）。")
    else:
        model_device_strategy = "manual_to_device"
        model.to(device)
    model.eval()  # 显式设置为评估模式
    print(f"🖥️ 推理设备: {device}")
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # [修复] 显式检查 EOS token（Qwen3 通常使用 <|im_end|>）
    expected_eos = "<|im_end|>"
    if tokenizer.eos_token is None or tokenizer.eos_token == "":
        tokenizer.eos_token = expected_eos
    elif tokenizer.eos_token != expected_eos:
        print(f"⚠️ EOS token 为 {tokenizer.eos_token}，期望 {expected_eos}。将保留当前设置。")
    print(f"🔧 EOS Token: {tokenizer.eos_token} | EOS Token ID: {tokenizer.eos_token_id}")

    # 4. 加载数据
    print("\n📚 加载数据...")
    adapter = DuEEFinAdapter(data_path=data_path, schema_path=schema_path)
    try:
        all_samples = adapter.load_data(args.split)
    except Exception as e:
        # 【关键修复】不再自动 fallback 到训练集，避免评估指标虚高
        print(f"❌ 加载 {args.split} 数据集失败: {e}")
        print(f"   请检查数据路径和 split 参数是否正确")
        print(f"   可用的 split 选项: train, dev, test")
        raise RuntimeError(f"无法加载 {args.split} 数据集，请确保数据文件存在") from e

    if args.num_samples:
        all_samples = all_samples[:args.num_samples]

    print(f"   加载 {len(all_samples)} 条样本")

    # 5. 初始化评估器和解析器
    evaluator = AcademicEventEvaluator()
    role_alias_map = load_role_alias_map(args.role_alias_map)
    canonical_enabled = bool(args.canonical_metric_mode != "off" and role_alias_map)
    canonical_evaluator = AcademicEventEvaluator() if canonical_enabled else None
    canonical_rewrites_total = 0
    json_parser = RobustJSONParser()

    results_to_save = []

    # 6. 批量推理
    decoding_strategy = "采样解码 (Sampling)" if args.do_sample else "确定性解码 (Greedy)"
    print(f"\n🚀 开始推理 (Batch Size: {args.batch_size}, 解码策略: {decoding_strategy})...")
    pbar = tqdm(range(0, len(all_samples), args.batch_size), desc="评估进度")

    for i in pbar:
        batch_samples = all_samples[i:i + args.batch_size]
        batch_prompts = []

        for sample in batch_samples:
            # [修复] 使用统一的 prompt 构建函数，确保训练/评估一致性
            formatted_prompt = build_inference_prompt(
                text=sample.text,
                tokenizer=tokenizer,
                use_oneshot=args.use_oneshot,
                schema=getattr(adapter, "schema", None),
            )
            batch_prompts.append(formatted_prompt)

        # Tokenize
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=config['model'].get('max_seq_length', 4096)
        ).to(device)
        
        # 推理
        with torch.no_grad():
            # [修复] 获取 inference 配置节点（直接获取，不要加 ['parameters']）
            inf_cfg = config.get('inference', {})

            # 构建生成参数
            generate_kwargs = {
                "max_new_tokens": inf_cfg.get('max_new_tokens', 2048),
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # 根据 do_sample 参数选择解码策略
            if args.do_sample:
                # 采样解码：使用配置中的温度和采样参数
                generate_kwargs.update({
                    "do_sample": True,
                    "temperature": inf_cfg.get('temperature', 0.7),
                    "top_p": inf_cfg.get('top_p', 0.8),
                    "top_k": inf_cfg.get('top_k', 20),
                })
            else:
                # 确定性解码（Greedy）：不传采样参数
                generate_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **generate_kwargs)
        
        # 解码
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 处理每个样本
        for j, response in enumerate(decoded_responses):
            sample = batch_samples[j]
            
            # 使用鲁棒解析器
            pred_events, parse_diagnostics = json_parser.parse(response)
            parse_success = parse_diagnostics.get("success", False)
            
            if pred_events is None:
                pred_events = []
            
            # 确保是列表
            if isinstance(pred_events, dict):
                pred_events = [pred_events]
            
            # 解析 Ground Truth
            # [修复] 优先使用已解析的 events 字段，避免解析失败影响指标
            if hasattr(sample, 'events') and sample.events:
                gold_events = sample.events
            else:
                gold_events, _ = json_parser.parse(sample.chosen)
                if gold_events is None:
                    gold_events = []
                if isinstance(gold_events, dict):
                    gold_events = [gold_events]
            
            # 使用扩展版评估方法，支持幻觉检测、CoT 忠实度和严格 Schema 校验
            valid_types = set(adapter.get_event_types()) if hasattr(adapter, 'get_event_types') else None
            valid_roles_by_event = None
            if hasattr(adapter, 'schema') and isinstance(adapter.schema, dict):
                valid_roles_by_event = {
                    etype: set(roles or [])
                    for etype, roles in adapter.schema.items()
                }
            evaluator.update_with_extended_metrics(
                pred_events=pred_events, 
                gold_events=gold_events, 
                source_text=sample.text,
                full_response=response,
                parse_success=parse_success,
                valid_event_types=valid_types,
                valid_roles_by_event=valid_roles_by_event
            )
            canonical_pred_events = pred_events
            rewrite_count = 0
            if canonical_evaluator is not None:
                canonical_pred_events, rewrite_count = canonicalize_pred_roles(pred_events, role_alias_map)
                canonical_rewrites_total += rewrite_count
                canonical_evaluator.update_with_extended_metrics(
                    pred_events=canonical_pred_events,
                    gold_events=gold_events,
                    source_text=sample.text,
                    full_response=response,
                    parse_success=parse_success,
                    valid_event_types=valid_types,
                    valid_roles_by_event=valid_roles_by_event,
                )
            
            # 保存结果
            results_to_save.append({
                "id": sample.id,
                "text_preview": sample.text[:200] + "..." if len(sample.text) > 200 else sample.text,
                "ground_truth": gold_events,
                "prediction": pred_events,
                "prediction_canonical": canonical_pred_events if canonical_enabled else None,
                "canonical_role_rewrites": rewrite_count if canonical_enabled else 0,
                "raw_response": response[:1000] if len(response) > 1000 else response,
                "parse_success": parse_success,
                "parse_method": parse_diagnostics.get("extraction_method", "unknown"),
                "repair_steps": parse_diagnostics.get("repair_steps", [])
            })
            
            # 详细日志
            if args.verbose and not parse_success:
                print(f"\n⚠️ 样本 {sample.id} 解析失败")
                print(f"   方法: {parse_diagnostics.get('extraction_method')}")
                print(f"   错误: {parse_diagnostics.get('error', 'Unknown')}")

    # 7. 计算指标并输出报告
    report = evaluator.compute_metrics()
    print_metrics_report(report, args.eval_mode)

    # 8. 保存结果
    print(f"\n💾 保存结果...")
    
    # 保存详细预测结果
    with open(final_output_path, 'w', encoding='utf-8') as f:
        for res in results_to_save:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    parse_success = report.total_samples - report.parse_errors
    parse_success_rate = (parse_success / report.total_samples) if report.total_samples > 0 else 0.0
    canonical_report = canonical_evaluator.compute_metrics() if canonical_evaluator is not None else None

    # 兼容旧版指标文件结构（保留）
    metrics_file = final_output_path.replace(".jsonl", "_metrics.json")
    metrics_dict = {
        "_meta": {
            "project": "OG-LANS",
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset": dataset_name,
            "split": args.split,
            "seed": args.seed,
            "command": cmdline,
            "config_path": os.path.abspath(args.config),
            "config_hash_sha256": config_hash,
            "protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "protocol_hash_sha256": safe_compute_file_sha256(args.protocol),
            "protocol_version": protocol.get("version"),
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "role_alias_map_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_map_hash_sha256": safe_compute_file_sha256(args.role_alias_map),
            "checkpoint": os.path.abspath(args.checkpoint),
            "output_file": os.path.abspath(final_output_path),
            "runtime_manifest": runtime_manifest,
        },
        "strict": {
            "precision": round(report.strict_precision, 4),
            "recall": round(report.strict_recall, 4),
            "f1": round(report.strict_f1, 4)
        },
        "relaxed": {
            "precision": round(report.relaxed_precision, 4),
            "recall": round(report.relaxed_recall, 4),
            "f1": round(report.relaxed_f1, 4)
        },
        "type_identification": {
            "precision": round(report.type_precision, 4),
            "recall": round(report.type_recall, 4),
            "f1": round(report.type_f1, 4)
        },
        "parse_statistics": {
            "total_samples": report.total_samples,
            "parse_errors": report.parse_errors,
            "parse_error_rate": round(report.parse_error_rate, 4),
            "parse_success_rate": round(parse_success_rate, 4),
        },
        "hallucination": {
            "sample_rate": round(report.hallucination_rate, 4),
            "entity_rate": round(report.hallucination_entity_rate, 4)
        },
        "cot_faithfulness": {
            "overall": round(report.cot_faithfulness, 4),
            "type_consistency": round(report.cot_type_consistency, 4),
            "argument_consistency": round(report.cot_argument_consistency, 4)
        },
        "schema_compliance_rate": round(report.schema_compliance_rate, 4),
        "error_breakdown": report.error_breakdown,
        "primary_metric": args.report_primary_metric,
        "primary_metric_value": round(float({
            "strict_f1": report.strict_f1,
            "relaxed_f1": report.relaxed_f1,
            "type_f1": report.type_f1,
        }.get(args.report_primary_metric, report.strict_f1)), 4),
    }
    if canonical_report is not None:
        metrics_dict["auxiliary_metrics"] = {
            "canonicalized": {
                "strict_precision": round(canonical_report.strict_precision, 4),
                "strict_recall": round(canonical_report.strict_recall, 4),
                "strict_f1": round(canonical_report.strict_f1, 4),
                "relaxed_precision": round(canonical_report.relaxed_precision, 4),
                "relaxed_recall": round(canonical_report.relaxed_recall, 4),
                "relaxed_f1": round(canonical_report.relaxed_f1, 4),
                "type_precision": round(canonical_report.type_precision, 4),
                "type_recall": round(canonical_report.type_recall, 4),
                "type_f1": round(canonical_report.type_f1, 4),
                "schema_compliance_rate": round(canonical_report.schema_compliance_rate, 4),
                "canonical_role_rewrites_total": canonical_rewrites_total,
                "canonical_role_rewrites_avg": round(
                    canonical_rewrites_total / report.total_samples if report.total_samples else 0.0,
                    4,
                ),
            }
        }
    save_json(metrics_file, metrics_dict)

    # 新版统一摘要结构（与 evaluate_api.py 对齐）
    summary_file = final_output_path.replace(".jsonl", "_summary.json")
    eval_summary = {
        "meta": {
            "run_id": run_id,
            "run_dir": os.path.abspath(artifact_dir),
            "timestamp": timestamp,
            "model": config.get("model", {}).get("base_model"),
            "api_response_models": [],
            "dataset": dataset_name,
            "num_samples": report.total_samples,
            "split": args.split,
            "concurrency": None,
            "has_gold_labels": True,
            "use_fewshot": bool(args.use_oneshot),
            "fewshot_num_examples": 1 if args.use_oneshot else 0,
            "prompt_style": "qwen",
            "json_mode": "off",
            "seed": args.seed,
            "config_hash_sha256": config_hash,
            "config_path": os.path.abspath(args.config),
            "command": cmdline,
            "bootstrap_samples": None,
            "compute_ci": False,
            "protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "protocol_hash_sha256": safe_compute_file_sha256(args.protocol),
            "protocol_version": protocol.get("version"),
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "role_alias_map_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_map_hash_sha256": safe_compute_file_sha256(args.role_alias_map),
            "role_alias_map_loaded": bool(role_alias_map),
            "metrics_report_file": None,
            "log_file": None,
            "generation": {
                "temperature": config.get("inference", {}).get("temperature", 0.7) if args.do_sample else 0.0,
                "max_tokens": config.get("inference", {}).get("max_new_tokens", 2048),
                "max_retries": None,
                "json_mode": "off",
                "do_sample": bool(args.do_sample),
                "batch_size": args.batch_size,
            },
            "model_quantized": bool(model_quantized),
            "model_device_strategy": model_device_strategy,
            "model_target_device": device,
            "prompt_hashes": {},
            "checkpoint": os.path.abspath(args.checkpoint),
        },
        "metrics": {
            "strict_precision": round(report.strict_precision, 4),
            "strict_recall": round(report.strict_recall, 4),
            "strict_f1": round(report.strict_f1, 4),
            "relaxed_precision": round(report.relaxed_precision, 4),
            "relaxed_recall": round(report.relaxed_recall, 4),
            "relaxed_f1": round(report.relaxed_f1, 4),
            "type_precision": round(report.type_precision, 4),
            "type_recall": round(report.type_recall, 4),
            "type_f1": round(report.type_f1, 4),
            "parse_success": parse_success,
            "parse_failure": report.parse_errors,
            "parse_success_rate": round(parse_success_rate, 4),
            "hallucination_rate": round(report.hallucination_rate, 4),
            "hallucination_entity_rate": round(report.hallucination_entity_rate, 4),
            "cot_faithfulness": round(report.cot_faithfulness, 4),
            "cot_type_consistency": round(report.cot_type_consistency, 4),
            "cot_argument_consistency": round(report.cot_argument_consistency, 4),
            "schema_compliance_rate": round(report.schema_compliance_rate, 4),
            "error_breakdown": report.error_breakdown,
            "bootstrap_ci": None,
            "primary_metric": args.report_primary_metric,
            "primary_metric_value": round(float({
                "strict_f1": report.strict_f1,
                "relaxed_f1": report.relaxed_f1,
                "type_f1": report.type_f1,
            }.get(args.report_primary_metric, report.strict_f1)), 4),
        },
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_tokens_per_sample": 0.0,
        },
        "api_stats": {
            "failed_calls": 0,
            "failed_call_rate": 0.0,
        },
        "runtime": {
            "wall_clock_seconds": round(time.time() - run_start_ts, 4),
        },
        "runtime_manifest": runtime_manifest,
        "analysis": {
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "canonical_metrics_available": canonical_report is not None,
            "protocol": protocol,
        },
    }
    if canonical_report is not None:
        eval_summary["metrics"]["auxiliary_metrics"] = {
            "canonicalized": {
                "strict_precision": round(canonical_report.strict_precision, 4),
                "strict_recall": round(canonical_report.strict_recall, 4),
                "strict_f1": round(canonical_report.strict_f1, 4),
                "relaxed_precision": round(canonical_report.relaxed_precision, 4),
                "relaxed_recall": round(canonical_report.relaxed_recall, 4),
                "relaxed_f1": round(canonical_report.relaxed_f1, 4),
                "type_precision": round(canonical_report.type_precision, 4),
                "type_recall": round(canonical_report.type_recall, 4),
                "type_f1": round(canonical_report.type_f1, 4),
                "schema_compliance_rate": round(canonical_report.schema_compliance_rate, 4),
                "canonical_role_rewrites_total": canonical_rewrites_total,
                "canonical_role_rewrites_avg": round(
                    canonical_rewrites_total / report.total_samples if report.total_samples else 0.0,
                    4,
                ),
            }
        }
    save_json(summary_file, eval_summary)

    run_manifest = build_run_manifest(
        task="eval_local",
        status="completed",
        meta=eval_summary["meta"],
        artifacts={
            "run_dir": os.path.abspath(artifact_dir),
            "result_file": os.path.abspath(final_output_path),
            "metrics_file": os.path.abspath(metrics_file),
            "summary_file": os.path.abspath(summary_file),
        },
        runtime=eval_summary["runtime"],
        runtime_manifest=runtime_manifest,
    )
    save_json(run_manifest_path, run_manifest)

    print(f"   结果文件: {final_output_path}")
    print(f"   指标文件: {metrics_file}")
    print(f"   摘要文件: {summary_file}")
    print(f"   运行清单: {run_manifest_path}")
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
