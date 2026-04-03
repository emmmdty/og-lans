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
import argparse
import hashlib
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
from oglans.config import ConfigManager
from oglans.utils.json_parser import (
    NORMALIZATION_VERSION,
    PARSER_VERSION,
    POSTPROCESS_VERSION,
    RobustJSONParser,
    compute_postprocess_metric_summary,
    parse_event_list_strict_with_diagnostics,
    postprocess_event_list,
    write_postprocess_diagnostics_sidecar,
)
from oglans.data.prompt_builder import (
    ChinesePromptBuilder,
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
)
from oglans.inference.cat_lite import apply_cat_lite_pipeline, perturb_text_for_counterfactual
from oglans.utils.eval_protocol import (
    canonicalize_pred_roles as shared_canonicalize_pred_roles,
    load_eval_protocol as shared_load_eval_protocol,
    load_role_alias_map as shared_load_role_alias_map,
    resolve_primary_metric_value,
    validate_primary_metric,
)
from oglans.utils.scv import evaluate_scv_lite
from oglans.utils.run_manifest import (
    build_contract_record,
    build_run_manifest,
    collect_runtime_manifest,
    compute_file_sha256,
    save_json,
)
from oglans.utils.model_quantization import is_quantized_model
from oglans.utils.hub_runtime import (
    build_unsloth_from_pretrained_kwargs,
    configure_model_download_runtime,
    get_model_download_runtime_snapshot,
    resolve_model_name_or_path,
)
from oglans.utils.model_profile import (
    load_local_model_profile,
    prepare_tokenizer_for_profile,
    resolve_profile_terminator_token_ids,
)


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
    parse_raw_success: int = 0
    parse_repair_success: int = 0
    parse_extraction_failures: int = 0
    parse_raw_success_rate: float = 0.0
    parse_repair_success_rate: float = 0.0
    parse_extraction_failure_rate: float = 0.0
    
    # 幻觉检测指标
    hallucination_rate: float = 0.0  # 包含幻觉的样本比例
    hallucination_entity_rate: float = 0.0  # 幻觉实体占比
    
    # CoT 忠实度指标
    cot_faithfulness: float = 0.0  # CoT 推理与 JSON 输出的一致性
    cot_type_consistency: float = 0.0  # 事件类型一致性
    cot_argument_consistency: float = 0.0  # 论元一致性
    cot_checked: int = 0
    cot_skipped: int = 0
    cot_parse_fail: int = 0
    cot_coverage_rate: float = 0.0
    cot_counterfactual_checked: int = 0
    cot_counterfactual_pass_rate: float = 0.0
    
    # Schema 符合度
    schema_compliance_rate: float = 0.0  # 输出符合 schema 的比例
    
    # 详细错误分析
    error_breakdown: Dict = field(default_factory=dict)
    hallucination_breakdown: Dict = field(default_factory=dict)
    schema_violation_breakdown: Dict = field(default_factory=dict)


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
    
    DEFAULT_METRIC_SETTINGS: Dict[str, Any] = {
        "version": "2.0",
        "report_level": "core_plus_diagnostics",  # core_only | core_plus_diagnostics
        "cot": {
            "enabled": False,
            "mode": "strict_span",  # strict_span | weak_mention
            "require_thought_block": False,
            "eval_mode": "self_consistency",  # self_consistency | counterfactual
            "counterfactual": {
                "enabled": False,
                "num_perturb": 1,
                "target_types": ["number", "date", "org"],
            },
        },
        "relaxed": {
            "match_mode": "include_or_char_overlap",  # include_or_char_overlap | span_iou
            "char_overlap_threshold": 0.5,
            "span_iou_threshold": 0.5,
        },
        "hallucination": {
            "match_mode": "normalized_substring",  # normalized_substring | exact_span
        },
        "schema": {
            "mode": "schema_strict",  # syntax_only | schema_strict
        },
    }

    def __init__(
        self,
        relaxed_match_threshold: float = 0.5,
        metric_settings: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化评估器
        
        Args:
            relaxed_match_threshold: Relaxed 模式的最小重叠比例
        """
        self.relaxed_threshold = relaxed_match_threshold
        self.metric_settings = self._merge_metric_settings(metric_settings or {})
        self.json_parser = RobustJSONParser()
        
        # 统计数据
        self.reset()

    @classmethod
    def _deep_merge_dict(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def _merge_metric_settings(cls, override: Dict[str, Any]) -> Dict[str, Any]:
        return cls._deep_merge_dict(cls.DEFAULT_METRIC_SETTINGS, override or {})
    
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
            "parse_raw_success": 0,
            "parse_repair_success": 0,
            "parse_extraction_failures": 0,
            
            # 错误类型分布
            "error_types": defaultdict(int),
            
            # 幻觉检测
            "hallucination_samples": 0,
            "total_entities": 0,
            "hallucinated_entities": 0,
            "hallucination_types": defaultdict(int),
            
            # CoT 忠实度
            "cot_checked": 0,
            "cot_skipped": 0,
            "cot_parse_fail": 0,
            "cot_type_consistent": 0,
            "cot_argument_consistent": 0,
            "cot_fully_consistent": 0,
            "cot_counterfactual_checked": 0,
            "cot_counterfactual_pass": 0,
            
            # Schema 符合度
            "schema_compliant": 0,
            "schema_violations": defaultdict(int),
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

    @staticmethod
    def _extract_thought_text(full_response: str, require_block: bool) -> Tuple[Optional[str], str]:
        """
        提取 thought 文本。

        Returns:
            (thought_text, reason)
            reason: ok | missing_thought | missing_json_boundary
        """
        if not full_response:
            return None, "missing_thought"

        thought_match = re.search(r"<thought>(.*?)</thought>", full_response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            return thought_match.group(1), "ok"

        if require_block:
            return None, "missing_thought"

        json_start = full_response.find("```json")
        if json_start > 0:
            return full_response[:json_start], "ok"
        return None, "missing_json_boundary"

    @staticmethod
    def _longest_common_substring_ratio(a: str, b: str) -> float:
        """基于最长公共子串的近似 Span-IoU。"""
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        dp = [0] * (lb + 1)
        best = 0
        for i in range(1, la + 1):
            prev = 0
            for j in range(1, lb + 1):
                tmp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                    if dp[j] > best:
                        best = dp[j]
                else:
                    dp[j] = 0
                prev = tmp
        den = la + lb - best
        return (best / den) if den > 0 else 0.0

    def _extract_thought_roles(self, thought_text: str) -> Set[Tuple[str, str]]:
        """
        从 thought 文本中抽取 (role, argument) 近似对。
        支持格式: 角色 = "值" / 角色: 值
        """
        if not thought_text:
            return set()
        pairs: Set[Tuple[str, str]] = set()
        for line in thought_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.search(r"([^\s:：=]+)\s*[:：=]\s*[\"“]?([^\"”\n]+)[\"”]?", line)
            if not match:
                continue
            role = self.normalize_text(match.group(1))
            arg = self.normalize_text(match.group(2))
            if role and arg:
                pairs.add((role, arg))
        return pairs
    
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
        
        mode = str(self.metric_settings.get("relaxed", {}).get("match_mode", "include_or_char_overlap"))
        char_thr = float(self.metric_settings.get("relaxed", {}).get("char_overlap_threshold", self.relaxed_threshold))
        span_thr = float(self.metric_settings.get("relaxed", {}).get("span_iou_threshold", self.relaxed_threshold))

        # 完全匹配
        if pred_norm == gold_norm:
            return True

        if mode == "span_iou":
            return self._longest_common_substring_ratio(pred_norm, gold_norm) >= span_thr

        # include_or_char_overlap: 默认兼容旧逻辑
        if pred_norm in gold_norm or gold_norm in pred_norm:
            return True

        pred_chars = set(pred_norm)
        gold_chars = set(gold_norm)
        if not pred_chars or not gold_chars:
            return False
        intersection = len(pred_chars & gold_chars)
        union = len(pred_chars | gold_chars)
        overlap = intersection / union if union > 0 else 0
        return overlap >= char_thr
    
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
    
    def update(
        self,
        pred_events: List[Dict],
        gold_events: List[Dict],
        parse_success: bool = True,
        parse_diagnostics: Optional[Dict[str, Any]] = None,
    ):
        """
        更新评估统计
        
        Args:
            pred_events: 预测的事件列表
            gold_events: 标准事件列表
            parse_success: 解析是否成功
            parse_diagnostics: 解析诊断信息（提取方式、修复步骤等）
        """
        self.stats["total_samples"] += 1

        parse_diagnostics = parse_diagnostics or {}
        repair_steps = parse_diagnostics.get("repair_steps", []) or []
        extraction_method = str(parse_diagnostics.get("extraction_method", ""))
        if not parse_success:
            self.stats["parse_errors"] += 1
            if extraction_method == "no_json_found":
                self.stats["parse_extraction_failures"] += 1
        else:
            if repair_steps:
                self.stats["parse_repair_success"] += 1
            else:
                self.stats["parse_raw_success"] += 1
        
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
        report.parse_raw_success = self.stats["parse_raw_success"]
        report.parse_repair_success = self.stats["parse_repair_success"]
        report.parse_extraction_failures = self.stats["parse_extraction_failures"]
        
        # 解析错误率
        if report.total_samples > 0:
            report.parse_error_rate = report.parse_errors / report.total_samples
            report.parse_raw_success_rate = report.parse_raw_success / report.total_samples
            report.parse_repair_success_rate = report.parse_repair_success / report.total_samples
            report.parse_extraction_failure_rate = report.parse_extraction_failures / report.total_samples
        
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
        report.cot_checked = cot_checked
        report.cot_skipped = self.stats["cot_skipped"]
        report.cot_parse_fail = self.stats["cot_parse_fail"]
        if cot_checked > 0:
            report.cot_faithfulness = self.stats["cot_fully_consistent"] / cot_checked
            report.cot_type_consistency = self.stats["cot_type_consistent"] / cot_checked
            report.cot_argument_consistency = self.stats["cot_argument_consistent"] / cot_checked
        cot_den = cot_checked + report.cot_skipped + report.cot_parse_fail
        if cot_den > 0:
            report.cot_coverage_rate = cot_checked / cot_den
        report.cot_counterfactual_checked = int(self.stats.get("cot_counterfactual_checked", 0))
        if report.cot_counterfactual_checked > 0:
            report.cot_counterfactual_pass_rate = (
                float(self.stats.get("cot_counterfactual_pass", 0))
                / float(report.cot_counterfactual_checked)
            )
        
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
        report.hallucination_breakdown = dict(
            sorted(self.stats["hallucination_types"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        report.schema_violation_breakdown = dict(
            sorted(self.stats["schema_violations"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return report
    
    def check_hallucination(
        self,
        source_text: str,
        pred_events: List[Dict],
    ) -> Tuple[bool, int, int, Dict[str, int]]:
        """
        检测幻觉
        
        Args:
            source_text: 原始输入文本
            pred_events: 预测事件列表
        
        Returns:
            (是否有幻觉, 幻觉实体数, 总实体数, 分类型统计)
        """
        has_hallucination = False
        hallucinated_count = 0
        total_count = 0
        breakdown: Dict[str, int] = defaultdict(int)

        match_mode = str(self.metric_settings.get("hallucination", {}).get("match_mode", "normalized_substring"))
        clean_source = re.sub(r'\s+', '', source_text)
        
        if not isinstance(pred_events, list):
            return False, 0, 0
        
        for event in pred_events:
            if not isinstance(event, dict):
                continue
            
            event_type = str(event.get("event_type", "UNKNOWN"))
            for arg in event.get("arguments", []):
                if not isinstance(arg, dict):
                    continue
                
                argument = str(arg.get("argument", ""))
                if len(argument) < 2:
                    continue
                
                total_count += 1
                clean_arg = re.sub(r'\s+', '', argument)
                role = str(arg.get("role", "UNKNOWN"))
                arg_in_text = False
                if match_mode == "exact_span":
                    arg_in_text = argument in source_text
                else:
                    arg_in_text = clean_arg in clean_source

                if not arg_in_text:
                    has_hallucination = True
                    hallucinated_count += 1
                    breakdown[f"{event_type}|{role}"] += 1

        return has_hallucination, hallucinated_count, total_count, dict(breakdown)
    
    def check_schema_compliance(
        self,
        pred_events: List[Dict],
        valid_event_types: Set[str] = None,
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        检测 Schema 符合度
        
        Args:
            pred_events: 预测事件列表
            valid_event_types: 有效的事件类型集合
            valid_roles_by_event:
                事件类型到合法角色集合的映射（严格模式）。
                若提供，则每个 argument.role 必须属于对应 event_type 的合法角色集合。
        
        Returns:
            (是否符合 Schema, 违规分布)
        """
        violations: Dict[str, int] = defaultdict(int)
        schema_mode = str(self.metric_settings.get("schema", {}).get("mode", "schema_strict"))
        if not isinstance(pred_events, list):
            violations["not_list"] += 1
            return False, dict(violations)
        
        for event in pred_events:
            if not isinstance(event, dict):
                violations["event_not_dict"] += 1
                return False, dict(violations)
            
            # 必须有 event_type
            if "event_type" not in event:
                violations["missing_event_type"] += 1
                return False, dict(violations)

            event_type = event["event_type"]
            # 如果提供了有效事件类型，检查是否匹配
            if (
                schema_mode == "schema_strict"
                and valid_event_types
                and event_type not in valid_event_types
            ):
                violations[f"invalid_event_type:{event_type}"] += 1
                return False, dict(violations)

            # 必须有 arguments 且为列表
            if "arguments" not in event or not isinstance(event.get("arguments"), list):
                violations["missing_or_invalid_arguments"] += 1
                return False, dict(violations)

            allowed_roles = None
            if schema_mode == "schema_strict" and valid_roles_by_event is not None:
                allowed_roles = valid_roles_by_event.get(event_type)
                # 提供了 role schema 时，未知事件类型也视为不合规
                if allowed_roles is None:
                    violations[f"unknown_event_roles:{event_type}"] += 1
                    return False, dict(violations)

            # 每个 argument 必须有 role 和 argument 字段
            for arg in event["arguments"]:
                if not isinstance(arg, dict):
                    violations["argument_not_dict"] += 1
                    return False, dict(violations)
                if "role" not in arg or "argument" not in arg:
                    violations["missing_role_or_argument"] += 1
                    return False, dict(violations)
                if allowed_roles is not None and arg.get("role") not in allowed_roles:
                    violations[f"invalid_role:{event_type}|{arg.get('role')}"] += 1
                    return False, dict(violations)

        return True, {}
    
    def update_with_extended_metrics(
        self, 
        pred_events: List[Dict], 
        gold_events: List[Dict], 
        source_text: str = "",
        full_response: str = "",
        parse_success: bool = True,
        parse_diagnostics: Optional[Dict[str, Any]] = None,
        valid_event_types: Set[str] = None,
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None
    ):
        """
        使用扩展指标更新统计
        
        这是 update() 方法的扩展版本，支持幻觉检测和 CoT 忠实度检测
        """
        # 调用原有的 update
        self.update(
            pred_events,
            gold_events,
            parse_success=parse_success,
            parse_diagnostics=parse_diagnostics,
        )
        
        # === 幻觉检测 ===
        if source_text:
            has_halluc, halluc_count, total_entities, halluc_breakdown = self.check_hallucination(
                source_text,
                pred_events,
            )
            if has_halluc:
                self.stats["hallucination_samples"] += 1
            self.stats["hallucinated_entities"] += halluc_count
            self.stats["total_entities"] += total_entities
            for key, value in halluc_breakdown.items():
                self.stats["hallucination_types"][key] += int(value)
        
        # === Schema 符合度 ===
        schema_ok, violations = self.check_schema_compliance(
            pred_events,
            valid_event_types=valid_event_types,
            valid_roles_by_event=valid_roles_by_event,
        )
        if schema_ok:
            self.stats["schema_compliant"] += 1
        else:
            for key, value in (violations or {}).items():
                self.stats["schema_violations"][key] += int(value)
        
        # === CoT 忠实度检测 ===
        cot_cfg = self.metric_settings.get("cot", {})
        if not cot_cfg.get("enabled", True):
            return
        if not full_response:
            self.stats["cot_skipped"] += 1
            return

        cot_result = self._check_cot_consistency(full_response, pred_events)
        if not cot_result.get("checked", False):
            reason = str(cot_result.get("reason", "skipped"))
            if reason in {"missing_thought", "missing_json_boundary"}:
                self.stats["cot_skipped"] += 1
            else:
                self.stats["cot_parse_fail"] += 1
            return

        self.stats["cot_checked"] += 1
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
            "checked": False,
            "reason": "uninitialized",
            "type_consistent": False,
            "argument_consistent": False,
            "fully_consistent": False,
        }

        cot_cfg = self.metric_settings.get("cot", {})
        cot_mode = str(cot_cfg.get("mode", "strict_span"))
        require_block = bool(cot_cfg.get("require_thought_block", True))

        thought_text, reason = self._extract_thought_text(full_response, require_block=require_block)
        if thought_text is None:
            result["reason"] = reason
            return result

        # 解析 thought 成功
        result["checked"] = True
        result["reason"] = "ok"

        json_event_types = self.extract_event_types(pred_events)
        no_event_clues = ["无事件", "未检测到", "没有检测到", "输出[]", "空数组"]
        no_event_claimed = any(clue in thought_text for clue in no_event_clues)

        thought_event_types = set()
        for etype in json_event_types:
            if etype and etype in thought_text:
                thought_event_types.add(etype)

        if not json_event_types:
            result["type_consistent"] = bool(no_event_claimed or not thought_event_types)
        elif cot_mode == "weak_mention":
            # 弱一致：JSON 中每个类型在 thought 至少出现一次
            result["type_consistent"] = all(etype in thought_text for etype in json_event_types)
        else:
            # 严格一致：事件类型集合必须一致
            result["type_consistent"] = thought_event_types == json_event_types

        pred_pairs = set()
        for event in pred_events or []:
            if not isinstance(event, dict):
                continue
            for arg in event.get("arguments", []):
                if not isinstance(arg, dict):
                    continue
                role = self.normalize_text(arg.get("role", ""))
                value = self.normalize_text(arg.get("argument", ""))
                if role and value:
                    pred_pairs.add((role, value))

        thought_pairs = self._extract_thought_roles(thought_text)
        if not pred_pairs:
            result["argument_consistent"] = (not thought_pairs) or no_event_claimed
        elif cot_mode == "weak_mention":
            # 弱一致：role 级覆盖
            pred_roles = {r for r, _ in pred_pairs}
            thought_roles = {r for r, _ in thought_pairs}
            result["argument_consistent"] = pred_roles.issubset(thought_roles) if pred_roles else True
        else:
            # 严格一致：必须包含全部 pred 对
            result["argument_consistent"] = pred_pairs.issubset(thought_pairs)

        result["fully_consistent"] = bool(result["type_consistent"] and result["argument_consistent"])
        return result

    def update_counterfactual_consistency(
        self,
        perturbed_pred_events: Optional[List[Dict[str, Any]]],
        perturbation: Optional[Dict[str, Any]],
    ) -> None:
        """
        Counterfactual 一致性统计：
        若输入被替换(old->new)，期望预测论元出现 new 且不再出现 old。
        """
        p = perturbation or {}
        if not p.get("changed", False):
            return
        old_value = self.normalize_text(str(p.get("old_value", "")))
        new_value = self.normalize_text(str(p.get("new_value", "")))
        if not old_value or not new_value:
            return

        self.stats["cot_counterfactual_checked"] += 1
        pred_events = perturbed_pred_events if isinstance(perturbed_pred_events, list) else []
        args_values: Set[str] = set()
        for event in pred_events:
            if not isinstance(event, dict):
                continue
            for arg in event.get("arguments", []) or []:
                if not isinstance(arg, dict):
                    continue
                value = self.normalize_text(arg.get("argument", ""))
                if value:
                    args_values.add(value)

        if (new_value in args_values) and (old_value not in args_values):
            self.stats["cot_counterfactual_pass"] += 1


# ===========================
# 3. 评估脚本主逻辑
# ===========================

def load_eval_protocol(path: Optional[str]) -> Dict[str, Any]:
    return shared_load_eval_protocol(path)


def load_role_alias_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    return shared_load_role_alias_map(path)


def canonicalize_pred_roles(
    pred_events: List[Dict[str, Any]],
    alias_map: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    return shared_canonicalize_pred_roles(pred_events, alias_map)


def safe_compute_file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    return compute_file_sha256(path)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OG-LANS 评估脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--protocol",
        type=str,
        default="configs/eval_protocol.yaml",
        help="评估协议文件（主指标与统计规范）",
    )
    parser.add_argument("--base_only", action="store_true", help="评估纯基座模型（不加载 LoRA adapter）")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="覆盖配置中的基座模型路径（用于 base-only 对照组）",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint 路径（LoRA 评估必填）")
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
        "--cot_eval_mode",
        type=str,
        default=None,
        choices=["self_consistency", "counterfactual"],
        help="CoT 评测模式（默认读取 protocol.metrics.cot.eval_mode）",
    )
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        default=None,
        choices=["e2e", "cat_lite"],
        help="推理流水线模式（默认读取 config.inference.pipeline_mode）",
    )
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
    return parser.parse_args(argv)


def validate_eval_args(args) -> None:
    if getattr(args, "base_only", False):
        if getattr(args, "checkpoint", None):
            raise ValueError("参数冲突：--base_only 模式下不应传入 --checkpoint。")
        return
    if not getattr(args, "checkpoint", None):
        raise ValueError("缺少 --checkpoint：LoRA 评估模式必须提供 checkpoint。")


def infer_dataset_name_for_eval(
    config: Dict[str, Any], checkpoint_path: Optional[str] = None
) -> str:
    if checkpoint_path:
        try:
            path_parts = os.path.normpath(checkpoint_path).split(os.sep)
            idx = path_parts.index("checkpoints")
            dataset_name = path_parts[idx - 1]
            if dataset_name == "debug":
                return "DuEE-Fin"
            if dataset_name:
                return dataset_name
        except (ValueError, IndexError):
            pass

    taxonomy_path = (
        config.get("algorithms", {})
        .get("ds_cns", {})
        .get("taxonomy_path")
    )
    if taxonomy_path:
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(str(taxonomy_path))))
        if dataset_name:
            return dataset_name

    return "DuEE-Fin"


def optional_abspath(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


def get_local_model_path(m_cfg: dict, *, project_root: str) -> str:
    """获取本地模型路径（本地路径优先，其次显式配置的模型源）"""
    return resolve_model_name_or_path(
        m_cfg["base_model"],
        source=m_cfg.get("source", "modelscope"),
        project_root=project_root,
    )


def resolve_eval_model_path(
    model_override: Optional[str],
    m_cfg: dict,
    *,
    project_root: str,
) -> str:
    """统一评测入口的模型解析逻辑，确保 CLI override 也走共享 resolver。"""
    model_candidate = model_override or m_cfg["base_model"]
    return resolve_model_name_or_path(
        model_candidate,
        source=m_cfg.get("source", "modelscope"),
        project_root=project_root,
    )


def print_metrics_report(report: MetricsReport, eval_mode: str = "both"):
    """打印格式化的评估报告"""
    print("\n" + "=" * 60)
    print("📊 OG-LANS 评估报告")
    print("=" * 60)
    
    print(f"\n📈 样本统计")
    print(f"   总样本数: {report.total_samples}")
    print(f"   解析失败: {report.parse_errors} ({report.parse_error_rate:.2%})")
    print(f"   原生解析成功: {report.parse_raw_success} ({report.parse_raw_success_rate:.2%})")
    print(f"   修复后解析成功: {report.parse_repair_success} ({report.parse_repair_success_rate:.2%})")
    print(f"   JSON提取失败: {report.parse_extraction_failures} ({report.parse_extraction_failure_rate:.2%})")
    
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
    print(f"   CoT 覆盖率:      {report.cot_coverage_rate:.4f} (checked={report.cot_checked}, skipped={report.cot_skipped}, parse_fail={report.cot_parse_fail})")
    if report.cot_counterfactual_checked > 0:
        print(
            "   CoT 反事实一致性: "
            f"{report.cot_counterfactual_pass_rate:.4f} "
            f"(checked={report.cot_counterfactual_checked})"
        )
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
    validate_eval_args(args)
    run_start_ts = time.time()
    cmdline = " ".join(os.sys.argv)
    repo_dir = os.path.dirname(os.path.abspath(__file__))

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

    # 1. 加载配置（支持 extends 继承与运行时默认值）
    config = ConfigManager().load_config(args.config)
    evaluation_mode = str(config.get("evaluation", {}).get("mode", "")).strip().lower()
    if evaluation_mode not in {"scored", "prediction_only"}:
        raise ValueError(
            f"Unsupported evaluation.mode: {evaluation_mode}. "
            "Expected one of scored, prediction_only."
        )
    model_source = str(config.get("model", {}).get("source", "modelscope"))
    model_profile = load_local_model_profile(config["model"]["profile"])
    model_runtime = configure_model_download_runtime(repo_dir, source=model_source)
    protocol = load_eval_protocol(args.protocol)
    comparison_cfg = config.get("comparison", {})
    if args.report_primary_metric is None:
        args.report_primary_metric = str(protocol.get("primary_metric", "strict_f1"))
    args.report_primary_metric = validate_primary_metric(args.report_primary_metric)
    if args.canonical_metric_mode is None:
        args.canonical_metric_mode = str(protocol.get("canonical_metric_mode", "analysis_only"))
    if args.canonical_metric_mode not in {"off", "analysis_only", "apply_for_aux_metric"}:
        raise ValueError(f"Unsupported canonical metric mode: {args.canonical_metric_mode}")
    metric_settings = protocol.get("metrics", {})
    if args.cot_eval_mode is None:
        args.cot_eval_mode = str(
            metric_settings.get("cot", {}).get("eval_mode", "self_consistency")
        )
    if args.cot_eval_mode not in {"self_consistency", "counterfactual"}:
        raise ValueError(f"Unsupported cot_eval_mode: {args.cot_eval_mode}")
    if args.pipeline_mode is None:
        args.pipeline_mode = str(config.get("inference", {}).get("pipeline_mode", "e2e"))
    if args.pipeline_mode not in {"e2e", "cat_lite"}:
        raise ValueError(f"Unsupported pipeline_mode: {args.pipeline_mode}")
    metric_settings.setdefault("cot", {})
    metric_settings["cot"]["eval_mode"] = args.cot_eval_mode

    # 2. 路径解析
    checkpoint_path = os.path.normpath(args.checkpoint) if args.checkpoint else None
    dataset_name = infer_dataset_name_for_eval(config, checkpoint_path=checkpoint_path)

    dataset_name_lower = dataset_name.lower().replace("-", "_")
    schema_path = f"./data/raw/{dataset_name}/{dataset_name_lower}_event_schema.json"
    data_path = f"./data/raw/{dataset_name}"
    
    # 创建输出目录（每次运行独立目录，避免覆盖）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.split}_seed{args.seed}_p{os.getpid()}"
    eval_task_name = "eval_base" if args.base_only else "eval_checkpoint"
    eval_output_dir = f"./logs/{dataset_name}/{eval_task_name}/{run_id}"
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
    runtime_manifest = collect_runtime_manifest(
        repo_dir,
        package_names=["torch", "transformers", "trl", "unsloth", "dirtyjson", "PyYAML"],
    )
    runtime_manifest["model_runtime"] = get_model_download_runtime_snapshot(source=model_source)
    config_hash = compute_file_sha256(args.config)

    print(f"📊 数据集: {dataset_name} | 划分: {args.split}")
    print(f"🧪 评估模式: {'Base-only Control' if args.base_only else 'LoRA Fine-tuned'}")
    print(f"📂 Schema: {schema_path}")
    print(f"🆔 Run ID: {run_id}")
    print(f"💾 结果保存至: {final_output_path}")
    print(f"📜 Protocol: {args.protocol}")
    print(f"🎯 Primary Metric: {args.report_primary_metric}")
    print(f"🧭 Canonical Metric Mode: {args.canonical_metric_mode}")
    print(f"🧠 CoT Eval Mode: {args.cot_eval_mode}")
    print(f"🧩 Pipeline Mode: {args.pipeline_mode}")
    print(f"🧪 Metric Spec Version: {metric_settings.get('version', '2.0')}")
    print(
        f"📦 Model Runtime: source={model_source} "
        + (
            f"cache={model_runtime.get('MODELSCOPE_CACHE')}"
            if model_source == "modelscope"
            else (
                f"disable_xet={model_runtime.get('HF_HUB_DISABLE_XET')} "
                f"download_timeout={model_runtime.get('HF_HUB_DOWNLOAD_TIMEOUT')} "
                f"etag_timeout={model_runtime.get('HF_HUB_ETAG_TIMEOUT')}"
            )
        )
    )

    # 3. 加载模型
    print("\n🔄 加载模型...")
    base_model_path = resolve_eval_model_path(
        args.model_name_or_path,
        config['model'],
        project_root=repo_dir,
    )
    contract = build_contract_record(
        model_profile=model_profile.name,
        model_source=model_source,
        effective_model_path=str(base_model_path),
    )
    load_in_4bit = config['model'].get('load_in_4bit', True)
    if device == "cpu" and load_in_4bit:
        raise RuntimeError(
            "CPU inference with load_in_4bit=true is not supported in official evaluation. "
            "Set model.load_in_4bit=false or use CUDA."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        **build_unsloth_from_pretrained_kwargs(
            model_name=base_model_path,
            max_seq_length=config['model'].get('max_seq_length', 4096),
            dtype=None,
            load_in_4bit=load_in_4bit,
            source=model_source,
            attn_implementation=config['model'].get('attn_implementation'),
        )
    )
    if args.base_only:
        adapter_loaded = False
        print("ℹ️ Base-only 对照评估：跳过 LoRA adapter 加载。")
    else:
        # [修复] 正确的加载顺序：先加载 adapter，再切换推理模式
        model.load_adapter(args.checkpoint)
        adapter_loaded = True
    FastLanguageModel.for_inference(model)
    model_variant = "base_only" if args.base_only else "lora_finetuned"
    model_quantized = bool(load_in_4bit) or is_quantized_model(model)
    if model_quantized:
        model_device_strategy = "auto_from_pretrained"
        print("ℹ️ 检测到量化模型，跳过 model.to(device)（由 from_pretrained 自动放置设备）。")
    else:
        model_device_strategy = "manual_to_device"
        model.to(device)
    model.eval()  # 显式设置为评估模式
    print(f"🖥️ 推理设备: {device}")

    tokenizer = prepare_tokenizer_for_profile(tokenizer, model_profile, mode="eval")
    terminator_token_ids = resolve_profile_terminator_token_ids(tokenizer, model_profile)
    if not terminator_token_ids:
        raise RuntimeError(
            f"Local model profile {model_profile.name} did not resolve any generation terminator token ids."
        )
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
    has_gold_labels = any(bool(getattr(s, "events", [])) for s in all_samples)
    if evaluation_mode == "scored" and not has_gold_labels:
        raise ValueError(
            f"evaluation.mode=scored requires gold labels, but split={args.split} has no gold event_list."
        )
    valid_types = set(adapter.get_event_types()) if hasattr(adapter, 'get_event_types') else None
    valid_roles_by_event = None
    if hasattr(adapter, 'schema') and isinstance(adapter.schema, dict):
        valid_roles_by_event = {
            etype: set(roles or [])
            for etype, roles in adapter.schema.items()
        }
    cf_cfg = metric_settings.get("cot", {}).get("counterfactual", {})
    cf_target_types = cf_cfg.get("target_types", ["number", "date", "org"])
    cf_num_perturb = max(1, int(cf_cfg.get("num_perturb", 1)))

    # 5. 初始化评估器和解析器
    evaluator = AcademicEventEvaluator(metric_settings=metric_settings)
    role_alias_map = load_role_alias_map(args.role_alias_map)
    inference_cfg = config.get("inference", {})
    postprocess_cfg = dict(inference_cfg.get("postprocess", {}))
    scv_lite_cfg = dict(inference_cfg.get("scv_lite", {}))
    canonical_enabled = bool(args.canonical_metric_mode != "off" and role_alias_map)
    if args.canonical_metric_mode != "off" and not role_alias_map:
        raise ValueError(
            "canonical_metric_mode requires a valid role alias map; no semantic fallback is allowed. "
            f"path={args.role_alias_map}"
        )
    canonical_evaluator = AcademicEventEvaluator(metric_settings=metric_settings) if canonical_enabled else None
    canonical_rewrites_total = 0

    results_to_save = []
    diagnostics_to_save = []
    scv_lite_call_count = 0
    scv_lite_total_seconds = 0.0

    # 6. 批量推理
    decoding_strategy = "采样解码 (Sampling)" if args.do_sample else "确定性解码 (Greedy)"
    print(f"\n🚀 开始推理 (Batch Size: {args.batch_size}, 解码策略: {decoding_strategy})...")
    pbar = tqdm(range(0, len(all_samples), args.batch_size), desc="评估进度")

    for i in pbar:
        batch_samples = all_samples[i:i + args.batch_size]
        batch_prompts = []

        for sample in batch_samples:
            prompt_payload = build_inference_prompt_payload(
                text=sample.text,
                tokenizer=tokenizer,
                use_oneshot=args.use_oneshot,
                schema=getattr(adapter, "schema", None),
                num_examples=3,
            )
            batch_prompts.append(prompt_payload["formatted_text"])

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
                "eos_token_id": (
                    terminator_token_ids[0]
                    if len(terminator_token_ids) == 1
                    else terminator_token_ids
                ),
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
            
            pred_events, parse_diagnostics = parse_event_list_strict_with_diagnostics(response)
            parse_success = parse_diagnostics.get("success", False)
            postprocess_diagnostics = {
                "enabled": bool(postprocess_cfg.get("enabled", False)),
                "changed": False,
                "grounding_breakdown": {},
                "grounded_arguments": 0,
                "ungrounded_arguments": 0,
                "alias_rewrites": 0,
                "illegal_roles_removed": 0,
                "duplicate_splits": 0,
                "argument_diagnostics": [],
                "event_diagnostics": [],
                "scv_lite_triggered": False,
                "scv_lite_reasons": [],
            }
            if postprocess_cfg.get("enabled", False):
                pred_events, postprocess_diagnostics = postprocess_event_list(
                    pred_events,
                    source_text=sample.text,
                    schema=getattr(adapter, "schema", None),
                    role_alias_map=role_alias_map,
                    config=postprocess_cfg,
                )

            scv_lite_decision = evaluate_scv_lite(
                postprocess_diagnostics,
                mode=scv_lite_cfg.get("mode", "off"),
                source_text=sample.text,
                pred_events=pred_events,
            )
            scv_lite_call_count += scv_lite_decision.call_count
            scv_lite_total_seconds += scv_lite_decision.wall_clock_seconds
            postprocess_diagnostics["scv_lite_triggered"] = scv_lite_decision.triggered
            postprocess_diagnostics["scv_lite_reasons"] = list(scv_lite_decision.reasons)

            cat_result = None
            if args.pipeline_mode == "cat_lite":
                cat_result = apply_cat_lite_pipeline(
                    pred_events=pred_events,
                    source_text=sample.text,
                    schema=getattr(adapter, "schema", None),
                    require_argument_in_text=True,
                )
                pred_events = cat_result.events
            
            # 解析 Ground Truth
            # [修复] 优先使用已解析的 events 字段，避免解析失败影响指标
            if hasattr(sample, 'events') and sample.events:
                gold_events = sample.events
            else:
                gold_events, _ = parse_event_list_strict_with_diagnostics(sample.chosen)

            if evaluation_mode == "scored":
                evaluator.update_with_extended_metrics(
                    pred_events=pred_events, 
                    gold_events=gold_events, 
                    source_text=sample.text,
                    full_response=response,
                    parse_success=parse_success,
                    parse_diagnostics=parse_diagnostics,
                    valid_event_types=valid_types,
                    valid_roles_by_event=valid_roles_by_event
                )
            canonical_pred_events = pred_events
            rewrite_count = 0
            if canonical_evaluator is not None and evaluation_mode == "scored":
                canonical_pred_events, rewrite_count = canonicalize_pred_roles(pred_events, role_alias_map)
                canonical_rewrites_total += rewrite_count
                canonical_evaluator.update_with_extended_metrics(
                    pred_events=canonical_pred_events,
                    gold_events=gold_events,
                    source_text=sample.text,
                    full_response=response,
                    parse_success=parse_success,
                    parse_diagnostics=parse_diagnostics,
                    valid_event_types=valid_types,
                    valid_roles_by_event=valid_roles_by_event,
                )

            if args.cot_eval_mode == "counterfactual" and bool(cf_cfg.get("enabled", True)):
                for _ in range(cf_num_perturb):
                    perturbed_text, perturbation = perturb_text_for_counterfactual(
                        sample.text,
                        target_types=cf_target_types,
                    )
                    if not perturbation.get("changed", False):
                        continue
                    cf_payload = build_inference_prompt_payload(
                        text=perturbed_text,
                        tokenizer=tokenizer,
                        use_oneshot=args.use_oneshot,
                        schema=getattr(adapter, "schema", None),
                        num_examples=3,
                    )
                    cf_prompt = cf_payload["formatted_text"]
                    cf_inputs = tokenizer(
                        [cf_prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=config['model'].get('max_seq_length', 4096),
                    ).to(device)
                    with torch.no_grad():
                        cf_outputs = model.generate(**cf_inputs, **generate_kwargs)
                    cf_ids = cf_outputs[:, cf_inputs.input_ids.shape[1]:]
                    cf_response = tokenizer.batch_decode(cf_ids, skip_special_tokens=True)[0]
                    cf_events, _ = parse_event_list_strict_with_diagnostics(cf_response)
                    if args.pipeline_mode == "cat_lite":
                        cf_cat_result = apply_cat_lite_pipeline(
                            pred_events=cf_events,
                            source_text=perturbed_text,
                            schema=getattr(adapter, "schema", None),
                            require_argument_in_text=True,
                        )
                        cf_events = cf_cat_result.events
                    if evaluation_mode == "scored":
                        evaluator.update_counterfactual_consistency(cf_events, perturbation)
                    if canonical_evaluator is not None and evaluation_mode == "scored":
                        cf_events_canonical, _ = canonicalize_pred_roles(cf_events, role_alias_map)
                        canonical_evaluator.update_counterfactual_consistency(
                            cf_events_canonical,
                            perturbation,
                        )
            
            # 保存结果
            results_to_save.append({
                "id": sample.id,
                "text_preview": sample.text[:200] + "..." if len(sample.text) > 200 else sample.text,
                "ground_truth": gold_events,
                "prediction": pred_events,
                "prediction_canonical": canonical_pred_events if canonical_enabled else None,
                "canonical_role_rewrites": rewrite_count if canonical_enabled else 0,
                "pipeline_mode": args.pipeline_mode,
                "cat_lite_kept_events": (cat_result.kept_events if cat_result else None),
                "cat_lite_dropped_events": (cat_result.dropped_events if cat_result else None),
                "cot_eval_mode": args.cot_eval_mode,
                "raw_response": response[:1000] if len(response) > 1000 else response,
                "parse_success": parse_success,
                "parse_method": parse_diagnostics.get("extraction_method", "unknown"),
                "repair_steps": parse_diagnostics.get("repair_steps", []),
                "postprocess_changed": postprocess_diagnostics.get("changed", False),
                "alias_rewrites": postprocess_diagnostics.get("alias_rewrites", 0),
                "illegal_roles_removed": postprocess_diagnostics.get("illegal_roles_removed", 0),
                "duplicate_splits": postprocess_diagnostics.get("duplicate_splits", 0),
                "grounding_summary": postprocess_diagnostics.get("grounding_breakdown", {}),
                "scv_lite_triggered": scv_lite_decision.triggered,
                "scv_lite_reasons": list(scv_lite_decision.reasons),
            })
            diagnostics_to_save.append({
                "id": sample.id,
                "split": args.split,
                "pipeline_mode": args.pipeline_mode,
                "parse_success": parse_success,
                "parse_diagnostics": parse_diagnostics,
                "postprocess_diagnostics": postprocess_diagnostics,
                "argument_diagnostics": postprocess_diagnostics.get("argument_diagnostics", []),
                "event_diagnostics": postprocess_diagnostics.get("event_diagnostics", []),
                "grounding_breakdown": postprocess_diagnostics.get("grounding_breakdown", {}),
                "scv_lite_triggered": scv_lite_decision.triggered,
                "scv_lite_reasons": list(scv_lite_decision.reasons),
                "scv_lite_mode": scv_lite_decision.mode,
                "scv_lite_call_count": scv_lite_decision.call_count,
                "scv_lite_wall_clock_seconds": round(scv_lite_decision.wall_clock_seconds, 6),
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

    diagnostics_sidecar_path = None
    if postprocess_cfg.get("enabled", False) and postprocess_cfg.get("sidecar_diagnostics", True):
        diagnostics_sidecar_path = write_postprocess_diagnostics_sidecar(
            final_output_path.replace(".jsonl", "_diagnostics.jsonl"),
            diagnostics_to_save,
        )

    wall_clock_seconds = round(time.time() - run_start_ts, 4)
    parse_success = report.total_samples - report.parse_errors
    parse_success_rate = (parse_success / report.total_samples) if report.total_samples > 0 else 0.0
    canonical_report = canonical_evaluator.compute_metrics() if canonical_evaluator is not None else None
    postprocess_metric_summary = compute_postprocess_metric_summary(
        diagnostics_to_save,
        scv_call_count=scv_lite_call_count,
        scv_total_seconds=scv_lite_total_seconds,
        total_runtime_seconds=wall_clock_seconds,
    )

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
            "metric_version": metric_settings.get("version", "2.0"),
            "command": cmdline,
            "config_path": os.path.abspath(args.config),
            "config_hash_sha256": config_hash,
            "protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "protocol_hash_sha256": safe_compute_file_sha256(args.protocol),
            "protocol_version": protocol.get("version"),
            "metric_version": metric_settings.get("version", "2.0"),
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "cot_eval_mode": args.cot_eval_mode,
            "pipeline_mode": args.pipeline_mode,
            "metric_settings": metric_settings,
            "role_alias_map_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_map_hash_sha256": safe_compute_file_sha256(args.role_alias_map),
            "checkpoint": optional_abspath(args.checkpoint),
            "model_variant": model_variant,
            "adapter_loaded": bool(adapter_loaded),
            "adapter_path": optional_abspath(args.checkpoint),
            "base_model_name_or_path": str(base_model_path),
            "base_model_override": bool(args.model_name_or_path),
            "output_file": os.path.abspath(final_output_path),
            "diagnostics_sidecar_file": os.path.abspath(diagnostics_sidecar_path) if diagnostics_sidecar_path else None,
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
            "raw_success": report.parse_raw_success,
            "raw_success_rate": round(report.parse_raw_success_rate, 4),
            "repair_success": report.parse_repair_success,
            "repair_success_rate": round(report.parse_repair_success_rate, 4),
            "extraction_failures": report.parse_extraction_failures,
            "extraction_failure_rate": round(report.parse_extraction_failure_rate, 4),
        },
        "hallucination": {
            "sample_rate": round(report.hallucination_rate, 4),
            "entity_rate": round(report.hallucination_entity_rate, 4)
        },
        "cot_faithfulness": {
            "overall": round(report.cot_faithfulness, 4),
            "type_consistency": round(report.cot_type_consistency, 4),
            "argument_consistency": round(report.cot_argument_consistency, 4),
            "coverage_rate": round(report.cot_coverage_rate, 4),
            "checked": report.cot_checked,
            "skipped": report.cot_skipped,
            "parse_fail": report.cot_parse_fail,
            "counterfactual_checked": report.cot_counterfactual_checked,
            "counterfactual_pass_rate": round(report.cot_counterfactual_pass_rate, 4),
        },
        "schema_compliance_rate": round(report.schema_compliance_rate, 4),
        "grounding_rate": postprocess_metric_summary["grounding_rate"],
        "ungrounded_argument_rate": postprocess_metric_summary["ungrounded_argument_rate"],
        "scv_lite_trigger_count": postprocess_metric_summary["scv_lite_trigger_count"],
        "scv_lite_triggered_samples": postprocess_metric_summary["scv_lite_triggered_samples"],
        "scv_call_count": postprocess_metric_summary["scv_call_count"],
        "scv_wall_clock_ratio": postprocess_metric_summary["scv_wall_clock_ratio"],
        "error_breakdown": report.error_breakdown,
        "hallucination_breakdown": report.hallucination_breakdown,
        "schema_violation_breakdown": report.schema_violation_breakdown,
        "primary_metric": args.report_primary_metric,
        "primary_metric_value": round(
            resolve_primary_metric_value(
                {
                    "strict_f1": report.strict_f1,
                    "relaxed_f1": report.relaxed_f1,
                    "type_f1": report.type_f1,
                },
                args.report_primary_metric,
            ),
            4,
        ),
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

    prompt_schema = getattr(adapter, "schema", None)
    prompt_schema_block = ChinesePromptBuilder.build_schema_constraints(prompt_schema)
    selected_fewshot_examples = (
        ChinesePromptBuilder.select_fewshot_examples(num_examples=3) if args.use_oneshot else []
    )
    prompt_hashes = {
        "system_prompt_sha256": hash_text(ChinesePromptBuilder.build_system_prompt(schema=prompt_schema)),
        "schema_constraints_sha256": hash_text(prompt_schema_block) if prompt_schema_block else None,
        "fewshot_example_indices": (
            list(range(min(3, len(ChinesePromptBuilder.FEW_SHOT_EXAMPLES)))) if args.use_oneshot else []
        ),
        "fewshot_examples_sha256": (
            [
                {
                    "user": hash_text(ex["user"]),
                    "assistant": hash_text(ex["assistant"]),
                }
                for ex in selected_fewshot_examples
            ]
            if args.use_oneshot else []
        ),
    }

    # 新版统一摘要结构（与 evaluate_api.py 对齐）
    summary_file = final_output_path.replace(".jsonl", "_summary.json")
    eval_summary = {
        "meta": {
            "run_id": run_id,
            "run_dir": os.path.abspath(artifact_dir),
            "timestamp": timestamp,
            "model": str(base_model_path),
            "api_response_models": [],
            "dataset": dataset_name,
            "num_samples": report.total_samples,
            "split": args.split,
            "concurrency": None,
            "has_gold_labels": True,
            "use_fewshot": bool(args.use_oneshot),
            "fewshot_num_examples": 1 if args.use_oneshot else 0,
            "prompt_style": "profile_contract",
            "json_mode": "off",
            "seed": args.seed,
            "evaluation_mode": evaluation_mode,
            "config_hash_sha256": config_hash,
            "config_path": os.path.abspath(args.config),
            "command": cmdline,
            "bootstrap_samples": None,
            "compute_ci": False,
            "protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "protocol_hash_sha256": safe_compute_file_sha256(args.protocol),
            "eval_protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "eval_protocol_hash": safe_compute_file_sha256(args.protocol),
            "protocol_version": protocol.get("version"),
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "cot_eval_mode": args.cot_eval_mode,
            "pipeline_mode": args.pipeline_mode,
            "role_alias_map_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_map_hash_sha256": safe_compute_file_sha256(args.role_alias_map),
            "role_alias_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_hash": safe_compute_file_sha256(args.role_alias_map),
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
            "decode_mode": "sampling" if args.do_sample else "deterministic_greedy",
            "seed_effective": bool(args.do_sample),
            "model_quantized": bool(model_quantized),
            "model_device_strategy": model_device_strategy,
            "model_target_device": device,
            "model_profile": model_profile.name,
            "model_source": model_source,
            "model_variant": model_variant,
            "adapter_loaded": bool(adapter_loaded),
            "adapter_path": optional_abspath(args.checkpoint),
            "base_model_name_or_path": str(base_model_path),
            "base_model_override": bool(args.model_name_or_path),
            "control_group_tag": f"{model_profile.name}_base_local" if args.base_only else None,
            "prompt_hashes": prompt_hashes,
            "prompt_variant": "fewshot" if args.use_oneshot else "zeroshot",
            "prompt_builder_version": str(comparison_cfg.get("prompt_builder_version", PROMPT_BUILDER_VERSION)),
            "parser_version": str(comparison_cfg.get("parser_version", PARSER_VERSION)),
            "normalization_version": str(comparison_cfg.get("normalization_version", NORMALIZATION_VERSION)),
            "postprocess_enabled": bool(postprocess_cfg.get("enabled", False)),
            "postprocess_version": POSTPROCESS_VERSION if postprocess_cfg.get("enabled", False) else None,
            "postprocess_diagnostics_file": os.path.abspath(diagnostics_sidecar_path) if diagnostics_sidecar_path else None,
            "scv_lite_mode": str(scv_lite_cfg.get("mode", "off")),
            "training_mode": str(config.get("training", {}).get("mode", "preference")),
            "checkpoint": optional_abspath(args.checkpoint),
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
            "total_samples": report.total_samples,
            "parse_errors": report.parse_errors,
            "parse_error_rate": round(report.parse_error_rate, 4),
            "parse_success": parse_success,
            "parse_failure": report.parse_errors,
            "parse_success_rate": round(parse_success_rate, 4),
            "parse_raw_success": report.parse_raw_success,
            "parse_raw_success_rate": round(report.parse_raw_success_rate, 4),
            "parse_repair_success": report.parse_repair_success,
            "parse_repair_success_rate": round(report.parse_repair_success_rate, 4),
            "parse_extraction_failures": report.parse_extraction_failures,
            "parse_extraction_failure_rate": round(report.parse_extraction_failure_rate, 4),
            "hallucination_rate": round(report.hallucination_rate, 4),
            "hallucination_entity_rate": round(report.hallucination_entity_rate, 4),
            "hallucination_breakdown": report.hallucination_breakdown,
            "cot_faithfulness": round(report.cot_faithfulness, 4),
            "cot_type_consistency": round(report.cot_type_consistency, 4),
            "cot_argument_consistency": round(report.cot_argument_consistency, 4),
            "cot_coverage_rate": round(report.cot_coverage_rate, 4),
            "cot_checked": report.cot_checked,
            "cot_skipped": report.cot_skipped,
            "cot_parse_fail": report.cot_parse_fail,
            "cot_counterfactual_checked": report.cot_counterfactual_checked,
            "cot_counterfactual_pass_rate": round(report.cot_counterfactual_pass_rate, 4),
            "schema_compliance_rate": round(report.schema_compliance_rate, 4),
            "grounding_rate": postprocess_metric_summary["grounding_rate"],
            "ungrounded_argument_rate": postprocess_metric_summary["ungrounded_argument_rate"],
            "scv_lite_trigger_count": postprocess_metric_summary["scv_lite_trigger_count"],
            "scv_lite_triggered_samples": postprocess_metric_summary["scv_lite_triggered_samples"],
            "scv_call_count": postprocess_metric_summary["scv_call_count"],
            "scv_wall_clock_ratio": postprocess_metric_summary["scv_wall_clock_ratio"],
            "schema_violation_breakdown": report.schema_violation_breakdown,
            "error_breakdown": report.error_breakdown,
            "bootstrap_ci": None,
            "primary_metric": args.report_primary_metric,
            "primary_metric_value": round(
                resolve_primary_metric_value(
                    {
                        "strict_f1": report.strict_f1,
                        "relaxed_f1": report.relaxed_f1,
                        "type_f1": report.type_f1,
                    },
                    args.report_primary_metric,
                ),
                4,
            ),
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
            "wall_clock_seconds": wall_clock_seconds,
        },
        "runtime_manifest": runtime_manifest,
        "analysis": {
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "canonical_metrics_available": canonical_report is not None,
            "metric_version": metric_settings.get("version", "2.0"),
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
        task=eval_task_name,
        status="completed",
        meta=eval_summary["meta"],
        artifacts={
            "run_dir": os.path.abspath(artifact_dir),
            "result_file": os.path.abspath(final_output_path),
            "metrics_file": os.path.abspath(metrics_file),
            "summary_file": os.path.abspath(summary_file),
            "diagnostics_sidecar_file": os.path.abspath(diagnostics_sidecar_path) if diagnostics_sidecar_path else None,
        },
        contract=contract,
        runtime=eval_summary["runtime"],
        runtime_manifest=runtime_manifest,
    )
    save_json(run_manifest_path, run_manifest)

    print(f"   结果文件: {final_output_path}")
    print(f"   指标文件: {metrics_file}")
    print(f"   摘要文件: {summary_file}")
    if diagnostics_sidecar_path:
        print(f"   诊断文件: {diagnostics_sidecar_path}")
    print(f"   运行清单: {run_manifest_path}")
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
