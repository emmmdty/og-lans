#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared evaluation primitives used by local and API evaluation entrypoints.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils.json_parser import RobustJSONParser


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

    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0

    relaxed_precision: float = 0.0
    relaxed_recall: float = 0.0
    relaxed_f1: float = 0.0

    type_precision: float = 0.0
    type_recall: float = 0.0
    type_f1: float = 0.0

    total_samples: int = 0
    parse_errors: int = 0
    parse_error_rate: float = 0.0
    parse_raw_success: int = 0
    parse_repair_success: int = 0
    parse_extraction_failures: int = 0
    parse_raw_success_rate: float = 0.0
    parse_repair_success_rate: float = 0.0
    parse_extraction_failure_rate: float = 0.0

    hallucination_rate: float = 0.0
    hallucination_entity_rate: float = 0.0

    cot_faithfulness: float = 0.0
    cot_type_consistency: float = 0.0
    cot_argument_consistency: float = 0.0
    cot_checked: int = 0
    cot_skipped: int = 0
    cot_parse_fail: int = 0
    cot_coverage_rate: float = 0.0
    cot_counterfactual_checked: int = 0
    cot_counterfactual_pass_rate: float = 0.0

    schema_compliance_rate: float = 0.0

    error_breakdown: Dict = field(default_factory=dict)
    hallucination_breakdown: Dict = field(default_factory=dict)
    schema_violation_breakdown: Dict = field(default_factory=dict)


class AcademicEventEvaluator:
    """
    学术级事件抽取评估器

    支持两种评估模式:
    - Strict: (event_type, role, argument) 完全匹配
    - Relaxed: argument 部分匹配（包含关系）
    """

    DEFAULT_METRIC_SETTINGS: Dict[str, Any] = {
        "version": "2.0",
        "report_level": "core_plus_diagnostics",
        "cot": {
            "enabled": True,
            "mode": "strict_span",
            "require_thought_block": False,
            "eval_mode": "self_consistency",
            "counterfactual": {
                "enabled": False,
                "num_perturb": 1,
                "target_types": ["number", "date", "org"],
            },
        },
        "relaxed": {
            "match_mode": "include_or_char_overlap",
            "char_overlap_threshold": 0.5,
            "span_iou_threshold": 0.5,
        },
        "hallucination": {
            "match_mode": "normalized_substring",
        },
        "schema": {
            "mode": "schema_strict",
        },
    }

    def __init__(
        self,
        relaxed_match_threshold: float = 0.5,
        metric_settings: Optional[Dict[str, Any]] = None,
    ):
        self.relaxed_threshold = relaxed_match_threshold
        self.metric_settings = self._merge_metric_settings(metric_settings or {})
        self.json_parser = RobustJSONParser()
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
        self.stats = {
            "strict_tp": 0,
            "strict_pred_total": 0,
            "strict_gold_total": 0,
            "relaxed_tp": 0,
            "relaxed_pred_total": 0,
            "relaxed_gold_total": 0,
            "type_tp": 0,
            "type_pred_total": 0,
            "type_gold_total": 0,
            "total_samples": 0,
            "parse_errors": 0,
            "parse_raw_success": 0,
            "parse_repair_success": 0,
            "parse_extraction_failures": 0,
            "error_types": defaultdict(int),
            "hallucination_samples": 0,
            "total_entities": 0,
            "hallucinated_entities": 0,
            "hallucination_types": defaultdict(int),
            "cot_checked": 0,
            "cot_skipped": 0,
            "cot_parse_fail": 0,
            "cot_type_consistent": 0,
            "cot_argument_consistent": 0,
            "cot_fully_consistent": 0,
            "cot_counterfactual_checked": 0,
            "cot_counterfactual_pass": 0,
            "schema_compliant": 0,
            "schema_violations": defaultdict(int),
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""
        text = re.sub(r"\s+", "", text)
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("，", ",").replace("。", ".")
        return text.lower()

    @staticmethod
    def _extract_thought_text(full_response: str, require_block: bool) -> Tuple[Optional[str], str]:
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
                if argument is not None and not isinstance(argument, str):
                    argument = str(argument)
                if role and argument:
                    norm_arg = self.normalize_text(argument)
                    if norm_arg:
                        triplets.add((event_type, role, norm_arg))
        return triplets

    def extract_triplets_relaxed(self, events: List[Dict]) -> List[Tuple[str, str, str]]:
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
        pred_norm = self.normalize_text(pred_arg)
        gold_norm = self.normalize_text(gold_arg)

        if not pred_norm or not gold_norm:
            return False

        mode = str(self.metric_settings.get("relaxed", {}).get("match_mode", "include_or_char_overlap"))
        char_thr = float(self.metric_settings.get("relaxed", {}).get("char_overlap_threshold", self.relaxed_threshold))
        span_thr = float(self.metric_settings.get("relaxed", {}).get("span_iou_threshold", self.relaxed_threshold))

        if pred_norm == gold_norm:
            return True
        if mode == "span_iou":
            return self._longest_common_substring_ratio(pred_norm, gold_norm) >= span_thr
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
        gold_triplets: List[Tuple],
    ) -> int:
        matched_gold = set()
        tp = 0
        for p_type, p_role, p_arg in pred_triplets:
            for g_idx, (g_type, g_role, g_arg) in enumerate(gold_triplets):
                if g_idx in matched_gold:
                    continue
                if p_type != g_type or p_role != g_role:
                    continue
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

        pred_strict = self.extract_triplets_strict(pred_events)
        gold_strict = self.extract_triplets_strict(gold_events)
        strict_tp = len(pred_strict & gold_strict)
        self.stats["strict_tp"] += strict_tp
        self.stats["strict_pred_total"] += len(pred_strict)
        self.stats["strict_gold_total"] += len(gold_strict)

        pred_relaxed = self.extract_triplets_relaxed(pred_events)
        gold_relaxed = self.extract_triplets_relaxed(gold_events)
        relaxed_tp = self.compute_relaxed_matches(pred_relaxed, gold_relaxed)
        self.stats["relaxed_tp"] += relaxed_tp
        self.stats["relaxed_pred_total"] += len(pred_relaxed)
        self.stats["relaxed_gold_total"] += len(gold_relaxed)

        pred_types = self.extract_event_types(pred_events)
        gold_types = self.extract_event_types(gold_events)
        type_tp = len(pred_types & gold_types)
        self.stats["type_tp"] += type_tp
        self.stats["type_pred_total"] += len(pred_types)
        self.stats["type_gold_total"] += len(gold_types)

        if pred_strict != gold_strict:
            missed = gold_strict - pred_strict
            for m_type, m_role, _ in missed:
                self.stats["error_types"][f"FN_{m_type}_{m_role}"] += 1
            spurious = pred_strict - gold_strict
            for s_type, s_role, _ in spurious:
                self.stats["error_types"][f"FP_{s_type}_{s_role}"] += 1

    def compute_metrics(self) -> MetricsReport:
        report = MetricsReport()
        report.total_samples = self.stats["total_samples"]
        report.parse_errors = self.stats["parse_errors"]
        report.parse_raw_success = self.stats["parse_raw_success"]
        report.parse_repair_success = self.stats["parse_repair_success"]
        report.parse_extraction_failures = self.stats["parse_extraction_failures"]

        if report.total_samples > 0:
            report.parse_error_rate = report.parse_errors / report.total_samples
            report.parse_raw_success_rate = report.parse_raw_success / report.total_samples
            report.parse_repair_success_rate = report.parse_repair_success / report.total_samples
            report.parse_extraction_failure_rate = report.parse_extraction_failures / report.total_samples

        s_tp = self.stats["strict_tp"]
        s_pred = self.stats["strict_pred_total"]
        s_gold = self.stats["strict_gold_total"]
        report.strict_precision = s_tp / s_pred if s_pred > 0 else 0.0
        report.strict_recall = s_tp / s_gold if s_gold > 0 else 0.0
        if report.strict_precision + report.strict_recall > 0:
            report.strict_f1 = 2 * report.strict_precision * report.strict_recall / (
                report.strict_precision + report.strict_recall
            )

        r_tp = self.stats["relaxed_tp"]
        r_pred = self.stats["relaxed_pred_total"]
        r_gold = self.stats["relaxed_gold_total"]
        report.relaxed_precision = r_tp / r_pred if r_pred > 0 else 0.0
        report.relaxed_recall = r_tp / r_gold if r_gold > 0 else 0.0
        if report.relaxed_precision + report.relaxed_recall > 0:
            report.relaxed_f1 = 2 * report.relaxed_precision * report.relaxed_recall / (
                report.relaxed_precision + report.relaxed_recall
            )

        t_tp = self.stats["type_tp"]
        t_pred = self.stats["type_pred_total"]
        t_gold = self.stats["type_gold_total"]
        report.type_precision = t_tp / t_pred if t_pred > 0 else 0.0
        report.type_recall = t_tp / t_gold if t_gold > 0 else 0.0
        if report.type_precision + report.type_recall > 0:
            report.type_f1 = 2 * report.type_precision * report.type_recall / (
                report.type_precision + report.type_recall
            )

        if report.total_samples > 0:
            report.hallucination_rate = self.stats["hallucination_samples"] / report.total_samples
            report.schema_compliance_rate = self.stats["schema_compliant"] / report.total_samples
        if self.stats["total_entities"] > 0:
            report.hallucination_entity_rate = self.stats["hallucinated_entities"] / self.stats["total_entities"]

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
            report.cot_counterfactual_pass_rate = float(self.stats.get("cot_counterfactual_pass", 0)) / float(
                report.cot_counterfactual_checked
            )

        report.error_breakdown = dict(
            sorted(self.stats["error_types"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
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
        has_hallucination = False
        hallucinated_count = 0
        total_count = 0
        breakdown: Dict[str, int] = defaultdict(int)

        match_mode = str(self.metric_settings.get("hallucination", {}).get("match_mode", "normalized_substring"))
        clean_source = re.sub(r"\s+", "", source_text)

        if not isinstance(pred_events, list):
            return False, 0, 0, {}
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
                clean_arg = re.sub(r"\s+", "", argument)
                role = str(arg.get("role", "UNKNOWN"))
                arg_in_text = argument in source_text if match_mode == "exact_span" else clean_arg in clean_source
                if not arg_in_text:
                    has_hallucination = True
                    hallucinated_count += 1
                    breakdown[f"{event_type}|{role}"] += 1
        return has_hallucination, hallucinated_count, total_count, dict(breakdown)

    def check_schema_compliance(
        self,
        pred_events: List[Dict],
        valid_event_types: Set[str] = None,
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[bool, Dict[str, int]]:
        violations: Dict[str, int] = defaultdict(int)
        schema_mode = str(self.metric_settings.get("schema", {}).get("mode", "schema_strict"))
        if not isinstance(pred_events, list):
            violations["not_list"] += 1
            return False, dict(violations)

        for event in pred_events:
            if not isinstance(event, dict):
                violations["event_not_dict"] += 1
                return False, dict(violations)
            if "event_type" not in event:
                violations["missing_event_type"] += 1
                return False, dict(violations)

            event_type = event["event_type"]
            if schema_mode == "schema_strict" and valid_event_types and event_type not in valid_event_types:
                violations[f"invalid_event_type:{event_type}"] += 1
                return False, dict(violations)
            if "arguments" not in event or not isinstance(event.get("arguments"), list):
                violations["missing_or_invalid_arguments"] += 1
                return False, dict(violations)

            allowed_roles = None
            if schema_mode == "schema_strict" and valid_roles_by_event is not None:
                allowed_roles = valid_roles_by_event.get(event_type)
                if allowed_roles is None:
                    violations[f"unknown_event_roles:{event_type}"] += 1
                    return False, dict(violations)

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
        valid_roles_by_event: Optional[Dict[str, Set[str]]] = None,
    ):
        self.update(
            pred_events,
            gold_events,
            parse_success=parse_success,
            parse_diagnostics=parse_diagnostics,
        )

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
            result["type_consistent"] = all(etype in thought_text for etype in json_event_types)
        else:
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
            pred_roles = {r for r, _ in pred_pairs}
            thought_roles = {r for r, _ in thought_pairs}
            result["argument_consistent"] = pred_roles.issubset(thought_roles) if pred_roles else True
        else:
            result["argument_consistent"] = pred_pairs.issubset(thought_pairs)

        result["fully_consistent"] = bool(result["type_consistent"] and result["argument_consistent"])
        return result

    def update_counterfactual_consistency(
        self,
        perturbed_pred_events: Optional[List[Dict[str, Any]]],
        perturbation: Optional[Dict[str, Any]],
    ) -> None:
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


def print_metrics_report(report: MetricsReport, eval_mode: str = "both"):
    """打印格式化的评估报告"""
    print("\n" + "=" * 60)
    print("📊 OG-LANS 评估报告")
    print("=" * 60)

    print("\n📈 样本统计")
    print(f"   总样本数: {report.total_samples}")
    print(f"   解析失败: {report.parse_errors} ({report.parse_error_rate:.2%})")
    print(f"   原生解析成功: {report.parse_raw_success} ({report.parse_raw_success_rate:.2%})")
    print(f"   修复后解析成功: {report.parse_repair_success} ({report.parse_repair_success_rate:.2%})")
    print(f"   JSON提取失败: {report.parse_extraction_failures} ({report.parse_extraction_failure_rate:.2%})")

    if eval_mode in ["strict", "both"]:
        print("\n📐 Strict 模式 (完全匹配)")
        print(f"   Precision: {report.strict_precision:.4f}")
        print(f"   Recall:    {report.strict_recall:.4f}")
        print(f"   F1 Score:  {report.strict_f1:.4f}")

    if eval_mode in ["relaxed", "both"]:
        print("\n📏 Relaxed 模式 (部分匹配)")
        print(f"   Precision: {report.relaxed_precision:.4f}")
        print(f"   Recall:    {report.relaxed_recall:.4f}")
        print(f"   F1 Score:  {report.relaxed_f1:.4f}")

    print("\n🏷️ 事件类型识别")
    print(f"   Type Precision: {report.type_precision:.4f}")
    print(f"   Type Recall:    {report.type_recall:.4f}")
    print(f"   Type F1 Score:  {report.type_f1:.4f}")

    if report.error_breakdown:
        print("\n❌ 主要错误类型 (Top 10)")
        for error_type, count in report.error_breakdown.items():
            print(f"   {error_type}: {count}")

    print("\n🔮 高级指标")
    print(f"   幻觉样本率:      {report.hallucination_rate:.4f}")
    print(f"   幻觉实体率:      {report.hallucination_entity_rate:.4f}")
    print(f"   CoT 忠实度:      {report.cot_faithfulness:.4f}")
    print(f"   CoT 类型一致性:  {report.cot_type_consistency:.4f}")
    print(f"   CoT 论元一致性:  {report.cot_argument_consistency:.4f}")
    print(
        "   CoT 覆盖率:      "
        f"{report.cot_coverage_rate:.4f} "
        f"(checked={report.cot_checked}, skipped={report.cot_skipped}, parse_fail={report.cot_parse_fail})"
    )
    if report.cot_counterfactual_checked > 0:
        print(
            "   CoT 反事实一致性: "
            f"{report.cot_counterfactual_pass_rate:.4f} "
            f"(checked={report.cot_counterfactual_checked})"
        )
    print(f"   Schema 符合率(类型+角色): {report.schema_compliance_rate:.4f}")

    print("\n" + "=" * 60)


__all__ = [
    "AcademicEventEvaluator",
    "EvaluationResult",
    "MetricsReport",
    "print_metrics_report",
]
