#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared evaluation primitives used by local and API evaluation entrypoints.
"""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils.json_parser import RobustJSONParser


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _metric_block(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision, recall, f1 = _compute_prf(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def _empty_multiplicity_bucket() -> Dict[str, int]:
    return {
        "support_samples": 0,
        "support_gold_events": 0,
        "doc_role_tp": 0,
        "doc_role_fp": 0,
        "doc_role_fn": 0,
    }


def _multiplicity_metric_payload(bucket: Dict[str, int]) -> Dict[str, Any]:
    precision, recall, f1 = _compute_prf(
        int(bucket.get("doc_role_tp", 0)),
        int(bucket.get("doc_role_fp", 0)),
        int(bucket.get("doc_role_fn", 0)),
    )
    return {
        "support_samples": int(bucket.get("support_samples", 0)),
        "support_gold_events": int(bucket.get("support_gold_events", 0)),
        "doc_role": {
            "MicroPrecision": precision,
            "MicroRecall": recall,
            "MicroF1": f1,
            "TP": int(bucket.get("doc_role_tp", 0)),
            "FP": int(bucket.get("doc_role_fp", 0)),
            "FN": int(bucket.get("doc_role_fn", 0)),
        },
    }


def _counter_overlap_count(pred_items: List[Tuple[Any, ...]], gold_items: List[Tuple[Any, ...]]) -> int:
    pred_counter = Counter(pred_items)
    gold_counter = Counter(gold_items)
    return sum((pred_counter & gold_counter).values())


def _non_empty_slot_count(record: Tuple[Optional[Tuple[str, ...]], ...]) -> int:
    return sum(1 for value in record if value is not None)


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
    doc_ee: Dict[str, Any] = field(default_factory=dict)
    ee_text_proxy: Dict[str, Any] = field(default_factory=dict)
    gold_event_multiplicity_breakdown: Dict[str, Any] = field(default_factory=dict)


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
            "doc_role_tp": 0,
            "doc_role_fp": 0,
            "doc_role_fn": 0,
            "doc_event_type_tp": 0,
            "doc_event_type_fp": 0,
            "doc_event_type_fn": 0,
            "doc_instance_tp": 0,
            "doc_instance_fp": 0,
            "doc_instance_fn": 0,
            "doc_combination_tp": 0,
            "doc_combination_fp": 0,
            "doc_combination_fn": 0,
            "doc_role_stats": defaultdict(lambda: defaultdict(lambda: [0, 0, 0])),
            "doc_event_type_stats": defaultdict(lambda: [0, 0, 0]),
            "doc_role_orders": defaultdict(list),
            "gold_event_multiplicity": {
                "single_event": _empty_multiplicity_bucket(),
                "multi_event": _empty_multiplicity_bucket(),
                "zero_gold": _empty_multiplicity_bucket(),
            },
            "text_proxy_trigger_text_id": [0, 0, 0],
            "text_proxy_trigger_text_cls": [0, 0, 0],
            "text_proxy_argument_text_id": [0, 0, 0],
            "text_proxy_argument_text_cls": [0, 0, 0],
            "text_proxy_argument_attached_text_id": [0, 0, 0],
            "text_proxy_argument_attached_text_cls": [0, 0, 0],
            "text_proxy_grounded_total": 0,
            "text_proxy_grounding_total": 0,
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

    def _collect_role_order_map(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
        role_order_by_event: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        role_map: Dict[str, List[str]] = {
            str(event_type): list(roles or [])
            for event_type, roles in (role_order_by_event or {}).items()
        }
        for events in (gold_events or [], pred_events or []):
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event_type", "")).strip()
                if not event_type:
                    continue
                role_map.setdefault(event_type, [])
                for arg in event.get("arguments", []) or []:
                    if not isinstance(arg, dict):
                        continue
                    role = str(arg.get("role", "")).strip()
                    if role and role not in role_map[event_type]:
                        role_map[event_type].append(role)
        for event_type, roles in role_map.items():
            if not roles:
                role_map[event_type] = []
        return role_map

    def _build_doc_record(
        self,
        event: Dict[str, Any],
        role_order: List[str],
    ) -> Tuple[Optional[Tuple[str, ...]], ...]:
        grouped: Dict[str, Set[str]] = {role: set() for role in role_order}
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = str(arg.get("role", "")).strip()
            if role not in grouped:
                continue
            value = self.normalize_text(arg.get("argument", ""))
            if value:
                grouped[role].add(value)
        record: List[Optional[Tuple[str, ...]]] = []
        for role in role_order:
            values = tuple(sorted(grouped.get(role, set())))
            record.append(values if values else None)
        return tuple(record)

    def _extract_doc_records(
        self,
        events: List[Dict[str, Any]],
        role_order_map: Dict[str, List[str]],
    ) -> Dict[str, List[Tuple[Optional[Tuple[str, ...]], ...]]]:
        records_by_type: Dict[str, List[Tuple[Optional[Tuple[str, ...]], ...]]] = defaultdict(list)
        if not isinstance(events, list):
            return {}
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", "")).strip()
            if not event_type:
                continue
            role_order = role_order_map.get(event_type, [])
            records_by_type[event_type].append(self._build_doc_record(event, role_order))
        return dict(records_by_type)

    def _match_doc_records(
        self,
        pred_records: List[Tuple[Optional[Tuple[str, ...]], ...]],
        gold_records: List[Tuple[Optional[Tuple[str, ...]], ...]],
        role_order: List[str],
    ) -> Dict[str, List[int]]:
        role_stats: Dict[str, List[int]] = {role: [0, 0, 0] for role in role_order}
        remaining_pred = sorted(pred_records, key=_non_empty_slot_count, reverse=True)
        remaining_gold = list(gold_records)

        while remaining_pred and remaining_gold:
            pred_record = remaining_pred.pop(0)

            def _score(gold_record: Tuple[Optional[Tuple[str, ...]], ...]) -> int:
                return sum(1 for pred_slot, gold_slot in zip(pred_record, gold_record) if pred_slot == gold_slot)

            best_idx = max(range(len(remaining_gold)), key=lambda idx: _score(remaining_gold[idx]))
            gold_record = remaining_gold.pop(best_idx)

            for role, pred_slot, gold_slot in zip(role_order, pred_record, gold_record):
                if gold_slot is None:
                    if pred_slot is not None:
                        role_stats[role][1] += 1
                    continue
                if pred_slot is None:
                    role_stats[role][2] += 1
                    continue
                if pred_slot == gold_slot:
                    role_stats[role][0] += 1
                else:
                    role_stats[role][1] += 1
                    role_stats[role][2] += 1

        for pred_record in remaining_pred:
            for role, pred_slot in zip(role_order, pred_record):
                if pred_slot is not None:
                    role_stats[role][1] += 1

        for gold_record in remaining_gold:
            for role, gold_slot in zip(role_order, gold_record):
                if gold_slot is not None:
                    role_stats[role][2] += 1

        return role_stats

    def _record_signature(
        self,
        event_type: str,
        role_order: List[str],
        record: Tuple[Optional[Tuple[str, ...]], ...],
        include_event_type: bool,
    ) -> Tuple[Any, ...]:
        role_pairs = tuple(
            (role, slot)
            for role, slot in zip(role_order, record)
            if slot is not None
        )
        if include_event_type:
            return (event_type, role_pairs)
        return role_pairs

    def _compute_doc_ee_sample_stats(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
        role_order_by_event: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        role_order_map = self._collect_role_order_map(pred_events, gold_events, role_order_by_event)
        pred_records = self._extract_doc_records(pred_events, role_order_map)
        gold_records = self._extract_doc_records(gold_events, role_order_map)

        sample_stats = {
            "role_order_map": role_order_map,
            "event_role_stats": {},
            "event_type_stats": {},
            "overall": [0, 0, 0],
            "classification": [0, 0, 0],
            "instance": [0, 0, 0],
            "combination": [0, 0, 0],
        }

        all_event_types = sorted(set(role_order_map) | set(pred_records) | set(gold_records))
        pred_instances: List[Tuple[Any, ...]] = []
        gold_instances: List[Tuple[Any, ...]] = []
        pred_combinations: List[Tuple[Any, ...]] = []
        gold_combinations: List[Tuple[Any, ...]] = []

        for event_type in all_event_types:
            role_order = role_order_map.get(event_type, [])
            event_pred_records = pred_records.get(event_type, [])
            event_gold_records = gold_records.get(event_type, [])
            role_stats = self._match_doc_records(event_pred_records, event_gold_records, role_order)
            sample_stats["event_role_stats"][event_type] = role_stats

            event_type_counts = [0, 0, 0]
            pred_present = bool(event_pred_records)
            gold_present = bool(event_gold_records)
            if pred_present and gold_present:
                event_type_counts[0] += 1
            elif pred_present:
                event_type_counts[1] += 1
            elif gold_present:
                event_type_counts[2] += 1
            sample_stats["event_type_stats"][event_type] = event_type_counts

            for role in role_order:
                tp, fp, fn = role_stats[role]
                sample_stats["overall"][0] += tp
                sample_stats["overall"][1] += fp
                sample_stats["overall"][2] += fn
            for idx, value in enumerate(event_type_counts):
                sample_stats["classification"][idx] += value

            for record in event_pred_records:
                pred_instances.append(self._record_signature(event_type, role_order, record, include_event_type=True))
                pred_combinations.append(self._record_signature(event_type, role_order, record, include_event_type=False))
            for record in event_gold_records:
                gold_instances.append(self._record_signature(event_type, role_order, record, include_event_type=True))
                gold_combinations.append(self._record_signature(event_type, role_order, record, include_event_type=False))

        instance_tp = _counter_overlap_count(pred_instances, gold_instances)
        combination_tp = _counter_overlap_count(pred_combinations, gold_combinations)
        sample_stats["instance"] = [
            instance_tp,
            max(0, len(pred_instances) - instance_tp),
            max(0, len(gold_instances) - instance_tp),
        ]
        sample_stats["combination"] = [
            combination_tp,
            max(0, len(pred_combinations) - combination_tp),
            max(0, len(gold_combinations) - combination_tp),
        ]
        return sample_stats

    @staticmethod
    def _gold_event_multiplicity_bucket(gold_events: List[Dict[str, Any]]) -> Tuple[str, int]:
        if not isinstance(gold_events, list):
            return "zero_gold", 0
        gold_event_count = sum(1 for event in gold_events if isinstance(event, dict))
        if gold_event_count == 1:
            return "single_event", gold_event_count
        if gold_event_count >= 2:
            return "multi_event", gold_event_count
        return "zero_gold", gold_event_count

    def _update_gold_event_multiplicity_stats(
        self,
        gold_events: List[Dict[str, Any]],
        doc_sample: Dict[str, Any],
    ) -> None:
        bucket_name, gold_event_count = self._gold_event_multiplicity_bucket(gold_events)
        bucket = self.stats["gold_event_multiplicity"][bucket_name]
        bucket["support_samples"] += 1
        bucket["support_gold_events"] += int(gold_event_count)
        bucket["doc_role_tp"] += int(doc_sample["overall"][0])
        bucket["doc_role_fp"] += int(doc_sample["overall"][1])
        bucket["doc_role_fn"] += int(doc_sample["overall"][2])

    def _extract_text_proxy_sets(self, events: List[Dict[str, Any]]) -> Dict[str, Set[Tuple[Any, ...]]]:
        metrics = {
            "trigger_text_id": set(),
            "trigger_text_cls": set(),
            "argument_text_id": set(),
            "argument_text_cls": set(),
            "argument_attached_text_id": set(),
            "argument_attached_text_cls": set(),
        }
        if not isinstance(events, list):
            return metrics
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", "")).strip()
            if not event_type:
                continue
            trigger = self.normalize_text(event.get("trigger", ""))
            if trigger:
                metrics["trigger_text_id"].add((trigger,))
                metrics["trigger_text_cls"].add((event_type, trigger))
            for arg in event.get("arguments", []) or []:
                if not isinstance(arg, dict):
                    continue
                role = str(arg.get("role", "")).strip()
                value = self.normalize_text(arg.get("argument", ""))
                if not value:
                    continue
                metrics["argument_text_id"].add((event_type, value))
                metrics["argument_text_cls"].add((event_type, role, value))
                if trigger:
                    metrics["argument_attached_text_id"].add((event_type, trigger, value))
                    metrics["argument_attached_text_cls"].add((event_type, trigger, role, value))
        return metrics

    def _update_text_proxy_stats(
        self,
        pred_events: List[Dict[str, Any]],
        gold_events: List[Dict[str, Any]],
    ) -> None:
        pred_sets = self._extract_text_proxy_sets(pred_events)
        gold_sets = self._extract_text_proxy_sets(gold_events)
        for metric_name, pred_values in pred_sets.items():
            gold_values = gold_sets[metric_name]
            tp = len(pred_values & gold_values)
            fp = len(pred_values - gold_values)
            fn = len(gold_values - pred_values)
            bucket = self.stats[f"text_proxy_{metric_name}"]
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn

    def _update_text_proxy_grounding(
        self,
        source_text: str,
        pred_events: List[Dict[str, Any]],
    ) -> None:
        clean_source = self.normalize_text(source_text)
        if not clean_source or not isinstance(pred_events, list):
            return
        grounded = 0
        total = 0
        for event in pred_events:
            if not isinstance(event, dict):
                continue
            trigger = self.normalize_text(event.get("trigger", ""))
            if trigger:
                total += 1
                if trigger in clean_source:
                    grounded += 1
            for arg in event.get("arguments", []) or []:
                if not isinstance(arg, dict):
                    continue
                value = self.normalize_text(arg.get("argument", ""))
                if not value:
                    continue
                total += 1
                if value in clean_source:
                    grounded += 1
        self.stats["text_proxy_grounded_total"] += grounded
        self.stats["text_proxy_grounding_total"] += total

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
        role_order_by_event: Optional[Dict[str, List[str]]] = None,
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

        doc_sample = self._compute_doc_ee_sample_stats(
            pred_events,
            gold_events,
            role_order_by_event=role_order_by_event,
        )
        self.stats["doc_role_tp"] += doc_sample["overall"][0]
        self.stats["doc_role_fp"] += doc_sample["overall"][1]
        self.stats["doc_role_fn"] += doc_sample["overall"][2]
        self.stats["doc_event_type_tp"] += doc_sample["classification"][0]
        self.stats["doc_event_type_fp"] += doc_sample["classification"][1]
        self.stats["doc_event_type_fn"] += doc_sample["classification"][2]
        self.stats["doc_instance_tp"] += doc_sample["instance"][0]
        self.stats["doc_instance_fp"] += doc_sample["instance"][1]
        self.stats["doc_instance_fn"] += doc_sample["instance"][2]
        self.stats["doc_combination_tp"] += doc_sample["combination"][0]
        self.stats["doc_combination_fp"] += doc_sample["combination"][1]
        self.stats["doc_combination_fn"] += doc_sample["combination"][2]
        self._update_gold_event_multiplicity_stats(gold_events, doc_sample)
        for event_type, role_order in doc_sample["role_order_map"].items():
            existing = self.stats["doc_role_orders"][event_type]
            for role in role_order:
                if role not in existing:
                    existing.append(role)
        for event_type, role_stats in doc_sample["event_role_stats"].items():
            for role, values in role_stats.items():
                bucket = self.stats["doc_role_stats"][event_type][role]
                bucket[0] += values[0]
                bucket[1] += values[1]
                bucket[2] += values[2]
        for event_type, values in doc_sample["event_type_stats"].items():
            bucket = self.stats["doc_event_type_stats"][event_type]
            bucket[0] += values[0]
            bucket[1] += values[1]
            bucket[2] += values[2]

        self._update_text_proxy_stats(pred_events, gold_events)

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

        doc_event_types = sorted(
            set(self.stats["doc_role_orders"].keys()) | set(self.stats["doc_event_type_stats"].keys())
        )
        doc_events_payload: List[Dict[str, Any]] = []
        macro_precision_sum = 0.0
        macro_recall_sum = 0.0
        macro_f1_sum = 0.0

        for event_type in doc_event_types:
            role_order = list(self.stats["doc_role_orders"].get(event_type, []))
            role_stats = self.stats["doc_role_stats"].get(event_type, {})
            if not role_order:
                role_order = sorted(role_stats.keys())

            role_payloads: List[Dict[str, Any]] = []
            event_tp = event_fp = event_fn = 0
            event_macro_precision = 0.0
            event_macro_recall = 0.0
            event_macro_f1 = 0.0

            for role in role_order:
                tp, fp, fn = role_stats.get(role, [0, 0, 0])
                role_precision, role_recall, role_f1 = _compute_prf(tp, fp, fn)
                event_tp += tp
                event_fp += fp
                event_fn += fn
                event_macro_precision += role_precision
                event_macro_recall += role_recall
                event_macro_f1 += role_f1
                role_payloads.append(
                    {
                        "RoleType": role,
                        "Precision": role_precision,
                        "Recall": role_recall,
                        "F1": role_f1,
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                    }
                )

            role_den = len(role_payloads) if role_payloads else 1
            macro_precision = event_macro_precision / role_den
            macro_recall = event_macro_recall / role_den
            macro_f1 = event_macro_f1 / role_den
            micro_precision, micro_recall, micro_f1 = _compute_prf(event_tp, event_fp, event_fn)
            macro_precision_sum += macro_precision
            macro_recall_sum += macro_recall
            macro_f1_sum += macro_f1

            event_payload = {
                "EventType": event_type,
                "MacroPrecision": macro_precision,
                "MacroRecall": macro_recall,
                "MacroF1": macro_f1,
                "MicroPrecision": micro_precision,
                "MicroRecall": micro_recall,
                "MicroF1": micro_f1,
                "TP": event_tp,
                "FP": event_fp,
                "FN": event_fn,
                "Roles": role_payloads,
            }
            doc_events_payload.append(event_payload)

        n_doc_events = len(doc_events_payload) if doc_events_payload else 1
        overall_micro_precision, overall_micro_recall, overall_micro_f1 = _compute_prf(
            self.stats["doc_role_tp"],
            self.stats["doc_role_fp"],
            self.stats["doc_role_fn"],
        )

        classification_events: List[Dict[str, Any]] = []
        classification_macro_precision = 0.0
        classification_macro_recall = 0.0
        classification_macro_f1 = 0.0
        for event_type in doc_event_types:
            tp, fp, fn = self.stats["doc_event_type_stats"].get(event_type, [0, 0, 0])
            precision, recall, f1 = _compute_prf(tp, fp, fn)
            classification_macro_precision += precision
            classification_macro_recall += recall
            classification_macro_f1 += f1
            classification_events.append(
                {
                    "EventType": event_type,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                }
            )
        cls_den = len(classification_events) if classification_events else 1
        cls_micro_precision, cls_micro_recall, cls_micro_f1 = _compute_prf(
            self.stats["doc_event_type_tp"],
            self.stats["doc_event_type_fp"],
            self.stats["doc_event_type_fn"],
        )
        inst_precision, inst_recall, inst_f1 = _compute_prf(
            self.stats["doc_instance_tp"],
            self.stats["doc_instance_fp"],
            self.stats["doc_instance_fn"],
        )
        comb_precision, comb_recall, comb_f1 = _compute_prf(
            self.stats["doc_combination_tp"],
            self.stats["doc_combination_fp"],
            self.stats["doc_combination_fn"],
        )
        report.gold_event_multiplicity_breakdown = {
            bucket_name: _multiplicity_metric_payload(bucket)
            for bucket_name, bucket in self.stats["gold_event_multiplicity"].items()
        }
        report.doc_ee = {
            "overall": {
                "MacroPrecision": macro_precision_sum / n_doc_events,
                "MacroRecall": macro_recall_sum / n_doc_events,
                "MacroF1": macro_f1_sum / n_doc_events,
                "MicroPrecision": overall_micro_precision,
                "MicroRecall": overall_micro_recall,
                "MicroF1": overall_micro_f1,
                "TP": self.stats["doc_role_tp"],
                "FP": self.stats["doc_role_fp"],
                "FN": self.stats["doc_role_fn"],
                "Events": doc_events_payload,
            },
            "classification": {
                "MacroPrecision": classification_macro_precision / cls_den,
                "MacroRecall": classification_macro_recall / cls_den,
                "MacroF1": classification_macro_f1 / cls_den,
                "MicroPrecision": cls_micro_precision,
                "MicroRecall": cls_micro_recall,
                "MicroF1": cls_micro_f1,
                "TP": self.stats["doc_event_type_tp"],
                "FP": self.stats["doc_event_type_fp"],
                "FN": self.stats["doc_event_type_fn"],
                "Events": classification_events,
            },
            "instance": {
                "MicroPrecision": inst_precision,
                "MicroRecall": inst_recall,
                "MicroF1": inst_f1,
                "TP": self.stats["doc_instance_tp"],
                "FP": self.stats["doc_instance_fp"],
                "FN": self.stats["doc_instance_fn"],
            },
            "combination": {
                "MicroPrecision": comb_precision,
                "MicroRecall": comb_recall,
                "MicroF1": comb_f1,
                "TP": self.stats["doc_combination_tp"],
                "FP": self.stats["doc_combination_fp"],
                "FN": self.stats["doc_combination_fn"],
            },
            "gold_event_multiplicity_breakdown": report.gold_event_multiplicity_breakdown,
        }

        report.ee_text_proxy = {
            "trigger_text_id": _metric_block(*self.stats["text_proxy_trigger_text_id"]),
            "trigger_text_cls": _metric_block(*self.stats["text_proxy_trigger_text_cls"]),
            "argument_text_id": _metric_block(*self.stats["text_proxy_argument_text_id"]),
            "argument_text_cls": _metric_block(*self.stats["text_proxy_argument_text_cls"]),
            "argument_attached_text_id": _metric_block(*self.stats["text_proxy_argument_attached_text_id"]),
            "argument_attached_text_cls": _metric_block(*self.stats["text_proxy_argument_attached_text_cls"]),
            "grounding_coverage": _safe_div(
                self.stats["text_proxy_grounded_total"],
                self.stats["text_proxy_grounding_total"],
            ),
        }
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
        role_order_by_event: Optional[Dict[str, List[str]]] = None,
    ):
        self.update(
            pred_events,
            gold_events,
            parse_success=parse_success,
            parse_diagnostics=parse_diagnostics,
            role_order_by_event=role_order_by_event,
        )

        if source_text:
            has_halluc, halluc_count, total_entities, halluc_breakdown = self.check_hallucination(
                source_text,
                pred_events,
            )
            self._update_text_proxy_grounding(source_text, pred_events)
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
    if report.doc_ee:
        print("\n🧾 DocEE 主表指标")
        print(f"   Role Micro-F1:  {report.doc_ee['overall']['MicroF1']:.4f}")
        print(f"   Role Macro-F1:  {report.doc_ee['overall']['MacroF1']:.4f}")
        print(f"   Type Micro-F1:  {report.doc_ee['classification']['MicroF1']:.4f}")
        print(f"   Instance F1:    {report.doc_ee['instance']['MicroF1']:.4f}")
        print(f"   Combination F1: {report.doc_ee['combination']['MicroF1']:.4f}")

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


def build_primary_metric_map(report: MetricsReport) -> Dict[str, float]:
    doc_ee = report.doc_ee or {}
    overall = doc_ee.get("overall", {})
    classification = doc_ee.get("classification", {})
    instance = doc_ee.get("instance", {})
    combination = doc_ee.get("combination", {})
    multiplicity = doc_ee.get("gold_event_multiplicity_breakdown", report.gold_event_multiplicity_breakdown) or {}
    single_event_role = (multiplicity.get("single_event", {}) or {}).get("doc_role", {}) or {}
    multi_event_role = (multiplicity.get("multi_event", {}) or {}).get("doc_role", {}) or {}
    return {
        "legacy_dueefin_overall_precision": float(report.strict_precision),
        "legacy_dueefin_overall_recall": float(report.strict_recall),
        "legacy_dueefin_overall_f1": float(report.strict_f1),
        "strict_f1": float(report.strict_f1),
        "relaxed_f1": float(report.relaxed_f1),
        "type_f1": float(report.type_f1),
        "doc_role_micro_f1": float(overall.get("MicroF1", 0.0)),
        "doc_role_macro_f1": float(overall.get("MacroF1", 0.0)),
        "doc_event_type_micro_f1": float(classification.get("MicroF1", 0.0)),
        "doc_event_type_macro_f1": float(classification.get("MacroF1", 0.0)),
        "doc_instance_micro_f1": float(instance.get("MicroF1", 0.0)),
        "doc_combination_micro_f1": float(combination.get("MicroF1", 0.0)),
        "single_event_doc_role_micro_f1": float(single_event_role.get("MicroF1", 0.0)),
        "multi_event_doc_role_micro_f1": float(multi_event_role.get("MicroF1", 0.0)),
    }


__all__ = [
    "AcademicEventEvaluator",
    "EvaluationResult",
    "MetricsReport",
    "build_primary_metric_map",
    "print_metrics_report",
]
