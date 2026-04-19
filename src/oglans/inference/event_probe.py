"""
Formal event-probe v2 repairs for conservative evaluation post-processing.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Mapping


_PLEDGE_PAIR_CUES = (
    "再质押",
    "质押给",
    "累计被质押",
    "被质押",
    "办理质押",
    "股份质押",
)
_STRONG_FINANCE_CUES = ("融资", "轮融资", "增资", "募资", "融资金额")
_STRONG_BUYBACK_CUES = ("股份回购", "回购股份", "回购公司股份", "回购价", "购回")
_DELISTING_CUES = ("终止上市", "退市", "暂停上市")
_BANKRUPTCY_TO_LISTING_ROLE_MAP = {
    "破产公司": "上市公司",
    "披露时间": "披露时间",
    "破产时间": "事件时间",
}
_FORMAL_LISTING_CUES = ("正式上市", "上市交易", "开始上市交易", "挂牌上市", "科创板上市")
_PREP_LISTING_CUES = (
    "上市辅导",
    "辅导备案",
    "招股",
    "IPO",
    "首次公开发行",
    "上会",
    "申报",
    "递交",
    "聆讯",
    "拟上市",
    "准备上市",
    "申请上市",
    "分拆上市",
    "注册上市",
)
_QUANTITY_ROLE_TOKENS = ("数量", "股份数量", "股票/股份数量")
_NUMERIC_FRAGMENT_PATTERN = re.compile(r"^\d+$")
_MONEY_VALUE_PATTERN = re.compile(r"^(?P<num>[\d,.]+)(?P<unit>[^\d,.]+)$")
_BUYBACK_PRICE_UNITS = ("元", "港元", "美元", "英镑", "欧元", "日元", "人民币")
_RUMOR_CUES = ("谣言", "传言", "传闻")
_RUMOR_DISMISS_CUES = ("澄清", "辟谣", "此前")
_BID_TARGET_SUFFIX_CUES = ("项目", "工程", "标段", "采购", "合同")
_BID_TARGET_CONNECTOR_PATTERN = re.compile(r"[及和、与暨]")


def _empty_probe_stats() -> Dict[str, int]:
    return {
        "event_probe_input_records": 0,
        "event_probe_output_records": 0,
        "event_probe_records_preserved": 0,
        "event_type_converted_count": 0,
        "event_type_dropped_count": 0,
        "pledge_pair_conversions": 0,
        "competing_event_drops": 0,
        "listing_stage_canonicalizations": 0,
        "bankruptcy_delisting_conversions": 0,
        "quantity_fragment_merges": 0,
        "financing_round_canonicalizations": 0,
        "buyback_price_canonicalizations": 0,
        "value_fragment_drops": 0,
        "grounded_span_merges": 0,
        "contextual_argument_drops": 0,
        "bid_target_merges": 0,
    }


def _has_any(source_text: str, cues: Iterable[str]) -> bool:
    return any(cue in source_text for cue in cues)


def _record_event_type(record: Mapping[str, Any]) -> str:
    return str(record.get("event_type", "")).strip()


def _normalized_arguments(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in record.get("arguments", []) or []:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", "")).strip()
        argument = str(item.get("argument", "")).strip()
        if role and argument:
            normalized.append({"role": role, "argument": argument})
    return normalized


def _dedupe_arguments(arguments: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in arguments:
        role = str(item.get("role", "")).strip()
        argument = str(item.get("argument", "")).strip()
        if not role or not argument:
            continue
        key = (role, argument)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"role": role, "argument": argument})
    return deduped


def _should_drop_competing_event(event_type: str, event_types: set[str], source_text: str) -> bool:
    if event_type == "企业融资" and {"企业收购", "股份回购"} & event_types:
        return not _has_any(source_text, _STRONG_FINANCE_CUES)
    if event_type == "股份回购" and {"企业收购", "股东减持"} & event_types:
        return not _has_any(source_text, _STRONG_BUYBACK_CUES)
    return False


def _canonicalize_listing_stage(value: str, source_text: str) -> str:
    context = f"{value} {source_text[:500]}"
    if _has_any(context, ("终止上市", "终止挂牌", "退市", "终止")):
        return "终止上市"
    if "暂停" in context:
        return "暂停上市"
    if _has_any(context, _PREP_LISTING_CUES):
        return "筹备上市"
    if _has_any(context, _FORMAL_LISTING_CUES):
        return "正式上市"
    if _has_any(value, ("上市", "挂牌", "发行", "科创板")):
        return "正式上市"
    return value


def _canonicalize_financing_round(value: str) -> str:
    for suffix in ("轮融资", "融资", "轮"):
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)].strip()
    return value


def _strip_buyback_price_wrapper(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("介乎") and len(stripped) > len("介乎"):
        stripped = stripped[len("介乎") :].strip()
    if stripped.endswith("/股") and len(stripped) > len("/股"):
        stripped = stripped[: -len("/股")].strip()
    return stripped


def _looks_like_valid_buyback_price(value: str) -> bool:
    if not re.search(r"\d", value):
        return False
    return any(unit in value for unit in _BUYBACK_PRICE_UNITS)


def _find_joined_money_span(values: List[str], source_text: str) -> str:
    if len(values) < 2:
        return ""

    positions = [(source_text.find(value), value) for value in values]
    if any(position < 0 for position, _ in positions):
        return ""
    ordered_values = [value for _, value in sorted(positions, key=lambda item: item[0])]

    normalized_groups: List[str] = []
    last_match = _MONEY_VALUE_PATTERN.fullmatch(ordered_values[-1].replace(" ", ""))
    if last_match is None:
        return ""

    for raw in ordered_values[:-1]:
        digits = raw.replace(",", "").strip()
        if not digits or not digits.isdigit():
            return ""
        normalized_groups.append(digits)

    last_numeric = last_match.group("num").replace(",", "")
    unit = last_match.group("unit").strip()
    if not unit or not re.search(r"\d", last_numeric):
        return ""

    if "." in last_numeric:
        integer_part, decimal_part = last_numeric.split(".", 1)
        decimal_suffix = f".{decimal_part}"
    else:
        integer_part = last_numeric
        decimal_suffix = ""
    if not integer_part.isdigit():
        return ""

    normalized_groups.append(integer_part)
    candidate = f"{','.join(normalized_groups)}{decimal_suffix}{unit}"
    if candidate in source_text or f"-{candidate}" in source_text:
        return candidate

    plain_candidate = f"{''.join(normalized_groups)}{decimal_suffix}{unit}"
    if plain_candidate in source_text or f"-{plain_candidate}" in source_text:
        return plain_candidate
    return ""


def _has_contextual_rumor_cue(source_text: str, value: str) -> bool:
    if not source_text or not value:
        return False
    start = source_text.find(value)
    while start >= 0:
        end = start + len(value)
        window = source_text[max(0, start - 16) : min(len(source_text), end + 16)]
        if _has_any(window, _RUMOR_CUES) and _has_any(window, _RUMOR_DISMISS_CUES):
            return True
        start = source_text.find(value, start + 1)
    return False


def _find_ordered_covering_span(values: List[str], source_text: str) -> str:
    if len(values) < 2 or not source_text:
        return ""

    search_from = 0
    spans: List[tuple[int, int]] = []
    for value in values:
        start = source_text.find(value, search_from)
        if start < 0:
            return ""
        end = start + len(value)
        spans.append((start, end))
        search_from = end

    candidate = source_text[spans[0][0] : spans[-1][1]].strip("，。；：:;, ")
    if candidate == values[0]:
        return ""
    if not all(value in candidate for value in values):
        return ""
    if not (
        _BID_TARGET_CONNECTOR_PATTERN.search(candidate)
        or any(cue in candidate for cue in _BID_TARGET_SUFFIX_CUES)
    ):
        return ""
    return candidate


def _find_joined_quantity_span(values: List[str], source_text: str) -> str:
    digits = [re.sub(r"\D", "", value) for value in values]
    if len(digits) < 2 or not all(_NUMERIC_FRAGMENT_PATTERN.fullmatch(value) for value in digits):
        return ""

    comma_joined = ",".join(digits)
    plain_joined = "".join(digits)
    candidates = (
        f"{comma_joined}股",
        f"{comma_joined}张",
        comma_joined,
        f"{plain_joined}股",
        f"{plain_joined}张",
        plain_joined,
    )
    return next((candidate for candidate in candidates if candidate and candidate in source_text), "")


def _merge_quantity_fragments(arguments: List[Dict[str, str]], source_text: str) -> tuple[List[Dict[str, str]], int]:
    if not source_text:
        return arguments, 0

    output: List[Dict[str, str]] = []
    consumed_indices: set[int] = set()
    merge_count = 0
    roles = list(dict.fromkeys(arg["role"] for arg in arguments))

    for role in roles:
        if not any(token in role for token in _QUANTITY_ROLE_TOKENS):
            continue
        role_indices = [idx for idx, arg in enumerate(arguments) if arg["role"] == role]
        values = [arguments[idx]["argument"] for idx in role_indices]
        joined_span = _find_joined_quantity_span(values, source_text)
        if not joined_span:
            continue
        first_index = role_indices[0]
        arguments[first_index] = {"role": role, "argument": joined_span}
        consumed_indices.update(role_indices[1:])
        merge_count += 1

    for idx, arg in enumerate(arguments):
        if idx not in consumed_indices:
            output.append(arg)
    return _dedupe_arguments(output), merge_count


def _apply_buyback_price_repairs(
    arguments: List[Dict[str, str]],
    *,
    source_text: str,
    stats: Dict[str, int],
) -> List[Dict[str, str]]:
    repaired: List[Dict[str, str]] = []
    for item in arguments:
        role = item["role"]
        value = item["argument"]
        if role != "每股交易价格":
            repaired.append(item)
            continue

        normalized = _strip_buyback_price_wrapper(value)
        if normalized != value and (not source_text or normalized in source_text):
            stats["buyback_price_canonicalizations"] += 1
            value = normalized
        if value in {"股", "/股"} or not _looks_like_valid_buyback_price(value):
            stats["value_fragment_drops"] += 1
            continue
        repaired.append({"role": role, "argument": value})
    return _dedupe_arguments(repaired)


def _apply_loss_repairs(
    arguments: List[Dict[str, str]],
    *,
    source_text: str,
    stats: Dict[str, int],
) -> List[Dict[str, str]]:
    repaired: List[Dict[str, str]] = []
    for item in arguments:
        if item["role"] == "净亏损" and _has_contextual_rumor_cue(source_text, item["argument"]):
            stats["contextual_argument_drops"] += 1
            continue
        repaired.append(item)

    loss_values = [item["argument"] for item in repaired if item["role"] == "净亏损"]
    merged_value = _find_joined_money_span(loss_values, source_text)
    if not merged_value:
        return _dedupe_arguments(repaired)

    output: List[Dict[str, str]] = []
    replaced = False
    consumed = False
    for item in repaired:
        if item["role"] != "净亏损":
            output.append(item)
            continue
        if not replaced:
            output.append({"role": "净亏损", "argument": merged_value})
            replaced = True
        elif item["argument"] != merged_value:
            consumed = True
    if replaced and consumed:
        stats["grounded_span_merges"] += 1
    return _dedupe_arguments(output)


def _apply_bid_target_repairs(
    arguments: List[Dict[str, str]],
    *,
    source_text: str,
    stats: Dict[str, int],
) -> List[Dict[str, str]]:
    target_values = [item["argument"] for item in arguments if item["role"] == "中标标的"]
    merged_value = _find_ordered_covering_span(target_values, source_text)
    if not merged_value:
        return _dedupe_arguments(arguments)

    output: List[Dict[str, str]] = []
    replaced = False
    consumed = False
    for item in arguments:
        if item["role"] != "中标标的":
            output.append(item)
            continue
        if not replaced:
            output.append({"role": "中标标的", "argument": merged_value})
            replaced = True
        elif item["argument"] != merged_value:
            consumed = True
    if replaced and consumed:
        stats["bid_target_merges"] += 1
    return _dedupe_arguments(output)


def _canonicalize_arguments(
    event_type: str,
    arguments: Iterable[Mapping[str, str]],
    *,
    source_text: str,
    stats: Dict[str, int],
) -> List[Dict[str, str]]:
    canonicalized: List[Dict[str, str]] = []
    for item in arguments:
        role = str(item.get("role", "")).strip()
        value = str(item.get("argument", "")).strip()
        if not role or not value:
            continue

        new_value = value
        if event_type == "公司上市" and role == "环节":
            new_value = _canonicalize_listing_stage(value, source_text)
            if new_value != value:
                stats["listing_stage_canonicalizations"] += 1
        elif event_type == "企业融资" and role == "融资轮次":
            new_value = _canonicalize_financing_round(value)
            if new_value != value:
                stats["financing_round_canonicalizations"] += 1
        canonicalized.append({"role": role, "argument": new_value})

    canonicalized, merge_count = _merge_quantity_fragments(_dedupe_arguments(canonicalized), source_text)
    stats["quantity_fragment_merges"] += merge_count

    if event_type == "股份回购":
        canonicalized = _apply_buyback_price_repairs(canonicalized, source_text=source_text, stats=stats)
    elif event_type == "亏损":
        canonicalized = _apply_loss_repairs(canonicalized, source_text=source_text, stats=stats)
    elif event_type == "中标":
        canonicalized = _apply_bid_target_repairs(canonicalized, source_text=source_text, stats=stats)
    return canonicalized


def _convert_bankruptcy_to_delisting(record: Mapping[str, Any]) -> Dict[str, Any]:
    converted_arguments: List[Dict[str, str]] = []
    for argument in _normalized_arguments(record):
        mapped_role = _BANKRUPTCY_TO_LISTING_ROLE_MAP.get(argument["role"])
        if mapped_role:
            converted_arguments.append({"role": mapped_role, "argument": argument["argument"]})
    converted_arguments.append({"role": "环节", "argument": "终止上市"})
    return {"event_type": "公司上市", "arguments": _dedupe_arguments(converted_arguments)}


def apply_event_probe_v2(
    records: Iterable[Mapping[str, Any]],
    *,
    source_text: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    copied_records = [deepcopy(dict(record)) for record in records or [] if isinstance(record, Mapping)]
    source = str(source_text or "")
    stats = _empty_probe_stats()
    stats["event_probe_input_records"] = len(copied_records)

    original_event_types = {_record_event_type(record) for record in copied_records}
    probed_records: List[Dict[str, Any]] = []
    for record in copied_records:
        event_type = _record_event_type(record)
        if _should_drop_competing_event(event_type, original_event_types, source):
            stats["event_type_dropped_count"] += 1
            stats["competing_event_drops"] += 1
            continue

        current = deepcopy(record)
        if event_type == "企业破产" and "公司上市" not in original_event_types and _has_any(source, _DELISTING_CUES):
            current = _convert_bankruptcy_to_delisting(current)
            event_type = "公司上市"
            stats["event_type_converted_count"] += 1
            stats["bankruptcy_delisting_conversions"] += 1

        current["event_type"] = event_type
        current["arguments"] = _canonicalize_arguments(
            event_type,
            _normalized_arguments(current),
            source_text=source,
            stats=stats,
        )
        probed_records.append(current)

    event_types = [_record_event_type(record) for record in probed_records]
    unpledge_indices = [idx for idx, event_type in enumerate(event_types) if event_type == "解除质押"]
    has_pledge_prediction = any(event_type == "质押" for event_type in event_types)
    if len(unpledge_indices) >= 2 and not has_pledge_prediction and _has_any(source, _PLEDGE_PAIR_CUES):
        convert_index = unpledge_indices[-1]
        probed_records[convert_index]["event_type"] = "质押"
        stats["event_type_converted_count"] += 1
        stats["pledge_pair_conversions"] += 1

    stats["event_probe_output_records"] = len(probed_records)
    stats["event_probe_records_preserved"] = max(
        0,
        len(probed_records) - stats["event_type_converted_count"],
    )
    return probed_records, stats


__all__ = ["apply_event_probe_v2"]
