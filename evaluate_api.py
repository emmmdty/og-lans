#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS API-based Evaluation Script

Evaluates Event Extraction models using external LLM APIs (DeepSeek, OpenAI, etc.)
for baseline comparison and ablation studies.

Features:
    - Multi-threaded API inference for faster evaluation
    - Automatic retry with exponential backoff
    - Token usage tracking and cost estimation
    - Compatible with AcademicEventEvaluator metrics

Usage:
    # Basic evaluation with DeepSeek
    python evaluate_api.py --split dev --model deepseek-chat

    # With few-shot examples
    python evaluate_api.py --split dev --use_fewshot --num_samples 100

    # Parallel inference
    python evaluate_api.py --split dev --concurrency 50

Environment Variables:
    DEEPSEEK_API_KEY: API key for DeepSeek endpoint
    DEEPSEEK_BASE_URL: Base URL for DeepSeek endpoint (optional)
    OPENAI_API_KEY: Fallback key for OpenAI-compatible endpoints
    OPENAI_BASE_URL: Fallback base URL for OpenAI-compatible endpoints

Authors:
    OG-LANS Research Team
"""

# evaluate_api.py
import os
import json
import argparse
import re
import time
import logging
import random
import io
import platform
import socket
import subprocess
import hashlib
from contextlib import redirect_stdout
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version as pkg_version, PackageNotFoundError

# 加载项目根目录的 .env 文件
def load_dotenv_file():
    """从项目根目录加载 .env 文件"""
    # 查找项目根目录（当前脚本所在目录）
    script_dir = Path(__file__).parent.resolve()
    env_file = script_dir / ".env"

    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 解析 KEY=VALUE 格式
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    # 移除引号
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    # 只在环境变量未设置时才加载
                    if key and key not in os.environ:
                        os.environ[key] = value

# 在导入其他模块前加载 .env
load_dotenv_file()

from oglans.data.adapter import DuEEFinAdapter
from oglans.config import ConfigManager
from oglans.utils.json_parser import (
    NORMALIZATION_VERSION,
    PARSER_VERSION,
    normalize_parsed_events,
    parse_event_list_strict_with_diagnostics,
)
from oglans.data.prompt_builder import (
    ChinesePromptBuilder,
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
)
from oglans.inference.cat_lite import apply_cat_lite_pipeline, perturb_text_for_counterfactual
from oglans.utils.academic_eval import bootstrap_confidence_intervals
from oglans.utils.run_manifest import (
    build_contract_record,
    build_run_manifest,
    save_json,
)
from oglans.utils.eval_protocol import (
    validate_primary_metric,
    resolve_primary_metric_value,
    compute_file_hash as shared_compute_file_hash,
    load_eval_protocol as shared_load_eval_protocol,
    load_role_alias_map as shared_load_role_alias_map,
    canonicalize_pred_roles as shared_canonicalize_pred_roles,
)
# 引用 evaluate.py 中的 Evaluator
from evaluate import AcademicEventEvaluator, print_metrics_report

# 日志设置保持不变
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def infer_dataset_name(config: Dict[str, Any]) -> str:
    """
    从配置中推断数据集名称，优先使用 taxonomy_path。
    失败时抛出异常，避免静默使用错误数据集。
    """
    # 1) 优先从 taxonomy_path 推断
    taxonomy_path = (
        config.get("algorithms", {})
        .get("ds_cns", {})
        .get("taxonomy_path")
    )
    if taxonomy_path:
        dataset_dir = os.path.basename(os.path.dirname(os.path.normpath(taxonomy_path)))
        if dataset_dir:
            return dataset_dir

    # 2) 从项目路径推断
    project = config.get("project", {})
    for key in ("dataset_cache_dir", "output_dir", "logging_dir"):
        path = project.get(key)
        if not path:
            continue
        base = os.path.basename(os.path.normpath(path))
        if base in {"checkpoints", "tensorboard", "samples", "eval", "logs", "log"}:
            base = os.path.basename(os.path.dirname(os.path.normpath(path)))
        if base:
            return base

    # 3) 最后尝试 project.name
    name = project.get("name")
    if name:
        return name

    raise ValueError(
        "无法推断数据集名称。请在配置中设置 algorithms.ds_cns.taxonomy_path 或 project.dataset_cache_dir。"
    )


def infer_eval_api_root(config: Dict[str, Any], dataset_name: str) -> str:
    """
    推断 API 评估输出根目录。

    规则：
    1) 若 project.* 路径位于 logs/<tag>/...，则输出到 logs/<tag>/eval_api
       （例如 debug 配置会写入 logs/debug/eval_api）。
    2) 否则回退到 logs/<dataset>/eval_api。
    """
    project = config.get("project", {})
    for key in ("output_dir", "logging_dir", "debug_data_dir"):
        raw = project.get(key)
        if not raw:
            continue
        norm = os.path.normpath(str(raw)).replace("\\", "/")
        parts = [p for p in norm.split("/") if p and p != "."]
        if "logs" not in parts:
            continue
        idx = parts.index("logs")
        if idx + 1 < len(parts):
            tag = parts[idx + 1]
            if tag:
                return os.path.join("logs", tag, "eval_api")
    return os.path.join("logs", dataset_name, "eval_api")


def is_retryable_error(exc: Exception) -> bool:
    """判断错误是否适合重试"""
    message = str(exc).lower()
    retry_keywords = [
        "timeout", "timed out", "rate limit", "too many requests", "429",
        "500", "502", "503", "504", "connection", "temporarily", "overloaded"
    ]
    return any(token in message for token in retry_keywords)


def normalize_pred_events(parsed_obj: Any) -> List[Dict[str, Any]]:
    """
    将解析结果统一转换为事件列表。
    兼容数组、单事件对象、以及 {"events": [...]} 包装格式。
    """
    return normalize_parsed_events(parsed_obj)


def get_pkg_version(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def get_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def get_git_dirty(repo_dir: Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        return bool(out.strip())
    except Exception:
        return None


def compute_config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_file_hash(path: Optional[str]) -> Optional[str]:
    return shared_compute_file_hash(path)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_eval_protocol(path: Optional[str]) -> Dict[str, Any]:
    return shared_load_eval_protocol(path)


def load_role_alias_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    return shared_load_role_alias_map(path)


def canonicalize_pred_roles(
    pred_events: List[Dict[str, Any]],
    alias_map: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    return shared_canonicalize_pred_roles(pred_events, alias_map)


def collect_runtime_manifest(repo_dir: Path) -> Dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.path.abspath(os.sys.executable),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "packages": {
            "openai": get_pkg_version("openai"),
            "tqdm": get_pkg_version("tqdm"),
            "PyYAML": get_pkg_version("PyYAML"),
            "dirtyjson": get_pkg_version("dirtyjson"),
            "transformers": get_pkg_version("transformers"),
            "torch": get_pkg_version("torch"),
            "unsloth": get_pkg_version("unsloth"),
        },
        "git": {
            "commit": get_git_commit(repo_dir),
            "dirty": get_git_dirty(repo_dir),
        },
    }


def sanitize_tag(text: str) -> str:
    """将任意字符串转换为可用于文件名的简短标签。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def render_metrics_report_text(report, eval_mode: str = "both") -> str:
    """将 print_metrics_report 的控制台输出捕获为文本。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_metrics_report(report, eval_mode=eval_mode)
    return buf.getvalue().strip("\n")


def log_and_save_metrics_report(
    report,
    logger: logging.Logger,
    report_file: Optional[str] = None,
    eval_mode: str = "both",
) -> str:
    """
    将 OG-LANS 评估报告同时写入日志与独立文本文件。
    返回报告文本，便于后续写入元数据。
    """
    text = render_metrics_report_text(report, eval_mode=eval_mode)
    if not text:
        return ""

    for line in text.splitlines():
        logger.info(line)

    if report_file:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.info(f"📄 Saved OG-LANS report to {report_file}")

    return text


def compute_sample_metric_row(
    evaluator: AcademicEventEvaluator,
    pred_events: List[Dict[str, Any]],
    gold_events: List[Dict[str, Any]],
) -> Dict[str, int]:
    """抽样统计行，用于 bootstrap 置信区间。"""
    pred_strict = evaluator.extract_triplets_strict(pred_events)
    gold_strict = evaluator.extract_triplets_strict(gold_events)
    pred_relaxed = evaluator.extract_triplets_relaxed(pred_events)
    gold_relaxed = evaluator.extract_triplets_relaxed(gold_events)
    pred_types = evaluator.extract_event_types(pred_events)
    gold_types = evaluator.extract_event_types(gold_events)
    return {
        "strict_tp": len(pred_strict & gold_strict),
        "strict_pred_total": len(pred_strict),
        "strict_gold_total": len(gold_strict),
        "relaxed_tp": evaluator.compute_relaxed_matches(pred_relaxed, gold_relaxed),
        "relaxed_pred_total": len(pred_relaxed),
        "relaxed_gold_total": len(gold_relaxed),
        "type_tp": len(pred_types & gold_types),
        "type_pred_total": len(pred_types),
        "type_gold_total": len(gold_types),
    }


def perform_api_inference(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_retries: int = 3,
    json_mode: str = "auto"
) -> Tuple[str, Dict[str, int], bool, Optional[str], Dict[str, Any]]:
    """API 推理重试逻辑（限流/超时友好）"""
    for attempt in range(max_retries):
        try:
            request_kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=0.0,  # Deterministic
                max_tokens=2048
            )
            # DeepSeek/OpenAI 兼容 JSON 约束。默认 auto 不强制，避免与数组根格式冲突。
            if json_mode == "on":
                request_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0
            }
            response_meta = {
                "response_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "response_created": getattr(response, "created", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None) if response.choices else None,
            }
            return content or "", usage, True, None, response_meta
        except Exception as e:
            retryable = is_retryable_error(e)
            if (not retryable) or attempt == max_retries - 1:
                print(f"Error calling API: {e}")
                return "", ZERO_USAGE.copy(), False, str(e), {}
            # 指数退避 + 抖动，减轻并发下的速率限制冲击
            sleep_s = (2 ** attempt) + random.uniform(0, 0.8)
            time.sleep(sleep_s)
    return "", ZERO_USAGE.copy(), False, "unknown_error", {}


def process_single_sample(
    client,
    model,
    max_retries,
    sample_idx: int,
    sample,
    use_fewshot: bool = False,
    fewshot_num_examples: int = 3,
    json_mode: str = "auto",
    schema: Optional[Dict[str, List[str]]] = None,
    pipeline_mode: str = "e2e",
):
    """
    样本推理与解析。支持 few-shot、prompt 风格和 JSON 模式。
    """
    text = sample.text
    gold_events = sample.events

    prompt_payload = build_inference_prompt_payload(
        text=text,
        schema=schema,
        use_oneshot=use_fewshot,
        num_examples=fewshot_num_examples,
    )
    messages = prompt_payload["messages"]

    # Inference
    response_text, usage, api_success, api_error, response_meta = perform_api_inference(
        client, 
        model, 
        messages,
        max_retries,
        json_mode
    )

    # Parse: 使用带诊断的解析器，区分 "[] 成功" 和 "解析失败"
    pred_events, parse_diagnostics = parse_event_list_strict_with_diagnostics(response_text)
    parse_success = bool(parse_diagnostics.get("success", False))
    cat_stats = {}
    if pipeline_mode == "cat_lite":
        cat_result = apply_cat_lite_pipeline(
            pred_events=pred_events,
            source_text=text,
            schema=schema,
            require_argument_in_text=True,
        )
        pred_events = cat_result.events
        cat_stats = {
            "kept_events": cat_result.kept_events,
            "dropped_events": cat_result.dropped_events,
            "kept_arguments": cat_result.kept_arguments,
            "dropped_arguments": cat_result.dropped_arguments,
        }

    return {
        "sample": sample,
        "sample_idx": sample_idx,
        "text": text,
        "gold_events": gold_events,
        "pred_events": pred_events,
        "response_text": response_text,
        "usage": usage,
        "parse_success": parse_success,
        "parse_diagnostics": parse_diagnostics,
        "api_success": api_success,
        "api_error": api_error,
        "response_meta": response_meta,
        "cat_stats": cat_stats,
    }


def main():
    run_start_ts = time.time()
    parser = argparse.ArgumentParser(description="DeepSeek/OpenAI API Evaluation Script")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--protocol",
        type=str,
        default="configs/eval_protocol.yaml",
        help="评估协议文件（主指标、种子、并发、统计设置）",
    )
    parser.add_argument("--model", type=str, default=None, help="覆盖配置文件中的 api_evaluation.model")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认读取 config.project.seed")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="eval_results_api.jsonl")
    parser.add_argument("--summary_file", type=str, default=None, help="评估汇总输出路径")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="并发请求数（默认读取 api_evaluation.concurrency，缺省为 50）"
    )
    parser.add_argument("--max_workers", type=int, default=None, help=argparse.SUPPRESS)  # 兼容旧参数名
    parser.add_argument("--json_mode", type=str, default="auto", choices=["auto", "on", "off"],
                        help="JSON 约束模式: auto(默认不强制), on(强制 json_object), off(关闭)")
    parser.add_argument("--bootstrap_samples", type=int, default=None, help="Bootstrap 采样次数")
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
        "--compute_ci",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算 bootstrap 置信区间（默认开启）"
    )
    # 【新增参数】
    parser.add_argument("--use_fewshot", action="store_true", help="使用 Few-shot 示例增强基线性能")
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
    args = parser.parse_args()

    # Load Config（支持 extends 继承与运行时默认值）
    config = ConfigManager().load_config(args.config)
    protocol = load_eval_protocol(args.protocol)
    comparison_cfg = config.get("comparison", {})
    model_profile = str(config.get("model", {}).get("profile"))

    api_cfg = config.get('api_evaluation', {})
    protocol_eval = protocol.get("evaluation", {}) if isinstance(protocol, dict) else {}
    protocol_primary_metric = str(protocol.get("primary_metric", "strict_f1"))
    if args.report_primary_metric is None:
        args.report_primary_metric = protocol_primary_metric
    args.report_primary_metric = validate_primary_metric(args.report_primary_metric)
    if args.canonical_metric_mode is None:
        args.canonical_metric_mode = str(protocol.get("canonical_metric_mode", "analysis_only"))
    if args.canonical_metric_mode not in {"off", "analysis_only", "apply_for_aux_metric"}:
        raise ValueError(f"Unsupported canonical metric mode: {args.canonical_metric_mode}")
    metric_settings = protocol.get("metrics", {})
    if args.cot_eval_mode is None:
        args.cot_eval_mode = str(metric_settings.get("cot", {}).get("eval_mode", "self_consistency"))
    if args.cot_eval_mode not in {"self_consistency", "counterfactual"}:
        raise ValueError(f"Unsupported cot_eval_mode: {args.cot_eval_mode}")
    if args.pipeline_mode is None:
        args.pipeline_mode = str(config.get("inference", {}).get("pipeline_mode", "e2e"))
    if args.pipeline_mode not in {"e2e", "cat_lite"}:
        raise ValueError(f"Unsupported pipeline_mode: {args.pipeline_mode}")
    metric_settings.setdefault("cot", {})
    metric_settings["cot"]["eval_mode"] = args.cot_eval_mode

    if args.max_workers is not None:
        args.concurrency = args.max_workers
    elif args.concurrency is None:
        args.concurrency = max(
            1,
            int(protocol_eval.get("concurrency", api_cfg.get("concurrency", 50))),
        )
    if args.seed is None:
        args.seed = int(config.get("project", {}).get("seed", 3407))
    if args.bootstrap_samples is None:
        args.bootstrap_samples = int(
            protocol_eval.get(
                "bootstrap_samples",
                config.get("experiment", {}).get("bootstrap_n_samples", 1000),
            )
        )
    random.seed(args.seed)

    # Initialize Client
    # 优先使用环境变量，配置文件中不应包含明文 API Key
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or api_cfg.get('api_key')
    )
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or api_cfg.get('base_url', "https://api.deepseek.com")
    )
    if not api_key:
        raise ValueError(
            "❌ API Key 未配置！请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
        )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=api_cfg.get('timeout', 60)
    )

    # Setup Logging & Output (学术复现友好：每次运行独立 run_id 目录)
    dataset_name = infer_dataset_name(config)
    eval_api_root = infer_eval_api_root(config, dataset_name)
    dataset_name_lower = dataset_name.lower().replace("-", "_")
    model_name = args.model or api_cfg.get('model', 'deepseek-chat')
    shot_tag = "fewshot" if args.use_fewshot else "zeroshot"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.split}_seed{args.seed}_{shot_tag}_{sanitize_tag(model_name)}_p{os.getpid()}"
    run_dir = os.path.join(eval_api_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 默认文件落到 run_dir，避免覆盖历史结果
    if args.output_file == "eval_results_api.jsonl":
        args.output_file = os.path.join(run_dir, "eval_results.jsonl")
    ensure_parent_dir(args.output_file)

    summary_file = args.summary_file or os.path.join(run_dir, "eval_summary.json")
    ensure_parent_dir(summary_file)

    # Setup Logger
    log_file = os.path.join(run_dir, "run.log")
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True 
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting API Evaluation (Concurrency: {args.concurrency})")
    logger.info(f"🆔 Run ID: {run_id}")
    logger.info(f"📁 Run Dir: {run_dir}")
    logger.info(f"📂 Output File: {args.output_file}")
    logger.info(f"⚙️ Config: Split={args.split}, Model={args.model or api_cfg.get('model')}, Seed={args.seed}")
    logger.info(f"📜 Protocol: {args.protocol}")
    logger.info(f"🎯 Primary Metric: {args.report_primary_metric}")
    logger.info(f"🧪 Metric Spec Version: {metric_settings.get('version', '2.0')}")
    logger.info(f"🧭 Canonical Metric Mode: {args.canonical_metric_mode}")
    logger.info(f"🧠 CoT Eval Mode: {args.cot_eval_mode}")
    logger.info(f"🧩 Pipeline Mode: {args.pipeline_mode}")
    logger.info(f"🌐 API Base URL: {base_url}")

    evaluation_mode = str(config.get("evaluation", {}).get("mode", "")).strip().lower()
    if evaluation_mode not in {"scored", "prediction_only"}:
        raise ValueError(
            f"Unsupported evaluation.mode: {evaluation_mode}. "
            "Expected one of scored, prediction_only."
        )

    # Load Data
    data_dir = f"./data/raw/{dataset_name}"
    schema_path = os.path.join(data_dir, f"{dataset_name_lower}_event_schema.json")
    adapter = DuEEFinAdapter(data_path=data_dir, schema_path=schema_path)
    try:
        samples = adapter.load_data(args.split)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"无法加载 split={args.split} 的数据文件，请检查路径: {data_dir}"
        ) from exc
    valid_event_types = set(adapter.get_event_types())
    valid_roles_by_event = {
        etype: set(roles or [])
        for etype, roles in getattr(adapter, "schema", {}).items()
    } if hasattr(adapter, "schema") else None
    prompt_schema = getattr(adapter, "schema", None)
    cf_cfg = metric_settings.get("cot", {}).get("counterfactual", {})
    cf_enabled = bool(cf_cfg.get("enabled", False)) and args.cot_eval_mode == "counterfactual"
    cf_target_types = cf_cfg.get("target_types", ["number", "date", "org"])
    cf_num_perturb = max(1, int(cf_cfg.get("num_perturb", 1)))
    role_alias_map = load_role_alias_map(args.role_alias_map)
    canonical_enabled = bool(args.canonical_metric_mode != "off" and role_alias_map)
    if args.canonical_metric_mode != "off" and not role_alias_map:
        raise ValueError(
            "canonical_metric_mode requires a valid role alias map; no semantic fallback is allowed. "
            f"path={args.role_alias_map}"
        )

    if not samples:
        raise ValueError(f"未加载到任何样本: split={args.split}, path={data_dir}")

    if args.num_samples:
        samples = samples[:args.num_samples]

    has_gold_labels = any(bool(getattr(s, "events", [])) for s in samples)
    if evaluation_mode == "scored" and not has_gold_labels:
        raise ValueError(
            f"evaluation.mode=scored requires gold labels, but split={args.split} has no gold event_list."
        )
	
    # Evaluator
    evaluator = AcademicEventEvaluator(metric_settings=metric_settings)
    canonical_evaluator = (
        AcademicEventEvaluator(metric_settings=metric_settings)
        if canonical_enabled and evaluation_mode == "scored" else None
    )
    canonical_row_evaluator = (
        AcademicEventEvaluator(metric_settings=metric_settings)
        if canonical_enabled and evaluation_mode == "scored" else None
    )
    canonical_sample_rows: List[Dict[str, int]] = []
    canonical_rewrites_total = 0
    results = []
    token_stats = ZERO_USAGE.copy()
    token_stats_counterfactual = ZERO_USAGE.copy()
    counterfactual_api_calls = 0
    parse_stats = {"success": 0, "failure": 0}

    print(f"🚀 Starting API Evaluation (Mode: {'Few-shot' if args.use_fewshot else 'Zero-shot'})")

    # ThreadPool Execution
    max_retries = api_cfg.get('max_retries', 3)
    fewshot_num_examples = int(api_cfg.get('fewshot_num_examples', 3))
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample,
                client,
                model_name,
                max_retries,
                idx,
                sample,
                args.use_fewshot,
                fewshot_num_examples,
                args.json_mode,
                prompt_schema,
                args.pipeline_mode,
            ): idx
            for idx, sample in enumerate(samples)
        }
        
        for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Evaluated"):
            try:
                res = future.result()
                
                # Update Metrics
                if res["parse_success"]:
                    parse_stats["success"] += 1
                else:
                    parse_stats["failure"] += 1

                if evaluation_mode == "scored":
                    evaluator.update_with_extended_metrics(
                        pred_events=res['pred_events'],
                        gold_events=res['gold_events'],
                        source_text=res['text'],
                        full_response=res['response_text'],
                        parse_success=res["parse_success"],
                        parse_diagnostics=res["parse_diagnostics"],
                        valid_event_types=valid_event_types,
                        valid_roles_by_event=valid_roles_by_event
                    )
                    canonical_pred_events = res["pred_events"]
                    rewrite_count = 0
                    if canonical_evaluator is not None:
                        canonical_pred_events, rewrite_count = canonicalize_pred_roles(
                            res["pred_events"],
                            role_alias_map,
                        )
                        canonical_rewrites_total += rewrite_count
                        canonical_evaluator.update_with_extended_metrics(
                            pred_events=canonical_pred_events,
                            gold_events=res["gold_events"],
                            source_text=res["text"],
                            full_response=res["response_text"],
                            parse_success=res["parse_success"],
                            parse_diagnostics=res["parse_diagnostics"],
                            valid_event_types=valid_event_types,
                            valid_roles_by_event=valid_roles_by_event,
                        )
                        canonical_sample_rows.append(
                            compute_sample_metric_row(
                                canonical_row_evaluator,
                                canonical_pred_events,
                                res["gold_events"],
                            )
                        )
                else:
                    canonical_pred_events = res["pred_events"]
                    rewrite_count = 0

                if cf_enabled:
                    for _ in range(cf_num_perturb):
                        perturbed_text, perturbation = perturb_text_for_counterfactual(
                            res["text"],
                            target_types=cf_target_types,
                        )
                        if not perturbation.get("changed", False):
                            continue
                        cf_payload = build_inference_prompt_payload(
                            text=perturbed_text,
                            schema=prompt_schema,
                            use_oneshot=args.use_fewshot,
                            num_examples=fewshot_num_examples,
                        )
                        cf_messages = cf_payload["messages"]
                        cf_text, cf_usage, _, _, _ = perform_api_inference(
                            client,
                            model_name,
                            cf_messages,
                            max_retries,
                            args.json_mode,
                        )
                        counterfactual_api_calls += 1
                        token_stats_counterfactual["prompt_tokens"] += cf_usage.get("prompt_tokens", 0)
                        token_stats_counterfactual["completion_tokens"] += cf_usage.get("completion_tokens", 0)
                        token_stats_counterfactual["total_tokens"] += cf_usage.get("total_tokens", 0)
                        cf_events, _ = parse_event_list_strict_with_diagnostics(cf_text)
                        if args.pipeline_mode == "cat_lite":
                            cf_cat_result = apply_cat_lite_pipeline(
                                pred_events=cf_events,
                                source_text=perturbed_text,
                                schema=prompt_schema,
                                require_argument_in_text=True,
                            )
                            cf_events = cf_cat_result.events
                        evaluator.update_counterfactual_consistency(cf_events, perturbation)
                        if canonical_evaluator is not None:
                            cf_events_canonical, _ = canonicalize_pred_roles(cf_events, role_alias_map)
                            canonical_evaluator.update_counterfactual_consistency(
                                cf_events_canonical,
                                perturbation,
                            )
                
                # Save Result
                results.append({
                    "id": getattr(res['sample'], 'id', ''),
                    "sample_idx": res["sample_idx"],
                    "text": res['text'],
                    "gold": res['gold_events'],
                    "pred": res['pred_events'],
                    "response": res['response_text'],
                    "usage": res['usage'],
                    "parse_success": res["parse_success"],
                    "parse_error": res["parse_diagnostics"].get("error"),
                    "parse_method": res["parse_diagnostics"].get("extraction_method"),
                    "repair_steps": res["parse_diagnostics"].get("repair_steps", []),
                    "api_success": res["api_success"],
                    "api_error": res["api_error"],
                    "response_meta": res["response_meta"],
                    "pipeline_mode": args.pipeline_mode,
                    "cat_stats": res.get("cat_stats", {}),
                    "cot_eval_mode": args.cot_eval_mode,
                    "pred_canonical": canonical_pred_events if canonical_enabled else None,
                    "canonical_role_rewrites": rewrite_count if canonical_enabled else 0,
                })

                # 累加 token 统计
                token_stats["prompt_tokens"] += res['usage'].get("prompt_tokens", 0)
                token_stats["completion_tokens"] += res['usage'].get("completion_tokens", 0)
                token_stats["total_tokens"] += res['usage'].get("total_tokens", 0)
                
            except Exception as exc:
                failed_idx = future_to_sample.get(future)
                raise RuntimeError(
                    f"API evaluation sample failed: sample_idx={failed_idx}, error={exc}"
                ) from exc

    # Save Results
    results.sort(key=lambda x: (str(x.get("id", "")), int(x.get("sample_idx", 0))))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Report
    metrics_report_file: Optional[str] = None
    canonical_metrics_block: Optional[Dict[str, Any]] = None
    if evaluation_mode == "scored":
        report = evaluator.compute_metrics()
        metrics_report_file = os.path.join(run_dir, "eval_report.txt")
        log_and_save_metrics_report(
            report=report,
            logger=logger,
            report_file=metrics_report_file,
            eval_mode="both",
        )
        metrics_dict = asdict(report)
        metrics_dict["parse_error_rate"] = round(report.parse_error_rate, 4)
        metrics_dict["parse_success"] = parse_stats["success"]
        metrics_dict["parse_failure"] = parse_stats["failure"]
        metrics_dict["parse_success_rate"] = round(
            parse_stats["success"] / len(samples) if samples else 0.0,
            4,
        )
        metrics_dict["primary_metric"] = args.report_primary_metric
        metrics_dict["primary_metric_value"] = resolve_primary_metric_value(
            metrics_dict,
            args.report_primary_metric,
        )

        if args.compute_ci:
            ci_evaluator = AcademicEventEvaluator(metric_settings=metric_settings)
            sample_rows = [
                compute_sample_metric_row(ci_evaluator, item.get("pred", []), item.get("gold", []))
                for item in results
            ]
            metrics_dict["bootstrap_ci"] = bootstrap_confidence_intervals(
                sample_rows,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
                confidence=0.95,
            )

        if canonical_evaluator is not None:
            canonical_report = canonical_evaluator.compute_metrics()
            canonical_metrics_block = {
                "canonicalized_strict_precision": canonical_report.strict_precision,
                "canonicalized_strict_recall": canonical_report.strict_recall,
                "canonicalized_strict_f1": canonical_report.strict_f1,
                "canonicalized_relaxed_precision": canonical_report.relaxed_precision,
                "canonicalized_relaxed_recall": canonical_report.relaxed_recall,
                "canonicalized_relaxed_f1": canonical_report.relaxed_f1,
                "canonicalized_type_precision": canonical_report.type_precision,
                "canonicalized_type_recall": canonical_report.type_recall,
                "canonicalized_type_f1": canonical_report.type_f1,
                "canonicalized_schema_compliance_rate": canonical_report.schema_compliance_rate,
                "canonical_role_rewrites_total": canonical_rewrites_total,
                "canonical_role_rewrites_avg": (canonical_rewrites_total / len(samples)) if samples else 0.0,
            }
            if args.compute_ci and canonical_sample_rows:
                canonical_metrics_block["bootstrap_ci"] = bootstrap_confidence_intervals(
                    canonical_sample_rows,
                    n_bootstrap=args.bootstrap_samples,
                    seed=args.seed,
                    confidence=0.95,
                )
            metrics_dict["auxiliary_metrics"] = {
                "canonicalized": canonical_metrics_block
            }
    else:
        parse_success_rate = parse_stats["success"] / len(samples) if samples else 0.0
        parse_error_rate = parse_stats["failure"] / len(samples) if samples else 0.0
        metrics_dict = {
            "evaluation_mode": "prediction_only",
            "reason": f"explicit evaluation.mode=prediction_only for split={args.split}",
            "total_samples": len(samples),
            "parse_errors": parse_stats["failure"],
            "parse_error_rate": round(parse_error_rate, 4),
            "parse_success": parse_stats["success"],
            "parse_failure": parse_stats["failure"],
            "parse_success_rate": round(parse_success_rate, 4),
            "primary_metric": args.report_primary_metric,
        }
        logger.info(
            f"📌 Prediction-only summary: samples={len(samples)}, "
            f"parse_success={parse_stats['success']}, parse_failure={parse_stats['failure']}, "
            f"parse_success_rate={parse_success_rate:.4f}, parse_error_rate={parse_error_rate:.4f}"
        )
    
    avg_tokens = token_stats['total_tokens'] / len(samples) if samples else 0
    avg_tokens_counterfactual = (
        token_stats_counterfactual["total_tokens"] / counterfactual_api_calls
        if counterfactual_api_calls > 0 else 0
    )
    api_response_models = sorted({
        item.get("response_meta", {}).get("response_model")
        for item in results
        if item.get("response_meta", {}).get("response_model")
    })
    api_call_failures = sum(1 for item in results if not item.get("api_success", False))
    manifest = collect_runtime_manifest(Path(__file__).parent.resolve())
    cmdline = " ".join(os.sys.argv)
    prompt_schema_block = ChinesePromptBuilder.build_schema_constraints(prompt_schema)
    selected_fewshot_examples = (
        ChinesePromptBuilder.select_fewshot_examples(num_examples=fewshot_num_examples)
        if args.use_fewshot
        else []
    )
    prompt_hashes = {
        "system_prompt_sha256": hash_text(ChinesePromptBuilder.build_system_prompt(schema=prompt_schema)),
        "schema_constraints_sha256": hash_text(prompt_schema_block) if prompt_schema_block else None,
        "fewshot_example_indices": (
            list(range(min(fewshot_num_examples, len(ChinesePromptBuilder.FEW_SHOT_EXAMPLES))))
            if args.use_fewshot else []
        ),
        "fewshot_examples_sha256": (
            [
                {
                    "user": hash_text(ex["user"]),
                    "assistant": hash_text(ex["assistant"]),
                }
                for ex in selected_fewshot_examples
            ]
            if args.use_fewshot else []
        ),
    }
    protocol_hash = compute_file_hash(args.protocol)
    role_alias_map_hash = compute_file_hash(args.role_alias_map)
    
    eval_summary = {
        "meta": {
            "run_id": run_id,
            "run_dir": os.path.abspath(run_dir),
            "timestamp": timestamp,
            "model": model_name,
            "api_response_models": api_response_models,
            "dataset": dataset_name,
            "num_samples": len(samples),
            "split": args.split,
            "concurrency": args.concurrency,
            "has_gold_labels": has_gold_labels,
            "use_fewshot": args.use_fewshot,
            "fewshot_num_examples": fewshot_num_examples if args.use_fewshot else 0,
            "prompt_style": "profile_contract",
            "json_mode": args.json_mode,
            "cot_eval_mode": args.cot_eval_mode,
            "pipeline_mode": args.pipeline_mode,
            "seed": args.seed,
            "model_profile": model_profile,
            "config_hash_sha256": compute_config_hash(config),
            "config_path": os.path.abspath(args.config),
            "command": cmdline,
            "bootstrap_samples": args.bootstrap_samples,
            "compute_ci": bool(args.compute_ci),
            "protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "protocol_hash_sha256": protocol_hash,
            "eval_protocol_path": os.path.abspath(args.protocol) if args.protocol else None,
            "eval_protocol_hash": protocol_hash,
            "protocol_version": protocol.get("version"),
            "metric_version": metric_settings.get("version", "2.0"),
            "primary_metric": args.report_primary_metric,
            "canonical_metric_mode": args.canonical_metric_mode,
            "metric_settings": metric_settings,
            "role_alias_map_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_map_hash_sha256": role_alias_map_hash,
            "role_alias_path": os.path.abspath(args.role_alias_map) if args.role_alias_map else None,
            "role_alias_hash": role_alias_map_hash,
            "role_alias_map_loaded": bool(role_alias_map),
            "metrics_report_file": os.path.abspath(metrics_report_file) if metrics_report_file else None,
            "log_file": os.path.abspath(log_file),
            "generation": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "max_retries": max_retries,
                "json_mode": args.json_mode,
            },
            "decode_mode": "api_temperature_0",
            "seed_effective": False,
            "prompt_hashes": prompt_hashes,
            "prompt_variant": "fewshot" if args.use_fewshot else "zeroshot",
            "prompt_builder_version": str(comparison_cfg.get("prompt_builder_version", PROMPT_BUILDER_VERSION)),
            "parser_version": str(comparison_cfg.get("parser_version", PARSER_VERSION)),
            "normalization_version": str(comparison_cfg.get("normalization_version", NORMALIZATION_VERSION)),
            "training_mode": "api_inference",
            "evaluation_mode": evaluation_mode,
        },
        "metrics": metrics_dict,
        "token_usage": {
            **token_stats,
            "avg_tokens_per_sample": avg_tokens,
            "counterfactual_api_calls": counterfactual_api_calls,
            "counterfactual_prompt_tokens": token_stats_counterfactual["prompt_tokens"],
            "counterfactual_completion_tokens": token_stats_counterfactual["completion_tokens"],
            "counterfactual_total_tokens": token_stats_counterfactual["total_tokens"],
            "counterfactual_avg_tokens_per_call": avg_tokens_counterfactual,
        },
        "api_stats": {
            "failed_calls": api_call_failures,
            "failed_call_rate": (api_call_failures / len(samples)) if samples else 0.0,
        },
        "runtime": {
            "wall_clock_seconds": time.time() - run_start_ts,
        },
        "runtime_manifest": manifest,
        "analysis": {
            "primary_metric": args.report_primary_metric,
            "primary_metric_value": metrics_dict.get(args.report_primary_metric),
            "canonical_metric_mode": args.canonical_metric_mode,
            "canonical_metrics_available": canonical_metrics_block is not None,
            "metric_version": metric_settings.get("version", "2.0"),
            "protocol": protocol,
        },
    }
    
    save_json(summary_file, eval_summary)

    run_manifest = build_run_manifest(
        task="eval_api",
        status="completed",
        meta=eval_summary["meta"],
        artifacts={
            "run_dir": os.path.abspath(run_dir),
            "log_file": os.path.abspath(log_file),
            "result_file": os.path.abspath(args.output_file),
            "summary_file": os.path.abspath(summary_file),
            "metrics_report_file": os.path.abspath(metrics_report_file) if metrics_report_file else None,
        },
        contract=build_contract_record(
            model_profile=model_profile,
            model_source="api",
            effective_model_path=model_name,
        ),
        runtime={
            "wall_clock_seconds": eval_summary["runtime"]["wall_clock_seconds"],
        },
        runtime_manifest=manifest,
    )
    run_manifest_file = os.path.join(run_dir, "run_manifest.json")
    save_json(run_manifest_file, run_manifest)

    logger.info(f"📊 Saved evaluation summary to {summary_file}")
    logger.info(f"🧾 Saved run manifest to {run_manifest_file}")

if __name__ == "__main__":
    main()

