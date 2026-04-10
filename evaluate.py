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
import random
import time
import warnings
from tqdm import tqdm
from typing import List, Dict, Set, Tuple, Optional, Any

_UNSLOTH_IMPORT_ERROR: Optional[Exception] = None
try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"WARNING: Unsloth should be imported before \[transformers\].*",
            category=UserWarning,
        )
        import unsloth  # noqa: F401
except Exception as exc:  # pragma: no cover - 仅在缺少本地模型评估依赖时触发
    _UNSLOTH_IMPORT_ERROR = exc

# 导入项目模块
from oglans.data.adapter import DuEEFinAdapter
from oglans.config import ConfigManager
from oglans.evaluation import (
    AcademicEventEvaluator,
    MetricsReport,
    build_primary_metric_map,
    print_metrics_report,
)
from oglans.utils.json_parser import (
    NORMALIZATION_VERSION,
    PARSER_VERSION,
    POSTPROCESS_VERSION,
    compute_postprocess_metric_summary,
    parse_event_list_strict_with_diagnostics,
    postprocess_event_list,
    write_postprocess_diagnostics_sidecar,
)
from oglans.data.prompt_builder import (
    ChinesePromptBuilder,
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
    resolve_prompt_settings,
)
from oglans.utils.research_protocol import (
    build_fewshot_example_pool,
    extract_event_types_from_events,
    resolve_stage_settings as shared_resolve_stage_settings,
    restrict_schema_to_event_types,
    select_fewshot_pool_samples,
    validate_stage_mode,
)
from oglans.inference.cat_lite import apply_cat_lite_pipeline, perturb_text_for_counterfactual
from oglans.utils.eval_protocol import (
    canonicalize_pred_roles as shared_canonicalize_pred_roles,
    load_eval_protocol as shared_load_eval_protocol,
    load_role_alias_map as shared_load_role_alias_map,
    resolve_primary_metric_value,
    validate_primary_metric,
)
from oglans.utils.compare_contract import build_compare_contract, build_result_diagnostics
from oglans.utils.run_manifest import (
    build_contract_record,
    build_run_manifest,
    collect_runtime_manifest,
    compute_file_sha256,
    save_json,
)
from oglans.utils.model_quantization import is_quantized_model
from oglans.utils.model_profile import (
    load_local_model_profile,
    prepare_tokenizer_for_profile,
    resolve_profile_terminator_token_ids,
)

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


def _build_arg_parser():
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
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="评估摘要输出文件（默认与 output_file 同目录自动命名）",
    )
    parser.add_argument("--eval_mode", type=str, default="both", choices=["strict", "relaxed", "both"], 
                        help="评估模式: strict/relaxed/both")
    parser.add_argument(
        "--use_oneshot",
        action="store_true",
        default=None,
        help="兼容旧参数名：等价于 --prompt_variant fewshot",
    )
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default=None,
        choices=["zeroshot", "fewshot"],
        help="推理 prompt 模式（默认读取 comparison.prompt_variant）",
    )
    parser.add_argument(
        "--fewshot_num_examples",
        type=int,
        default=None,
        help="few-shot 示例数（默认读取 comparison.fewshot_num_examples）",
    )
    parser.add_argument(
        "--stage_mode",
        type=str,
        default=None,
        choices=["single_pass", "two_stage"],
        help="推理阶段模式（single_pass | two_stage）",
    )
    parser.add_argument(
        "--fewshot_selection_mode",
        type=str,
        default=None,
        choices=["static", "dynamic"],
        help="few-shot 示例选择模式",
    )
    parser.add_argument(
        "--fewshot_pool_split",
        type=str,
        default=None,
        choices=["train", "train_fit"],
        help="few-shot 检索池来源（完整 train 或 train_fit）",
    )
    parser.add_argument(
        "--train_tune_ratio",
        type=float,
        default=None,
        help="research protocol 中 train_tune 占 train 的比例",
    )
    parser.add_argument(
        "--research_split_manifest",
        type=str,
        default=None,
        help="固定 train_fit/train_tune 划分清单路径；用于保证 baseline 与方法可比",
    )
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
    return parser


def parse_args(argv=None):
    return _build_arg_parser().parse_args(argv)


def parse_args_with_unknown(argv=None):
    return _build_arg_parser().parse_known_args(argv)


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


def resolve_model_name_or_path(
    model_name_or_path: str,
    *,
    source: str,
    project_root: str,
) -> str:
    from oglans.utils.hub_runtime import resolve_model_name_or_path as shared_resolve_model_name_or_path

    return shared_resolve_model_name_or_path(
        model_name_or_path,
        source=source,
        project_root=project_root,
    )


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


def resolve_stage_settings(
    *,
    stage_mode: Optional[str] = None,
    fewshot_selection_mode: Optional[str] = None,
    fewshot_pool_split: Optional[str] = None,
    comparison_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    return shared_resolve_stage_settings(
        stage_mode=stage_mode,
        fewshot_selection_mode=fewshot_selection_mode,
        fewshot_pool_split=fewshot_pool_split,
        comparison_cfg=comparison_cfg,
        default_stage_mode="single_pass",
        default_selection_mode="dynamic",
        default_pool_split="train_fit",
    )


def count_non_pad_tokens(batch_ids, pad_token_id: Optional[int]) -> int:
    if batch_ids is None:
        return 0
    if pad_token_id is None:
        return int(batch_ids.numel())
    return int((batch_ids != pad_token_id).sum().item())


def main(argv=None):
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
        ) from (_UNSLOTH_IMPORT_ERROR or e)
    from oglans.utils.hub_runtime import (
        build_unsloth_from_pretrained_kwargs,
        configure_model_download_runtime,
        get_model_download_runtime_snapshot,
    )
    from oglans.utils.scv import evaluate_scv_lite

    args, unknown = parse_args_with_unknown(argv)
    validate_eval_args(args)
    run_start_ts = time.time()
    cmdline = " ".join(os.sys.argv if argv is None else ["evaluate.py", *argv])
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
    config = ConfigManager().load_config(args.config, unknown)
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
    prompt_settings = resolve_prompt_settings(
        prompt_variant=args.prompt_variant,
        fewshot_num_examples=args.fewshot_num_examples,
        use_oneshot=args.use_oneshot,
        default_prompt_variant=str(comparison_cfg.get("prompt_variant", "zeroshot")),
        default_num_examples=int(comparison_cfg.get("fewshot_num_examples", 3)),
    )
    prompt_variant = prompt_settings["prompt_variant"]
    use_fewshot = bool(prompt_settings["use_oneshot"])
    fewshot_num_examples = int(prompt_settings["fewshot_num_examples"])
    stage_settings = resolve_stage_settings(
        stage_mode=args.stage_mode,
        fewshot_selection_mode=args.fewshot_selection_mode,
        fewshot_pool_split=args.fewshot_pool_split,
        comparison_cfg=comparison_cfg,
    )
    args.stage_mode = stage_settings["stage_mode"]
    args.fewshot_selection_mode = stage_settings["fewshot_selection_mode"]
    args.fewshot_pool_split = stage_settings["fewshot_pool_split"]
    if args.train_tune_ratio is None:
        args.train_tune_ratio = float(comparison_cfg.get("train_tune_ratio", 0.1))
    args.research_split_manifest = (
        args.research_split_manifest
        or comparison_cfg.get("research_split_manifest_path")
    )
    if args.report_primary_metric is None:
        args.report_primary_metric = str(protocol.get("primary_metric", "doc_role_micro_f1"))
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
    print(f"🪜 Stage Mode: {args.stage_mode}")
    if args.research_split_manifest:
        print(f"🧪 Research Split Manifest: {args.research_split_manifest}")
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
    fewshot_example_pool = None
    fewshot_split_manifest = {
        "seed": args.seed,
        "tune_ratio": float(args.train_tune_ratio),
        "pool_split": args.fewshot_pool_split,
        "fit_ids": [],
        "tune_ids": [],
        "fit_count": 0,
        "tune_count": 0,
        "manifest_path": args.research_split_manifest,
    }
    if use_fewshot:
        retrieval_samples = adapter.load_data("train")
        pool_source_samples, fewshot_split_manifest = select_fewshot_pool_samples(
            retrieval_samples,
            pool_split=args.fewshot_pool_split,
            tune_ratio=args.train_tune_ratio,
            seed=args.seed,
            split_manifest=args.research_split_manifest,
        )
        fewshot_example_pool = build_fewshot_example_pool(
            pool_source_samples,
            schema=getattr(adapter, "schema", None),
            source_split=args.fewshot_pool_split,
        )
        print(
            f"   Few-shot 检索池: mode={args.fewshot_selection_mode}, "
            f"pool_split={args.fewshot_pool_split}, size={len(fewshot_example_pool)}"
        )
    has_gold_labels = any(bool(getattr(s, "events", [])) for s in all_samples)
    if evaluation_mode == "scored" and not has_gold_labels:
        raise ValueError(
            f"evaluation.mode=scored requires gold labels, but split={args.split} has no gold event_list."
        )
    valid_types = set(adapter.get_event_types()) if hasattr(adapter, 'get_event_types') else None
    valid_roles_by_event = None
    role_order_by_event = None
    if hasattr(adapter, 'schema') and isinstance(adapter.schema, dict):
        role_order_by_event = {
            etype: list(roles or [])
            for etype, roles in adapter.schema.items()
        }
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
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "stage1_prompt_tokens": 0,
        "stage1_completion_tokens": 0,
        "stage1_total_tokens": 0,
        "stage2_prompt_tokens": 0,
        "stage2_completion_tokens": 0,
        "stage2_total_tokens": 0,
    }
    stage_runtime = {
        "stage1_wall_clock_seconds": 0.0,
        "stage2_wall_clock_seconds": 0.0,
    }

    # 6. 批量推理
    decoding_strategy = "采样解码 (Sampling)" if args.do_sample else "确定性解码 (Greedy)"
    print(f"\n🚀 开始推理 (Batch Size: {args.batch_size}, 解码策略: {decoding_strategy})...")
    pbar = tqdm(range(0, len(all_samples), args.batch_size), desc="评估进度")

    for i in pbar:
        batch_samples = all_samples[i:i + args.batch_size]
        stage1_payloads = []
        stage1_event_types_by_sample: List[List[str]] = [[] for _ in batch_samples]
        stage1_parse_successes: List[Optional[bool]] = [None for _ in batch_samples]
        stage1_parse_errors: List[Optional[str]] = [None for _ in batch_samples]
        stage2_schema_by_sample = []

        if args.stage_mode == "two_stage":
            for sample in batch_samples:
                stage1_payload = ChinesePromptBuilder.build_event_type_payload(
                    text=sample.text,
                    schema=getattr(adapter, "schema", None),
                    tokenizer=tokenizer,
                )
                stage1_payloads.append(stage1_payload["formatted_text"])

            stage1_inputs = tokenizer(
                stage1_payloads,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config['model'].get('max_seq_length', 4096)
            ).to(device)
            token_usage["stage1_prompt_tokens"] += count_non_pad_tokens(
                stage1_inputs.input_ids,
                tokenizer.pad_token_id,
            )
            stage1_generate_kwargs = {
                "max_new_tokens": 256,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": (
                    terminator_token_ids[0]
                    if len(terminator_token_ids) == 1
                    else terminator_token_ids
                ),
                "do_sample": False,
            }
            stage1_start = time.perf_counter()
            with torch.no_grad():
                stage1_outputs = model.generate(**stage1_inputs, **stage1_generate_kwargs)
            stage_runtime["stage1_wall_clock_seconds"] += time.perf_counter() - stage1_start
            stage1_generated_ids = stage1_outputs[:, stage1_inputs.input_ids.shape[1]:]
            token_usage["stage1_completion_tokens"] += count_non_pad_tokens(
                stage1_generated_ids,
                tokenizer.pad_token_id,
            )
            token_usage["stage1_total_tokens"] = (
                token_usage["stage1_prompt_tokens"] + token_usage["stage1_completion_tokens"]
            )
            stage1_decoded = tokenizer.batch_decode(stage1_generated_ids, skip_special_tokens=True)

            for idx, response in enumerate(stage1_decoded):
                stage1_events, stage1_diagnostics = parse_event_list_strict_with_diagnostics(response)
                stage1_event_types_by_sample[idx] = extract_event_types_from_events(
                    stage1_events,
                    valid_event_types=list(valid_types or []),
                )
                stage1_parse_successes[idx] = bool(stage1_diagnostics.get("success", False))
                stage1_parse_errors[idx] = stage1_diagnostics.get("error")
                stage2_schema, _ = restrict_schema_to_event_types(
                    getattr(adapter, "schema", None),
                    stage1_event_types_by_sample[idx],
                )
                stage2_schema_by_sample.append(stage2_schema)
        else:
            stage2_schema_by_sample = [getattr(adapter, "schema", None) for _ in batch_samples]

        batch_prompts = []
        batch_prompt_meta = []
        for sample, stage2_schema in zip(batch_samples, stage2_schema_by_sample):
            prompt_payload = build_inference_prompt_payload(
                text=sample.text,
                tokenizer=tokenizer,
                prompt_variant=prompt_variant,
                use_oneshot=use_fewshot,
                schema=stage2_schema,
                num_examples=fewshot_num_examples,
                fewshot_selection_mode=args.fewshot_selection_mode,
                fewshot_example_pool=fewshot_example_pool,
            )
            batch_prompts.append(prompt_payload["formatted_text"])
            batch_prompt_meta.append(prompt_payload)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config['model'].get('max_seq_length', 4096)
        ).to(device)
        token_usage["stage2_prompt_tokens"] += count_non_pad_tokens(
            inputs.input_ids,
            tokenizer.pad_token_id,
        )
        
        # 推理
        stage2_start = time.perf_counter()
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
        stage_runtime["stage2_wall_clock_seconds"] += time.perf_counter() - stage2_start
        
        # 解码
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        token_usage["stage2_completion_tokens"] += count_non_pad_tokens(
            generated_ids,
            tokenizer.pad_token_id,
        )
        token_usage["stage2_total_tokens"] = (
            token_usage["stage2_prompt_tokens"] + token_usage["stage2_completion_tokens"]
        )
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 处理每个样本
        for j, response in enumerate(decoded_responses):
            sample = batch_samples[j]
            stage2_schema_event_types = list((stage2_schema_by_sample[j] or {}).keys()) if isinstance(stage2_schema_by_sample[j], dict) else []
            
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
                    valid_roles_by_event=valid_roles_by_event,
                    role_order_by_event=role_order_by_event,
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
                    role_order_by_event=role_order_by_event,
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
                        prompt_variant=prompt_variant,
                        use_oneshot=use_fewshot,
                        schema=getattr(adapter, "schema", None),
                        num_examples=fewshot_num_examples,
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
                "prompt_meta": {
                    "fewshot_selection_mode": batch_prompt_meta[j].get("fewshot_selection_mode", "none"),
                    "fewshot_example_ids": batch_prompt_meta[j].get("fewshot_example_ids", []),
                    "fewshot_count": batch_prompt_meta[j].get("fewshot_count", 0),
                },
                "stage_meta": {
                    "stage_mode": args.stage_mode,
                    "stage1_predicted_event_types": stage1_event_types_by_sample[j],
                    "stage1_parse_success": stage1_parse_successes[j],
                    "stage1_parse_error": stage1_parse_errors[j],
                    "stage2_schema_event_types": stage2_schema_event_types,
                },
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
    primary_metric_values = build_primary_metric_map(report)
    rounded_primary_metric_values = {
        key: round(value, 4)
        for key, value in primary_metric_values.items()
    }
    legacy_metrics_block = {
        "strict": {
            "precision": round(report.strict_precision, 4),
            "recall": round(report.strict_recall, 4),
            "f1": round(report.strict_f1, 4),
        },
        "relaxed": {
            "precision": round(report.relaxed_precision, 4),
            "recall": round(report.relaxed_recall, 4),
            "f1": round(report.relaxed_f1, 4),
        },
        "type_identification": {
            "precision": round(report.type_precision, 4),
            "recall": round(report.type_recall, 4),
            "f1": round(report.type_f1, 4),
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
            "entity_rate": round(report.hallucination_entity_rate, 4),
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
    }
    academic_metrics_block = {
        "doc_ee": report.doc_ee,
        "ee_text_proxy": report.ee_text_proxy,
        "primary_metric": args.report_primary_metric,
        "primary_metric_value": round(
            resolve_primary_metric_value(primary_metric_values, args.report_primary_metric),
            4,
        ),
    }

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
        **rounded_primary_metric_values,
        **legacy_metrics_block,
        "legacy_metrics": legacy_metrics_block,
        "academic_metrics": academic_metrics_block,
        "primary_metric": args.report_primary_metric,
        "primary_metric_value": round(
            resolve_primary_metric_value(primary_metric_values, args.report_primary_metric),
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
    prompt_hashes = {
        "system_prompt_sha256": hash_text(ChinesePromptBuilder.build_system_prompt(schema=prompt_schema)),
        "schema_constraints_sha256": hash_text(prompt_schema_block) if prompt_schema_block else None,
        "research_split_manifest_sha256": (
            safe_compute_file_sha256(args.research_split_manifest)
            if args.research_split_manifest
            else None
        ),
        "fewshot_selection_mode": args.fewshot_selection_mode if use_fewshot else "none",
        "fewshot_retrieval_pool_size": len(fewshot_example_pool or []) if use_fewshot else 0,
        "fewshot_pool_split": args.fewshot_pool_split if use_fewshot else "none",
        "stage_mode": args.stage_mode,
    }
    primary_metric_value = round(
        resolve_primary_metric_value(primary_metric_values, args.report_primary_metric),
        4,
    )
    diagnostics_block = build_result_diagnostics(results_to_save)
    diagnostics_block.update(
        {
            "schema_compliance_rate": round(report.schema_compliance_rate, 4),
            "hallucination_rate": round(report.hallucination_rate, 4),
            "hallucination_entity_rate": round(report.hallucination_entity_rate, 4),
        }
    )
    total_tokens = token_usage["stage1_total_tokens"] + token_usage["stage2_total_tokens"]
    avg_tokens_per_sample = (total_tokens / report.total_samples) if report.total_samples else 0.0
    cost_block = {
        "prompt_tokens": token_usage["stage1_prompt_tokens"] + token_usage["stage2_prompt_tokens"],
        "completion_tokens": token_usage["stage1_completion_tokens"] + token_usage["stage2_completion_tokens"],
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "token_usage_kind": "estimated",
        "f1_per_1k_tokens": ((primary_metric_value * 1000.0) / total_tokens) if total_tokens else None,
    }
    compare_block = build_compare_contract(
        {
            "model_family": "local_base" if args.base_only else "local_checkpoint",
            "model_kind": "base_only" if args.base_only else "adapter_checkpoint",
            "split": args.split,
            "primary_metric": args.report_primary_metric,
            "stage_mode": args.stage_mode,
            "prompt_variant": prompt_variant,
            "fewshot_num_examples": fewshot_num_examples if use_fewshot else 0,
            "fewshot_selection_mode": args.fewshot_selection_mode if use_fewshot else "none",
            "fewshot_pool_split": args.fewshot_pool_split if use_fewshot else "none",
            "train_tune_ratio": float(args.train_tune_ratio),
            "research_split_manifest_path": (
                os.path.abspath(args.research_split_manifest) if args.research_split_manifest else "none"
            ),
            "research_split_manifest_hash": (
                safe_compute_file_sha256(args.research_split_manifest) if args.research_split_manifest else "none"
            ),
            "pipeline_mode": args.pipeline_mode,
            "canonical_metric_mode": args.canonical_metric_mode,
            "protocol_hash": safe_compute_file_sha256(args.protocol) or "none",
            "role_alias_hash": safe_compute_file_sha256(args.role_alias_map) or "none",
            "seed": args.seed,
            "seed_effective": bool(args.do_sample),
            "token_usage_kind": "estimated",
        }
    )

    # 新版统一摘要结构（与 evaluate_api.py 对齐）
    summary_file = args.summary_file or final_output_path.replace(".jsonl", "_summary.json")
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
            "use_fewshot": use_fewshot,
            "fewshot_num_examples": fewshot_num_examples if use_fewshot else 0,
            "fewshot_selection_mode": args.fewshot_selection_mode if use_fewshot else "none",
            "fewshot_pool_split": args.fewshot_pool_split if use_fewshot else "none",
            "train_tune_ratio": float(args.train_tune_ratio),
            "fewshot_split_manifest": fewshot_split_manifest if use_fewshot else None,
            "research_split_manifest_path": (
                os.path.abspath(args.research_split_manifest)
                if args.research_split_manifest
                else None
            ),
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
            "stage_mode": args.stage_mode,
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
            "prompt_variant": prompt_variant,
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
        "compare": compare_block,
        "metrics": {
            **rounded_primary_metric_values,
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
            "legacy_metrics": legacy_metrics_block,
            "academic_metrics": academic_metrics_block,
            "primary_metric": args.report_primary_metric,
            "primary_metric_value": primary_metric_value,
        },
        "diagnostics": diagnostics_block,
        "cost": cost_block,
        "token_usage": {
            "prompt_tokens": token_usage["stage1_prompt_tokens"] + token_usage["stage2_prompt_tokens"],
            "completion_tokens": token_usage["stage1_completion_tokens"] + token_usage["stage2_completion_tokens"],
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens_per_sample,
            "prompt_tokens_estimated": token_usage["stage1_prompt_tokens"] + token_usage["stage2_prompt_tokens"],
            "completion_tokens_estimated": token_usage["stage1_completion_tokens"] + token_usage["stage2_completion_tokens"],
            "total_tokens_estimated": total_tokens,
            "stage1_prompt_tokens": token_usage["stage1_prompt_tokens"],
            "stage1_completion_tokens": token_usage["stage1_completion_tokens"],
            "stage1_total_tokens": token_usage["stage1_total_tokens"],
            "stage2_prompt_tokens": token_usage["stage2_prompt_tokens"],
            "stage2_completion_tokens": token_usage["stage2_completion_tokens"],
            "stage2_total_tokens": token_usage["stage2_total_tokens"],
        },
        "api_stats": {
            "failed_calls": 0,
            "failed_call_rate": 0.0,
        },
        "runtime": {
            "wall_clock_seconds": wall_clock_seconds,
            "samples_per_second": (report.total_samples / wall_clock_seconds) if wall_clock_seconds > 0 else 0.0,
            "seconds_per_100_samples": (wall_clock_seconds * 100.0 / report.total_samples) if report.total_samples else 0.0,
            "stage1_wall_clock_seconds": round(stage_runtime["stage1_wall_clock_seconds"], 6),
            "stage2_wall_clock_seconds": round(stage_runtime["stage2_wall_clock_seconds"], 6),
            "end_to_end_wall_clock_seconds": wall_clock_seconds,
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
