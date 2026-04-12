# src/utils/__init__.py
"""
工具模块导出
"""
from .logger import setup_logger
from .run_manifest import (
    append_validation_error,
    build_contract_record,
    build_run_manifest,
    collect_runtime_manifest,
    compute_file_sha256,
    compute_json_sha256,
    filter_wrapper_cli_args,
    load_effective_config_metadata,
    make_validation_error,
    save_json,
)
from .hub_runtime import (
    configure_model_download_runtime,
    configure_modelscope_runtime,
    get_model_download_runtime_snapshot,
    get_modelscope_runtime_snapshot,
    resolve_model_name_or_path,
)
from .model_profile import (
    DEFAULT_LOCAL_MODEL_PROFILE,
    LOCAL_MODEL_PROFILES,
    LocalModelProfile,
    ModelProfileError,
    load_local_model_profile,
    prepare_tokenizer_for_profile,
    resolve_profile_terminator_token_ids,
)

_dscns_import_error = None
_scv_import_error = None
_json_parser_import_error = None
try:
    from .ds_cns import DSCNSampler
except Exception as exc:  # pragma: no cover - 可选依赖缺失时触发
    _dscns_import_error = exc
try:
    from .scv import SemanticConsistencyVerifier
except Exception as exc:  # pragma: no cover - 可选依赖缺失时触发
    _scv_import_error = exc
try:
    from .json_parser import (
        compute_postprocess_metric_summary,
        parse_event_list_strict,
        parse_event_list_with_diagnostics,
        RobustJSONParser,
        postprocess_event_list,
        parse_llm_output,
        parse_with_diagnostics,
        validate_event_structure,
        write_postprocess_diagnostics_sidecar,
        LLMSelfCorrector
    )
except Exception as exc:  # pragma: no cover - 可选依赖缺失时触发
    _json_parser_import_error = exc

__all__ = [
    # 日志
    "setup_logger",
    
    # 算法
    "DSCNSampler",
    "SemanticConsistencyVerifier",

    # 复现性清单
    "build_run_manifest",
    "build_contract_record",
    "collect_runtime_manifest",
    "compute_file_sha256",
    "compute_json_sha256",
    "make_validation_error",
    "append_validation_error",
    "filter_wrapper_cli_args",
    "load_effective_config_metadata",
    "save_json",
    "configure_model_download_runtime",
    "configure_modelscope_runtime",
    "get_model_download_runtime_snapshot",
    "get_modelscope_runtime_snapshot",
    "resolve_model_name_or_path",
    "DEFAULT_LOCAL_MODEL_PROFILE",
    "LOCAL_MODEL_PROFILES",
    "LocalModelProfile",
    "ModelProfileError",
    "load_local_model_profile",
    "prepare_tokenizer_for_profile",
    "resolve_profile_terminator_token_ids",
    
    # JSON 解析
    "RobustJSONParser",
    "parse_event_list_strict",
    "parse_event_list_with_diagnostics",
    "postprocess_event_list",
    "compute_postprocess_metric_summary",
    "write_postprocess_diagnostics_sidecar",
    "parse_llm_output",
    "parse_with_diagnostics",
    "validate_event_structure",
    "LLMSelfCorrector",
]


def __getattr__(name):
    if name == "DSCNSampler":
        if "DSCNSampler" in globals():
            return globals()[name]
        raise ImportError("DSCNSampler 依赖未安装（例如 networkx）。") from _dscns_import_error
    if name == "SemanticConsistencyVerifier":
        if "SemanticConsistencyVerifier" in globals():
            return globals()[name]
        raise ImportError("SemanticConsistencyVerifier 依赖未安装（例如 transformers/torch）。") from _scv_import_error
    if name in {
        "RobustJSONParser",
        "parse_event_list_strict",
        "parse_event_list_with_diagnostics",
        "postprocess_event_list",
        "compute_postprocess_metric_summary",
        "write_postprocess_diagnostics_sidecar",
        "parse_llm_output",
        "parse_with_diagnostics",
        "validate_event_structure",
        "LLMSelfCorrector",
    }:
        if name in globals():
            return globals()[name]
        raise ImportError("json_parser 依赖未安装（例如 dirtyjson）。") from _json_parser_import_error
    raise AttributeError(f"module 'oglans.utils' has no attribute '{name}'")
