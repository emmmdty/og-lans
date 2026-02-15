# src/utils/__init__.py
"""
工具模块导出
"""
from .logger import setup_logger
from .run_manifest import (
    build_run_manifest,
    collect_runtime_manifest,
    compute_file_sha256,
    compute_json_sha256,
    save_json,
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
        RobustJSONParser,
        parse_llm_output,
        parse_with_diagnostics,
        validate_event_structure,
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
    "collect_runtime_manifest",
    "compute_file_sha256",
    "compute_json_sha256",
    "save_json",
    
    # JSON 解析
    "RobustJSONParser",
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
        "parse_llm_output",
        "parse_with_diagnostics",
        "validate_event_structure",
        "LLMSelfCorrector",
    }:
        if name in globals():
            return globals()[name]
        raise ImportError("json_parser 依赖未安装（例如 dirtyjson）。") from _json_parser_import_error
    raise AttributeError(f"module 'oglans.utils' has no attribute '{name}'")
