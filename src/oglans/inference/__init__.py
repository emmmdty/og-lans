"""
Inference utilities.
"""

from .cat_lite import (
    CatLiteResult,
    apply_cat_lite_pipeline,
    perturb_text_for_counterfactual,
)
from .record_corrector import (
    RecordCorrectionResult,
    StructuredPipelineResult,
    apply_record_corrector,
    apply_structured_event_pipeline,
    validate_pipeline_mode,
)

__all__ = [
    "CatLiteResult",
    "RecordCorrectionResult",
    "StructuredPipelineResult",
    "apply_cat_lite_pipeline",
    "apply_record_corrector",
    "apply_structured_event_pipeline",
    "perturb_text_for_counterfactual",
    "validate_pipeline_mode",
]
