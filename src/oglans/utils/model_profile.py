"""
Strict local model profile definitions shared by training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


class ModelProfileError(ValueError):
    """Raised when a local model profile contract is violated."""


@dataclass(frozen=True)
class LocalModelProfile:
    name: str
    expected_eos_token: str
    pad_token_policy: str
    padding_side_train: str
    padding_side_eval: str
    supports_unsloth: bool
    supports_lora_adapter: bool
    generation_terminators: List[str]
    chat_template_required: bool


DEFAULT_LOCAL_MODEL_PROFILE = "qwen3_instruct"


LOCAL_MODEL_PROFILES: Dict[str, LocalModelProfile] = {
    "qwen3_instruct": LocalModelProfile(
        name="qwen3_instruct",
        expected_eos_token="<|im_end|>",
        pad_token_policy="eos_as_pad",
        padding_side_train="left",
        padding_side_eval="left",
        supports_unsloth=True,
        supports_lora_adapter=True,
        generation_terminators=["<|im_end|>"],
        chat_template_required=True,
    ),
}


def load_local_model_profile(profile_name: str) -> LocalModelProfile:
    normalized = str(profile_name or "").strip()
    profile = LOCAL_MODEL_PROFILES.get(normalized)
    if profile is None:
        raise ModelProfileError(f"Unknown local model profile: {normalized}")
    return profile


def prepare_tokenizer_for_profile(tokenizer: Any, profile: LocalModelProfile, *, mode: str) -> Any:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"train", "eval"}:
        raise ModelProfileError(f"Unsupported tokenizer preparation mode: {mode}")
    if tokenizer is None:
        raise ModelProfileError("Tokenizer is required for local model profile preparation.")

    if profile.chat_template_required and not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ModelProfileError(
            f"Local model profile '{profile.name}' requires tokenizer.apply_chat_template()."
        )

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        raise ModelProfileError(
            f"Local model profile '{profile.name}' expected eos token {profile.expected_eos_token!r}, got empty."
        )
    if eos_token != profile.expected_eos_token:
        raise ModelProfileError(
            f"Local model profile '{profile.name}' expected eos token "
            f"{profile.expected_eos_token!r}, got {eos_token!r}."
        )

    if profile.pad_token_policy == "eos_as_pad" and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = eos_token

    tokenizer.padding_side = (
        profile.padding_side_train if normalized_mode == "train" else profile.padding_side_eval
    )
    return tokenizer


def resolve_profile_terminator_token_ids(tokenizer: Any, profile: LocalModelProfile) -> List[int]:
    token_ids: List[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids.append(int(eos_token_id))
    return token_ids
