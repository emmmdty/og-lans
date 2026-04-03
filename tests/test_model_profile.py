import importlib

import pytest


def test_load_qwen3_profile_defaults():
    module = importlib.import_module("oglans.utils.model_profile")

    profile = module.load_local_model_profile("qwen3_instruct")

    assert profile.name == "qwen3_instruct"
    assert profile.expected_eos_token == "<|im_end|>"
    assert profile.padding_side_train == "left"
    assert profile.padding_side_eval == "left"
    assert profile.chat_template_required is True
    assert profile.supports_unsloth is True


def test_unknown_profile_fails_fast():
    module = importlib.import_module("oglans.utils.model_profile")

    with pytest.raises(module.ModelProfileError, match="Unknown local model profile"):
        module.load_local_model_profile("llama3_unknown")


def test_prepare_tokenizer_for_profile_requires_matching_eos():
    module = importlib.import_module("oglans.utils.model_profile")
    profile = module.load_local_model_profile("qwen3_instruct")

    class DummyTokenizer:
        eos_token = "</s>"
        eos_token_id = 2
        pad_token = None
        pad_token_id = None
        padding_side = "right"

        def apply_chat_template(self, *args, **kwargs):
            return "ok"

    tokenizer = DummyTokenizer()
    with pytest.raises(module.ModelProfileError, match="expected eos token"):
        module.prepare_tokenizer_for_profile(tokenizer, profile, mode="train")


def test_prepare_tokenizer_for_profile_sets_pad_and_padding_side():
    module = importlib.import_module("oglans.utils.model_profile")
    profile = module.load_local_model_profile("qwen3_instruct")

    class DummyTokenizer:
        eos_token = "<|im_end|>"
        eos_token_id = 151645
        pad_token = None
        pad_token_id = None
        padding_side = "right"

        def apply_chat_template(self, *args, **kwargs):
            return "ok"

    tokenizer = DummyTokenizer()
    prepared = module.prepare_tokenizer_for_profile(tokenizer, profile, mode="eval")

    assert prepared.pad_token == "<|im_end|>"
    assert prepared.padding_side == "left"
