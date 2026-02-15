import os
import sys
import torch
from unsloth import FastLanguageModel

sys.path.append(os.getcwd())

from oglans.data.prompt_builder import ChinesePromptBuilder


def test_base_inference():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"🔄 Loading {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)

    text = "2024年1月，阿里巴巴集团宣布回购100亿美元股份。"

    messages = [
        {"role": "system", "content": ChinesePromptBuilder.build_system_prompt()},
        {"role": "user", "content": ChinesePromptBuilder.build_user_prompt(text)},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    print("🚀 Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n[Output]:")
    print(response)

    if "```json" in response and "股份回购" in response:
        print("\n✅ Base model understands the prompt template.")
    else:
        print("\n❌ Base model struggled with the format.")


if __name__ == "__main__":
    test_base_inference()
