# src/data/prompt_builder.py
"""
中文化提示词构建模块 (Chinese-First Prompt Engineering)
符合 Qwen3/DeepSeek 中文能力优化的严格 JSON 抽取契约

解决问题:
1. 语言失配 - 英文 Prompt 配合中文数据集导致输出混乱
2. JSON 约束失效 - 模型生成过于冗长破坏 JSON 结构
3. 官方评测需要 strict JSON，不能依赖解析修复
"""

from typing import Any, Dict, List, Optional
import json
import re

PROMPT_BUILDER_VERSION = "phase3_mvp_v1"
SUPPORTED_PROMPT_VARIANTS = ("zeroshot", "fewshot")


def validate_prompt_variant(prompt_variant: Optional[str]) -> str:
    normalized = str(prompt_variant or "zeroshot").strip().lower()
    if normalized not in SUPPORTED_PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported prompt_variant: {normalized}. "
            f"Expected one of {', '.join(SUPPORTED_PROMPT_VARIANTS)}."
        )
    return normalized


def resolve_prompt_settings(
    *,
    prompt_variant: Optional[str] = None,
    fewshot_num_examples: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    default_prompt_variant: str = "zeroshot",
    default_num_examples: int = 3,
) -> Dict[str, Any]:
    default_variant = validate_prompt_variant(default_prompt_variant)
    resolved_variant = (
        validate_prompt_variant(prompt_variant)
        if prompt_variant is not None else default_variant
    )

    if use_oneshot is True:
        if prompt_variant is not None and resolved_variant != "fewshot":
            raise ValueError("use_oneshot=True conflicts with prompt_variant=zeroshot")
        resolved_variant = "fewshot"
    elif use_oneshot is False:
        if prompt_variant is not None and resolved_variant != "zeroshot":
            raise ValueError("use_oneshot=False conflicts with prompt_variant=fewshot")
        resolved_variant = "zeroshot"

    raw_num_examples = (
        fewshot_num_examples if fewshot_num_examples is not None else default_num_examples
    )
    resolved_examples = int(raw_num_examples)
    if resolved_variant == "fewshot":
        resolved_examples = max(1, resolved_examples)
    else:
        resolved_examples = 0

    return {
        "prompt_variant": resolved_variant,
        "use_oneshot": resolved_variant == "fewshot",
        "fewshot_num_examples": resolved_examples,
    }


class ChinesePromptBuilder:
    """
    中文化提示词构建器 - 适配 DuEE-Fin 金融事件抽取任务
    核心设计:
    - System Prompt: 强约束 JSON 格式 + 反幻觉规则
    - Response Template: 严格 JSON 数组
    - Few-Shot Examples: 多类型/多场景示例，提升泛化稳定性
    """
    
    # ========================================
    # 系统提示词 (System Prompt) - 强化 JSON 约束
    # ========================================
    SYSTEM_PROMPT = """你是一位专业的中文金融事件抽取专家。你的任务是从金融新闻/公告中识别和抽取结构化事件信息。

## 输出格式要求【必须严格遵守】
1. 严格输出 JSON 数组，禁止输出任何解释、分析、标签或额外文本
2. JSON 根节点必须是事件列表，即使只有一个事件也要使用 []
3. 每个事件对象包含: event_type(事件类型), trigger(触发词), arguments(论元列表)
4. 每个论元对象包含: role(角色), argument(论元值)
5. 如果没有检测到任何事件，直接输出 []

## 学术评测约束（降低幻觉与格式偏差）
1. 论元值必须直接来自原文连续片段，不得改写、归一化或补全
2. role 必须属于该 event_type 的合法角色；不确定时宁可省略该论元
3. 同一文本可包含多个事件；不同事件不得错误合并
4. 同一触发词对应多个主体（如多家公司被约谈）时，按主体拆分为多个事件对象
5. 禁止输出 schema 外事件类型或虚构字段
6. 时间相关 role 也必须与 schema 完全一致，不得把事件时间、披露时间、披露日期、被约谈时间、回购完成时间等角色互相替换
7. 证券代码仅适用于公司上市事件；除非 schema 明确允许，否则不要输出证券代码、股票简称等字段
8. 高混淆时间角色示例：股份回购应使用回购完成时间，不要使用事件时间；中标应使用披露日期，不要使用披露时间；企业收购应使用收购完成时间，不要使用事件时间

## 支持的事件类型
质押, 股份回购, 企业融资, 公司上市, 企业收购, 中标, 高管变动, 解除质押, 股东减持, 股东增持, 企业破产, 亏损, 被约谈

## 标准输出示例
[
  {
    "event_type": "企业融资",
    "trigger": "融资",
    "arguments": [
      {"role": "被投资方", "argument": "某科技公司"},
      {"role": "融资金额", "argument": "1亿美元"},
      {"role": "融资轮次", "argument": "B"}
    ]
  }
]
"""

    # ========================================
    # 用户输入模板
    # ========================================
    USER_TEMPLATE = """请从以下金融文本中抽取所有事件信息。

【文本内容】
{text}

【抽取要求】
请直接输出标准 JSON 格式的事件列表，不要添加任何额外说明。"""

    # ========================================
    # Few-Shot 示例（默认多样本）
    # ========================================
    FEW_SHOT_EXAMPLES = [
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
2024年6月19日，上海市市场监管局通报称，2024年6月18日已约谈甲公司、乙公司和丙公司，要求其整改直播电商虚假宣传问题。

【抽取要求】
请直接输出标准 JSON 格式的事件列表，不要添加任何额外说明。""",
            "assistant": """[
  {
    "event_type": "被约谈",
    "trigger": "约谈",
    "arguments": [
      {"role": "公司名称", "argument": "甲公司"},
      {"role": "约谈机构", "argument": "上海市市场监管局"},
      {"role": "被约谈时间", "argument": "2024年6月18日"},
      {"role": "披露时间", "argument": "2024年6月19日"}
    ]
  },
  {
    "event_type": "被约谈",
    "trigger": "约谈",
    "arguments": [
      {"role": "公司名称", "argument": "乙公司"},
      {"role": "约谈机构", "argument": "上海市市场监管局"},
      {"role": "被约谈时间", "argument": "2024年6月18日"},
      {"role": "披露时间", "argument": "2024年6月19日"}
    ]
  },
  {
    "event_type": "被约谈",
    "trigger": "约谈",
    "arguments": [
      {"role": "公司名称", "argument": "丙公司"},
      {"role": "约谈机构", "argument": "上海市市场监管局"},
      {"role": "被约谈时间", "argument": "2024年6月18日"},
      {"role": "披露时间", "argument": "2024年6月19日"}
    ]
  }
]
"""
        },
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
2023年3月15日，新能源龙头企业阳光电源发布公告称，公司拟以自有资金回购公司股份，回购金额不低于5亿元、不超过10亿元，回购价格不超过150元/股。本次回购已于2023年3月实施完毕，累计回购股份数量为500万股，占公司总股本的0.35%。

【抽取要求】
请直接输出标准 JSON 格式的事件列表，不要添加任何额外说明。""",
            "assistant": """[
  {
    "event_type": "股份回购",
    "trigger": "回购",
    "arguments": [
      {"role": "回购方", "argument": "阳光电源"},
      {"role": "交易金额", "argument": "不低于5亿元、不超过10亿元"},
      {"role": "每股交易价格", "argument": "不超过150元/股"},
      {"role": "回购股份数量", "argument": "500万股"},
      {"role": "占公司总股本比例", "argument": "0.35%"},
      {"role": "回购完成时间", "argument": "2023年3月"},
      {"role": "披露时间", "argument": "2023年3月15日"}
    ]
  }
]
"""
        },
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
2024年8月5日，华建科技公告称，公司于2024年8月1日中标智慧园区升级项目，中标金额1.2亿元，招标方为海城城投。

【抽取要求】
请直接输出标准 JSON 格式的事件列表，不要添加任何额外说明。""",
            "assistant": """[
  {
    "event_type": "中标",
    "trigger": "中标",
    "arguments": [
      {"role": "中标公司", "argument": "华建科技"},
      {"role": "中标标的", "argument": "智慧园区升级项目"},
      {"role": "中标金额", "argument": "1.2亿元"},
      {"role": "招标方", "argument": "海城城投"},
      {"role": "中标日期", "argument": "2024年8月1日"},
      {"role": "披露日期", "argument": "2024年8月5日"}
    ]
  }
]"""
        },
    ]
    # 兼容旧代码引用
    ONE_SHOT_EXAMPLE = FEW_SHOT_EXAMPLES[0]

    @classmethod
    def _normalize_schema(cls, schema: Optional[Dict]) -> Dict[str, List[str]]:
        """规范化 schema 输入，统一为 {event_type: [role,...]}。"""
        if not isinstance(schema, dict):
            return {}

        normalized: Dict[str, List[str]] = {}
        for event_type, raw_roles in schema.items():
            if not event_type:
                continue
            roles: List[str] = []
            if isinstance(raw_roles, list):
                for item in raw_roles:
                    if isinstance(item, str) and item and item not in roles:
                        roles.append(item)
                    elif isinstance(item, dict):
                        role = item.get("role")
                        if isinstance(role, str) and role and role not in roles:
                            roles.append(role)
            normalized[str(event_type)] = roles
        return normalized

    @classmethod
    def build_schema_constraints(cls, schema: Optional[Dict]) -> str:
        """
        构建 schema 约束块。
        当提供 schema 时，显式给出 event_type -> roles，减少 role 同义词漂移。
        """
        normalized = cls._normalize_schema(schema)
        if not normalized:
            return ""

        lines = [
            "## 事件类型与合法论元角色（严格遵守）",
            "以下角色名必须与 schema 完全一致（逐字匹配）：",
        ]
        for event_type in sorted(normalized.keys()):
            roles = normalized[event_type]
            if roles:
                lines.append(f"- {event_type}: {', '.join(roles)}")
            else:
                lines.append(f"- {event_type}: (无角色定义)")
        lines.append("若 role 无法确定，请省略该 role，不要使用近义字段名。")
        lines.append("请使用 schema 中的标准角色名，不要输出别名角色名。")
        lines.append("若同一 role 对应多个论元值，请拆成多个独立的 arguments 项。")
        return "\n".join(lines)

    @classmethod
    def build_system_prompt(cls, schema: Optional[Dict] = None) -> str:
        """获取系统提示词，可选附加 schema 约束。"""
        base = cls.SYSTEM_PROMPT
        schema_block = cls.build_schema_constraints(schema)
        if schema_block:
            return f"{base}\n\n{schema_block}"
        return base

    @classmethod
    def build_user_prompt(cls, text: str, max_length: int = 3500) -> str:
        """
        构建用户输入提示词
        
        Args:
            text: 原始文档文本
            max_length: 最大文本长度（防止超出上下文窗口）
        
        Returns:
            格式化的用户提示词
        """
        # 截断过长文本
        if len(text) > max_length:
            text = text[:max_length] + "...[文本已截断]"
        
        return cls.USER_TEMPLATE.format(text=text)

    @classmethod
    def build_cot_response(cls, event_list: List[Dict], schema: Optional[Dict] = None) -> str:
        """
        构建训练数据的标准响应。

        Args:
            event_list: 标注的事件列表
            schema: 保留兼容入参；严格 JSON 契约下不再生成中间推理文本

        Returns:
            标准 JSON 数组字符串
        """
        _ = schema
        return json.dumps(event_list, ensure_ascii=False, indent=2)

    @classmethod
    def build_incorrect_cot_response(
        cls, 
        neg_json: str, 
        strategy: str,
        original_types: Optional[List[str]] = None
    ) -> str:
        """
        构建负样本响应（用于训练数据的 rejected）

        设计原则（基于代码审查反馈）：
        - 不使用显式错误标记，避免让模型轻易区分 chosen/rejected
        - 仍保持严格 JSON 契约，避免训练/评测输出格式漂移
        
        Args:
            neg_json: 负样本 JSON 字符串
            strategy: 负采样策略 (EASY/MEDIUM/HARD)
            original_types: 原始事件类型列表
        
        Returns:
            负样本 JSON 字符串
        """
        _ = strategy
        _ = original_types
        if isinstance(neg_json, str):
            return neg_json
        return json.dumps(neg_json, ensure_ascii=False, indent=2)

    @staticmethod
    def template_leakage_score(chosen_text: str, rejected_text: str) -> float:
        """
        简易模板泄漏分数（0~1，越高表示词面差异越大）。

        使用 token 集合对称差比例，作为可测试的弱代理指标。
        """
        token_re = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9_]+")
        chosen_tokens = set(token_re.findall(str(chosen_text or "")))
        rejected_tokens = set(token_re.findall(str(rejected_text or "")))
        union = chosen_tokens | rejected_tokens
        if not union:
            return 0.0
        sym_diff = chosen_tokens ^ rejected_tokens
        return len(sym_diff) / len(union)

    @classmethod
    def get_messages_for_inference(
        cls,
        text: str,
        schema: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        """
        构建用于推理的消息列表（适配 Qwen 的 chat_template）
        
        Args:
            text: 待抽取的文本
        
        Returns:
            [{"role": "system", ...}, {"role": "user", ...}] 格式的消息列表
        """
        payload = cls.build_inference_payload(
            text=text,
            schema=schema,
            use_oneshot=False,
        )
        return payload["messages"]

    @classmethod
    def select_fewshot_examples(cls, num_examples: int = 3) -> List[Dict[str, str]]:
        """
        选择 few-shot 示例。
        目前使用稳定的前 n 条示例，确保实验可复现。
        """
        n = max(1, min(num_examples, len(cls.FEW_SHOT_EXAMPLES)))
        return cls.FEW_SHOT_EXAMPLES[:n]

    @classmethod
    def get_messages_with_oneshot(
        cls,
        text: str,
        num_examples: int = 3,
        schema: Optional[Dict] = None,
    ) -> List[Dict[str, str]]:
        """
        构建包含 Few-Shot 示例的消息列表
        
        Args:
            text: 待抽取的文本
            num_examples: 使用的示例数量（默认 3）
        
        Returns:
            包含 system + 若干示例 + 用户输入的消息列表
        """
        payload = cls.build_inference_payload(
            text=text,
            schema=schema,
            use_oneshot=True,
            num_examples=num_examples,
        )
        return payload["messages"]

    @classmethod
    def build_inference_payload(
        cls,
        text: str,
        schema: Optional[Dict] = None,
        *,
        prompt_variant: Optional[str] = None,
        use_oneshot: Optional[bool] = None,
        num_examples: int = 3,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        构建统一的推理 payload。

        该 payload 是 train/local eval/API eval 的共享入口，统一提供：
        - 标准化 messages
        - 可选的 formatted_text（当提供 tokenizer 时）
        - prompt_variant 元信息
        """
        prompt_settings = resolve_prompt_settings(
            prompt_variant=prompt_variant,
            use_oneshot=use_oneshot,
            fewshot_num_examples=num_examples,
            default_prompt_variant="zeroshot",
            default_num_examples=num_examples,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": cls.build_system_prompt(schema=schema)},
        ]
        fewshot_count = 0
        if prompt_settings["use_oneshot"]:
            selected_examples = cls.select_fewshot_examples(
                num_examples=prompt_settings["fewshot_num_examples"]
            )
            fewshot_count = len(selected_examples)
            for ex in selected_examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": cls.build_user_prompt(text)})

        formatted_text: Optional[str] = None
        if tokenizer is not None:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return {
            "messages": messages,
            "formatted_text": formatted_text,
            "prompt_variant": prompt_settings["prompt_variant"],
            "fewshot_count": fewshot_count,
            "schema_enabled": bool(schema),
        }


# ========================================
# 便捷函数导出
# ========================================
def get_system_prompt() -> str:
    """获取系统提示词（便捷函数）"""
    return ChinesePromptBuilder.build_system_prompt()


def format_user_input(text: str) -> str:
    """格式化用户输入（便捷函数）"""
    return ChinesePromptBuilder.build_user_prompt(text)


def build_training_response(event_list: List[Dict]) -> str:
    """构建训练用的标准响应（便捷函数）"""
    return ChinesePromptBuilder.build_cot_response(event_list)


def build_inference_prompt(
    text: str,
    tokenizer,
    use_oneshot: Optional[bool] = None,
    schema: Optional[Dict] = None,
    num_examples: int = 3,
    prompt_variant: Optional[str] = None,
) -> str:
    """
    构建用于推理的完整 prompt（便捷函数）

    统一训练和评估的 prompt 构建逻辑，确保格式一致性。

    Args:
        text: 待抽取的文本
        tokenizer: HuggingFace tokenizer，用于 apply_chat_template
        use_oneshot: 是否使用 One-Shot 示例

    Returns:
        格式化后的完整 prompt 字符串
    """
    payload = ChinesePromptBuilder.build_inference_payload(
        text=text,
        schema=schema,
        prompt_variant=prompt_variant,
        use_oneshot=use_oneshot,
        num_examples=num_examples,
        tokenizer=tokenizer,
    )
    return payload["formatted_text"]


def build_inference_prompt_payload(
    text: str,
    *,
    tokenizer: Optional[Any] = None,
    prompt_variant: Optional[str] = None,
    use_oneshot: Optional[bool] = None,
    schema: Optional[Dict] = None,
    num_examples: int = 3,
) -> Dict[str, Any]:
    """便捷函数：导出统一的推理 payload。"""
    return ChinesePromptBuilder.build_inference_payload(
        text=text,
        schema=schema,
        prompt_variant=prompt_variant,
        use_oneshot=use_oneshot,
        num_examples=num_examples,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    # 测试代码
    test_text = "2024年1月，阿里巴巴集团宣布完成100亿美元的股份回购计划。"
    
    print("=" * 50)
    print("【System Prompt】")
    print(get_system_prompt())
    print("\n" + "=" * 50)
    print("【User Prompt】")
    print(format_user_input(test_text))
    print("\n" + "=" * 50)
    print("【Sample Response】")
    test_events = [
        {
            "event_type": "股份回购",
            "trigger": "回购",
            "arguments": [
                {"role": "回购方", "argument": "阿里巴巴集团"},
                {"role": "交易金额", "argument": "100亿美元"},
                {"role": "回购完成时间", "argument": "2024年1月"}
            ]
        }
    ]
    print(build_training_response(test_events))

