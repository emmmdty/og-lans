# src/data/prompt_builder.py
"""
中文化提示词构建模块 (Chinese-First Prompt Engineering)
符合 Qwen3/DeepSeek 中文能力优化的 CoT 引导策略

解决问题:
1. 语言失配 - 英文 Prompt 配合中文数据集导致输出混乱
2. JSON 约束失效 - 模型生成过于冗长破坏 JSON 结构
3. CoT 规范化 - Wait-then-Answer 模式标准化
"""

from typing import List, Dict, Optional
import json


class ChinesePromptBuilder:
    """
    中文化提示词构建器 - 适配 DuEE-Fin 金融事件抽取任务
    核心设计:
    - System Prompt: 强约束 JSON 格式 + 反幻觉规则
    - CoT Template: 中文三步推理法
    - Few-Shot Examples: 多类型/多场景示例，提升泛化稳定性
    """
    
    # ========================================
    # 系统提示词 (System Prompt) - 强化 JSON 约束
    # ========================================
    SYSTEM_PROMPT = """你是一位专业的中文金融事件抽取专家。你的任务是从金融新闻/公告中识别和抽取结构化事件信息。

## 输出格式要求【必须严格遵守】
1. 先在 <thought> 标签内用中文进行三步推理分析
2. 推理结束后，紧跟一个 ```json 代码块输出最终结果
3. JSON 必须是一个事件列表 (数组)，即使只有一个事件也要用 []
4. 每个事件包含: event_type(事件类型), trigger(触发词), arguments(论元列表)
5. 每个论元包含: role(角色), argument(论元值)
6. 仅允许在 JSON 之前输出一个 <thought> 分析块；除 <thought> 外，禁止添加任何解释性文字
7. 如果没有检测到任何事件，输出: ```json\n[]\n```

## 学术评测约束（降低幻觉与格式偏差）
1. 论元值必须直接来自原文连续片段，不得改写、归一化或补全
2. role 必须属于该 event_type 的合法角色；不确定时宁可省略该论元
3. 同一文本可包含多个事件；不同事件不得错误合并
4. 同一触发词对应多个主体（如多家公司被约谈）时，按主体拆分为多个事件对象
5. 禁止输出 schema 外事件类型或虚构字段

## 支持的事件类型
质押, 股份回购, 企业融资, 公司上市, 企业收购, 中标, 高管变动, 解除质押, 股东减持, 股东增持, 企业破产, 亏损, 被约谈

## 标准输出示例
<thought>
第一步：Schema 分析（模式识别）
- 文本中出现"融资"、"投资方"等关键词，可能触发"企业融资"事件
- 检测到"上市"触发词，可能存在"公司上市"事件

第二步：实体扫描（论元抽取）
- 企业融资事件：被投资方="某科技公司", 融资金额="1亿美元", 融资轮次="B"
- 公司上市事件：上市公司="某科技公司", 证券代码="ABC"

第三步：约束校验（自我修正）
- 校验论元边界是否准确
- 确认多值论元（如多个投资方）已全部识别
- 所有约束检查通过
</thought>

```json
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
```"""

    # ========================================
    # 用户输入模板
    # ========================================
    USER_TEMPLATE = """请从以下金融文本中抽取所有事件信息。

【文本内容】
{text}

【抽取要求】
请按照"三步推理法"进行分析：
1. Schema 分析：识别可能触发的事件类型
2. 实体扫描：提取各事件的论元角色和值
3. 约束校验：验证抽取结果的准确性和完整性

完成推理后，输出标准 JSON 格式的事件列表。"""

    # ========================================
    # Few-Shot 示例（默认多样本）
    # ========================================
    FEW_SHOT_EXAMPLES = [
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
2023年3月15日，新能源龙头企业阳光电源发布公告称，公司拟以自有资金回购公司股份，回购金额不低于5亿元、不超过10亿元，回购价格不超过150元/股，回购股份将用于员工持股计划。本次回购已于2023年3月实施完毕，累计回购股份数量为500万股，占公司总股本的0.35%。

【抽取要求】
请按照"三步推理法"进行分析：
1. Schema 分析：识别可能触发的事件类型
2. 实体扫描：提取各事件的论元角色和值
3. 约束校验：验证抽取结果的准确性和完整性

完成推理后，输出标准 JSON 格式的事件列表。""",
            "assistant": """<thought>
第一步：Schema 分析（模式识别）
- 检测到"回购"触发词，触发事件类型：股份回购
- 文本描述的是一个已完成的股份回购事件

第二步：实体扫描（论元抽取）
- 回购方：阳光电源（主语，执行回购动作的公司）
- 交易金额：不低于5亿元、不超过10亿元（文本提及"不低于5亿元、不超过10亿元"，取区间描述）
- 每股交易价格：不超过150元/股
- 回购股份数量：500万股（明确数字）
- 占公司总股本比例：0.35%
- 回购完成时间：2023年3月
- 披露时间：2023年3月15日

第三步：约束校验（自我修正）
- 校验论元边界："阳光电源"边界准确，未包含多余字符
- 确认数值类论元格式规范
- 所有角色均属于"股份回购"事件的标准 Schema
- 约束检查通过
</thought>

```json
[
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
```"""
        },
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
2024年4月1日，星河科技公告称公司于2024年3月完成B轮融资，融资金额2亿元，领投方为红杉资本，投资方包括腾讯。公告同时披露，公司计划于2025年启动上市筹备。

【抽取要求】
请按照"三步推理法"进行分析：
1. Schema 分析：识别可能触发的事件类型
2. 实体扫描：提取各事件的论元角色和值
3. 约束校验：验证抽取结果的准确性和完整性

完成推理后，输出标准 JSON 格式的事件列表。""",
            "assistant": """<thought>
第一步：Schema 分析（模式识别）
- 检测到"融资"触发词，触发事件类型：企业融资
- 检测到"上市筹备"语义，触发事件类型：公司上市（环节=上市筹备）

第二步：实体扫描（论元抽取）
- 企业融资事件：被投资方=星河科技，融资金额=2亿元，融资轮次=B轮，领投方=红杉资本，投资方=腾讯，事件时间=2024年3月，披露时间=2024年4月1日
- 公司上市事件：上市公司=星河科技，环节=上市筹备，事件时间=2025年，披露时间=2024年4月1日

第三步：约束校验（自我修正）
- 论元均为原文可定位片段
- role 与事件类型匹配，未使用 schema 外字段
- 约束检查通过
</thought>

```json
[
  {
    "event_type": "企业融资",
    "trigger": "融资",
    "arguments": [
      {"role": "被投资方", "argument": "星河科技"},
      {"role": "融资金额", "argument": "2亿元"},
      {"role": "融资轮次", "argument": "B轮"},
      {"role": "领投方", "argument": "红杉资本"},
      {"role": "投资方", "argument": "腾讯"},
      {"role": "事件时间", "argument": "2024年3月"},
      {"role": "披露时间", "argument": "2024年4月1日"}
    ]
  },
  {
    "event_type": "公司上市",
    "trigger": "上市筹备",
    "arguments": [
      {"role": "上市公司", "argument": "星河科技"},
      {"role": "环节", "argument": "上市筹备"},
      {"role": "事件时间", "argument": "2025年"},
      {"role": "披露时间", "argument": "2024年4月1日"}
    ]
  }
]
```"""
        },
        {
            "user": """请从以下金融文本中抽取所有事件信息。

【文本内容】
公司今日召开年度战略发布会，介绍了新产品路线图与组织升级计划。公告未涉及融资、上市、收购、股东增减持、质押、回购、亏损或被约谈等事项。

【抽取要求】
请按照"三步推理法"进行分析：
1. Schema 分析：识别可能触发的事件类型
2. 实体扫描：提取各事件的论元角色和值
3. 约束校验：验证抽取结果的准确性和完整性

完成推理后，输出标准 JSON 格式的事件列表。""",
            "assistant": """<thought>
第一步：Schema 分析（模式识别）
- 文本不包含 DuEE-Fin 定义的事件触发模式

第二步：实体扫描（论元抽取）
- 未发现可抽取事件及其论元

第三步：约束校验（自我修正）
- 按 schema 约束，应返回空列表而非猜测事件
- 约束检查通过
</thought>

```json
[]
```"""
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
        构建中文化的 CoT 响应（用于训练数据的 chosen）
        
        Args:
            event_list: 标注的事件列表
            schema: 可选的 Schema 信息，用于增强推理描述
        
        Returns:
            包含 <thought> 推理过程和 JSON 结果的完整响应
        """
        # 提取检测到的事件类型
        detected_types = list(set([e.get('event_type', '未知') for e in event_list]))
        
        # 构建思维链
        cot_parts = ["<thought>"]
        
        # 第一步：Schema 分析
        cot_parts.append("第一步：Schema 分析（模式识别）")
        if not detected_types or not event_list:
            cot_parts.append("- 未检测到明确的金融事件触发词")
        else:
            triggers = [e.get('trigger', '') for e in event_list if e.get('trigger')]
            if triggers:
                cot_parts.append(f"- 检测到触发词：{', '.join(triggers)}")
            cot_parts.append(f"- 识别的事件类型：{', '.join(detected_types)}")
        
        # 第二步：实体扫描
        cot_parts.append("\n第二步：实体扫描（论元抽取）")
        if event_list:
            for event in event_list:
                etype = event.get('event_type', '未知')
                args = event.get('arguments', [])
                cot_parts.append(f"- {etype}事件：")
                for arg in args:
                    role = arg.get('role', '')
                    value = arg.get('argument', '')
                    if role and value:
                        # 截断过长的论元值
                        display_value = value if len(value) <= 30 else value[:27] + "..."
                        cot_parts.append(f"  · {role} = \"{display_value}\"")
        else:
            cot_parts.append("- 无需抽取论元")
        
        # 第三步：约束校验
        cot_parts.append("\n第三步：约束校验（自我修正）")
        cot_parts.append("- 校验论元边界准确性")
        cot_parts.append("- 确认多值论元已完整识别")
        cot_parts.append("- 所有约束检查通过")
        
        cot_parts.append("</thought>")
        
        # 构建 JSON 输出
        json_output = json.dumps(event_list, ensure_ascii=False, indent=2)
        
        return "\n".join(cot_parts) + f"\n\n```json\n{json_output}\n```"

    @classmethod
    def build_incorrect_cot_response(
        cls, 
        neg_json: str, 
        strategy: str,
        original_types: Optional[List[str]] = None
    ) -> str:
        """
        构建负样本的 CoT 响应（用于训练数据的 rejected）
        
        设计原则（基于代码审查反馈）：
        - 不使用显式错误标记（如 [推理偏差: ...]），避免让模型轻易区分 chosen/rejected
        - 生成"看似合理但结论错误"的推理链，增加 DPO 的学习难度
        - 根据不同策略模拟不同类型的推理偏差
        - 【关键改进】增加模板多样性，防止模型学习简单文本特征
        
        Args:
            neg_json: 负样本 JSON 字符串
            strategy: 负采样策略 (EASY/MEDIUM/HARD)
            original_types: 原始事件类型列表
        
        Returns:
            模拟错误推理过程的响应
        """
        import random as _random
        
        # 尝试解析负样本 JSON 获取错误的事件类型
        try:
            neg_data = json.loads(neg_json) if isinstance(neg_json, str) else neg_json
            if isinstance(neg_data, list) and neg_data:
                neg_event = neg_data[0]
            else:
                neg_event = neg_data
            wrong_type = neg_event.get('event_type', '未知事件')
            wrong_args = neg_event.get('arguments', [])
        except (json.JSONDecodeError, TypeError, AttributeError):
            wrong_type = '未知事件'
            wrong_args = []
        
        cot_parts = ["<thought>"]
        
        # 【关键改进】多样化的模板库
        # 第一步模板变体
        step1_templates = {
            "EASY": [
                [f"- 检测到与{wrong_type}相关的语义特征", f"- 触发事件类型判定：{wrong_type}"],
                [f"- 文本语义指向{wrong_type}类事件", f"- 事件类型确定为：{wrong_type}"],
                [f"- 分析文本结构，识别出{wrong_type}模式", f"- 事件类型：{wrong_type}"],
                [f"- 关键词匹配显示{wrong_type}特征", f"- 判定类型为{wrong_type}"],
            ],
            "MEDIUM": [
                [f"- 文本涉及{', '.join(original_types[:2]) if original_types else '金融'}相关语义", f"- 综合判断触发类型：{wrong_type}"],
                [f"- 多个事件特征共现", f"- 主要事件识别为：{wrong_type}"],
                [f"- 语义分析显示复合事件特征", f"- 核心事件类型：{wrong_type}"],
            ],
            "HARD": [
                [f"- 识别到{wrong_type}事件的典型模式", "- 实体边界需要精确定位"],
                [f"- {wrong_type}事件特征明确", "- 论元精确抽取中"],
                [f"- 检测到标准{wrong_type}模式", "- 进行细粒度论元识别"],
            ]
        }
        
        # 第三步模板变体
        step3_templates = {
            "EASY": [
                ["- 事件类型与文本语境基本吻合", "- 约束检查完成"],
                ["- 类型判定结果校验通过", "- 完成约束验证"],
                ["- 事件类型符合文本语义", "- 校验结束"],
            ],
            "MEDIUM": [
                ["- 论元角色分配符合 Schema 定义", "- 约束检查完成"],
                ["- 角色分配验证通过", "- 完成校验"],
                ["- 论元与角色映射正确", "- 校验结束"],
            ],
            "HARD": [
                ["- 实体边界校验通过", "- 约束检查完成"],
                ["- 数值精度验证完成", "- 校验通过"],
                ["- 论元边界确认无误", "- 完成校验"],
            ]
        }
        
        # 随机选择模板
        step1_choice = _random.choice(step1_templates.get(strategy, step1_templates["EASY"]))
        step3_choice = _random.choice(step3_templates.get(strategy, step3_templates["EASY"]))
        
        # 第一步：Schema 分析
        cot_parts.append("第一步：Schema 分析（模式识别）")
        cot_parts.extend(step1_choice)
        
        # 第二步：实体扫描 - 展示错误的论元抽取
        cot_parts.append("\n第二步：实体扫描（论元抽取）")
        cot_parts.append(f"- {wrong_type}事件：")
        
        if wrong_args:
            for arg in wrong_args[:3]:  # 最多展示3个
                role = arg.get('role', '')
                value = arg.get('argument', '')
                if role and value:
                    display_value = value if len(value) <= 25 else value[:22] + "..."
                    cot_parts.append(f"  · {role} = \"{display_value}\"")
        else:
            cot_parts.append("  · 论元信息待确认")
        
        # 第三步：约束校验
        cot_parts.append("\n第三步：约束校验（自我修正）")
        cot_parts.extend(step3_choice)
        
        cot_parts.append("</thought>")
        
        return "\n".join(cot_parts) + f"\n\n```json\n{neg_json}\n```"

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
        return [
            {"role": "system", "content": cls.build_system_prompt(schema=schema)},
            {"role": "user", "content": cls.build_user_prompt(text)}
        ]

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
        selected_examples = cls.select_fewshot_examples(num_examples=num_examples)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": cls.build_system_prompt(schema=schema)},
        ]
        for ex in selected_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": cls.build_user_prompt(text)})
        return messages


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
    use_oneshot: bool = False,
    schema: Optional[Dict] = None,
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
    if use_oneshot:
        messages = ChinesePromptBuilder.get_messages_with_oneshot(text, schema=schema)
    else:
        messages = ChinesePromptBuilder.get_messages_for_inference(text, schema=schema)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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

