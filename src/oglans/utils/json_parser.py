# src/utils/json_parser.py
"""
鲁棒 JSON 解析器 (Robust JSON Parser)
专门处理 LLM 生成的"脏" JSON 数据

解决问题:
1. Markdown 代码块污染 (```json ... ```)
2. 缺少闭合括号 (Unclosed brackets)
3. 中文标点符号 (全角逗号、冒号等)
4. 尾部逗号 (Trailing commas)
5. 单引号替代双引号
6. 非法控制字符
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

try:
    import dirtyjson as _DIRTYJSON_MODULE
except ImportError:
    _DIRTYJSON_MODULE = None

logger = logging.getLogger("OGLANS")


class RobustJSONParser:
    """
    鲁棒 JSON 解析器
    
    采用多级降级策略：
    1. 标准解析 -> 2. 格式修复 -> 3. 括号补全 -> 4. dirtyjson 兜底
    """
    
    # 中文标点到英文标点的映射
    PUNCTUATION_MAP = {
        '，': ',',   # 全角逗号 U+FF0C
        '：': ':',   # 全角冒号 U+FF1A
        '；': ';',   # 全角分号 U+FF1B
        '\u201c': '"',   # 中文左双引号 U+201C
        '\u201d': '"',   # 中文右双引号 U+201D
        '\u2018': "'",   # 中文左单引号 U+2018
        '\u2019': "'",   # 中文右单引号 U+2019
        '【': '[',   # 中文左方括号 U+3010
        '】': ']',   # 中文右方括号 U+3011
        '（': '(',   # 中文左圆括号 U+FF08
        '）': ')',   # 中文右圆括号 U+FF09
        '｛': '{',   # 全角左花括号 U+FF5B
        '｝': '}',   # 全角右花括号 U+FF5D
    }
    
    def __init__(self, enable_dirtyjson: bool = True, max_repair_attempts: int = 5):
        """
        初始化解析器
        
        Args:
            enable_dirtyjson: 是否启用 dirtyjson 库作为最终兜底
            max_repair_attempts: 最大修复尝试次数
        """
        self.enable_dirtyjson = enable_dirtyjson
        self.max_repair_attempts = max_repair_attempts
        
        # 尝试导入 dirtyjson（可选依赖）
        self._dirtyjson = None
        if enable_dirtyjson:
            if _DIRTYJSON_MODULE is not None:
                self._dirtyjson = _DIRTYJSON_MODULE
            else:
                logger.warning("dirtyjson 未安装，将禁用该兜底策略。安装命令: pip install dirtyjson")

    def extract_json_from_text(self, text: str) -> Tuple[Optional[str], str]:
        """
        从文本中提取 JSON 部分
        
        Args:
            text: 包含 JSON 的完整文本（可能包含 CoT、Markdown 等）
        
        Returns:
            (提取的 JSON 字符串, 提取方式描述)
        """
        if not text or not text.strip():
            return None, "empty_input"
        
        # 策略 1: 匹配 ```json ... ``` 代码块
        pattern_markdown = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern_markdown, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # 取最后一个匹配（通常 CoT 后的 JSON 才是最终答案）
            return matches[-1].strip(), "markdown_block"
        
        # 策略 2: 匹配 [ ... ] 数组
        pattern_array = r'\[\s*\{.*\}\s*\]'
        match = re.search(pattern_array, text, re.DOTALL)
        if match:
            return match.group(0).strip(), "array_pattern"
        
        # 策略 3: 匹配 { ... } 对象（可能是单个事件）
        pattern_object = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(pattern_object, text, re.DOTALL)
        if match:
            return match.group(0).strip(), "object_pattern"
        
        # 策略 4: 匹配空数组 []
        if '[]' in text:
            return '[]', "empty_array"
        
        # 策略 5: 直接返回原文本（可能本身就是 JSON）
        text_stripped = text.strip()
        if text_stripped.startswith('[') or text_stripped.startswith('{'):
            return text_stripped, "raw_json"
        
        return None, "no_json_found"

    def normalize_punctuation(self, json_str: str) -> str:
        """
        将中文标点转换为英文标点
        
        Args:
            json_str: 待处理的 JSON 字符串
        
        Returns:
            标点规范化后的字符串
        """
        for cn, en in self.PUNCTUATION_MAP.items():
            json_str = json_str.replace(cn, en)
        return json_str

    def fix_trailing_comma(self, json_str: str) -> str:
        """
        修复尾部逗号问题
        
        例如: {"a": 1,} -> {"a": 1}
        """
        # 移除 ] 或 } 前的逗号（含空白）
        json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
        return json_str

    def fix_single_quotes(self, json_str: str) -> str:
        """
        将单引号替换为双引号（简单场景）
        
        注意: 这可能在字符串内容包含单引号时出错，仅作为兜底使用
        """
        # 匹配键名中的单引号
        # 'key': -> "key":
        json_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
        return json_str

    def fix_unclosed_brackets(self, json_str: str) -> str:
        """
        修复未闭合的括号
        
        Args:
            json_str: 可能缺少闭合括号的 JSON 字符串
        
        Returns:
            补全括号后的字符串
        """
        # 统计括号数量
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 补全缺失的闭合括号
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str

    def remove_control_characters(self, json_str: str) -> str:
        """
        移除非法控制字符
        """
        # 移除除换行、回车、制表符外的控制字符
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)

    def parse(self, text: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        主解析入口 - 多级降级策略
        
        Args:
            text: 包含 JSON 的文本
        
        Returns:
            (解析结果, 诊断信息字典)
        """
        diagnostics = {
            "success": False,
            "extraction_method": None,
            "repair_steps": [],
            "error": None
        }
        
        # 步骤 1: 提取 JSON 部分
        json_str, extraction_method = self.extract_json_from_text(text)
        diagnostics["extraction_method"] = extraction_method
        
        if json_str is None:
            diagnostics["error"] = "JSON 提取失败"
            return None, diagnostics
        
        # 步骤 2: 尝试直接解析
        try:
            result = json.loads(json_str)
            diagnostics["success"] = True
            return result, diagnostics
        except json.JSONDecodeError:
            pass  # 继续修复流程
        
        # 步骤 3: 应用修复策略
        repair_sequence = [
            ("normalize_punctuation", self.normalize_punctuation),
            ("fix_trailing_comma", self.fix_trailing_comma),
            ("remove_control_characters", self.remove_control_characters),
            ("fix_unclosed_brackets", self.fix_unclosed_brackets),
            ("fix_single_quotes", self.fix_single_quotes),
        ]
        
        current = json_str
        for repair_name, repair_func in repair_sequence:
            current = repair_func(current)
            diagnostics["repair_steps"].append(repair_name)
            
            try:
                result = json.loads(current)
                diagnostics["success"] = True
                return result, diagnostics
            except json.JSONDecodeError:
                continue  # 尝试下一个修复
        
        # 步骤 4: 使用 dirtyjson 兜底
        if self._dirtyjson is not None:
            diagnostics["repair_steps"].append("dirtyjson_fallback")
            try:
                result = self._dirtyjson.loads(current)
                # dirtyjson 返回的是 OrderedDict，转换为普通 dict/list
                result = json.loads(json.dumps(result))
                diagnostics["success"] = True
                return result, diagnostics
            except Exception as e:
                diagnostics["error"] = f"dirtyjson 也无法解析: {str(e)}"
        else:
            diagnostics["error"] = f"所有修复策略均失败"
        
        return None, diagnostics

    def parse_events(self, text: str) -> List[Dict]:
        """
        便捷方法: 解析事件列表
        
        Args:
            text: LLM 输出文本
        
        Returns:
            事件列表（解析失败返回空列表）
        """
        result, diagnostics = self.parse(text)
        
        if not diagnostics["success"]:
            logger.debug(f"JSON 解析失败: {diagnostics.get('error', 'Unknown')}")
            return []
        
        # 确保结果是列表
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # 单个事件，包装成列表
            return [result]
        else:
            return []


# ========================================
# 全局单例 & 便捷函数
# ========================================
_default_parser: Optional[RobustJSONParser] = None


def get_parser() -> RobustJSONParser:
    """获取全局解析器实例"""
    global _default_parser
    if _default_parser is None:
        _default_parser = RobustJSONParser()
    return _default_parser


def parse_llm_output(text: str) -> Optional[List[Dict]]:
    """
    解析 LLM 输出（便捷函数）
    
    Args:
        text: LLM 生成的文本
    
    Returns:
        事件列表，解析失败返回 None
    """
    parser = get_parser()
    result = parser.parse_events(text)
    return result if result else None


def parse_with_diagnostics(text: str) -> Tuple[Optional[Any], Dict]:
    """
    解析并返回诊断信息（便捷函数）
    
    Args:
        text: LLM 生成的文本
    
    Returns:
        (解析结果, 诊断信息)
    """
    return get_parser().parse(text)


def validate_event_structure(events: List[Dict]) -> Tuple[bool, List[str]]:
    """
    验证事件结构是否符合 DuEE-Fin 格式要求
    
    Args:
        events: 事件列表
    
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    if not isinstance(events, list):
        return False, ["根元素必须是列表"]
    
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"事件 {i}: 必须是字典类型")
            continue
        
        # 检查必需字段
        if "event_type" not in event:
            errors.append(f"事件 {i}: 缺少 event_type 字段")
        
        args = event.get("arguments", [])
        if not isinstance(args, list):
            errors.append(f"事件 {i}: arguments 必须是列表")
            continue
        
        for j, arg in enumerate(args):
            if not isinstance(arg, dict):
                errors.append(f"事件 {i} 论元 {j}: 必须是字典类型")
                continue
            if "role" not in arg:
                errors.append(f"事件 {i} 论元 {j}: 缺少 role 字段")
            if "argument" not in arg:
                errors.append(f"事件 {i} 论元 {j}: 缺少 argument 字段")
    
    return len(errors) == 0, errors


# ========================================
# 自我修正策略 (Self-Correction via LLM)
# ========================================
class LLMSelfCorrector:
    """
    LLM 自我修正器
    
    当 JSON 解析失败时，构造修正 Prompt 让 LLM 重新生成
    """
    
    CORRECTION_PROMPT_TEMPLATE = """你之前的输出存在 JSON 格式错误，请修正。

【错误信息】
{error_message}

【原始输出】
{original_output}

【修正要求】
1. 只输出修正后的 JSON，不要包含任何解释
2. 确保 JSON 格式正确（所有括号闭合、使用英文标点）
3. 输出格式: ```json\n[...事件列表...]\n```

请输出修正后的 JSON:"""

    @classmethod
    def build_correction_prompt(
        cls, 
        original_output: str, 
        error_message: str
    ) -> str:
        """
        构建修正提示词
        
        Args:
            original_output: LLM 原始输出
            error_message: 解析错误信息
        
        Returns:
            用于自我修正的 Prompt
        """
        # 截断过长的原始输出
        if len(original_output) > 1000:
            original_output = original_output[:1000] + "...[已截断]"
        
        return cls.CORRECTION_PROMPT_TEMPLATE.format(
            error_message=error_message,
            original_output=original_output
        )

    @classmethod
    def should_attempt_correction(cls, diagnostics: Dict) -> bool:
        """
        判断是否应该尝试自我修正
        
        Args:
            diagnostics: 解析诊断信息
        
        Returns:
            True 表示应该尝试修正
        """
        # 如果完全没有提取到 JSON，修正可能无效
        if diagnostics.get("extraction_method") == "no_json_found":
            return False
        
        # 如果已经尝试了多种修复但仍失败，值得尝试 LLM 修正
        return not diagnostics.get("success", False)


if __name__ == "__main__":
    # 测试代码
    test_cases = [
        # 正常 JSON
        '```json\n[{"event_type": "股份回购", "arguments": []}]\n```',
        
        # 缺少闭合括号
        '```json\n[{"event_type": "企业融资", "arguments": [{"role": "融资金额", "argument": "1亿"}',
        
        # 中文标点
        '```json\n[{"event_type"："质押"，"arguments"：[]}]\n```',
        
        # 尾部逗号
        '```json\n[{"event_type": "中标", "arguments": [],}]\n```',
        
        # 包含 CoT
        '<thought>\n分析中...\n</thought>\n\n```json\n[{"event_type": "企业收购", "trigger": "收购", "arguments": []}]\n```',
    ]
    
    parser = RobustJSONParser()
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"测试用例 {i+1}:")
        print(f"输入: {case[:50]}...")
        
        result, diag = parser.parse(case)
        
        print(f"成功: {diag['success']}")
        print(f"提取方式: {diag['extraction_method']}")
        print(f"修复步骤: {diag['repair_steps']}")
        print(f"结果: {result}")
