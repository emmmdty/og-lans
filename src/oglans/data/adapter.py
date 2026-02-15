"""
DuEE-Fin 数据集适配器
重构版本: 集成中文化 Prompt Builder，解决语言失配问题
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .prompt_builder import ChinesePromptBuilder

logger = logging.getLogger("OGLANS")


@dataclass
class DUEE_FIN_EESample:
    """DuEE-Fin 事件抽取样本数据类"""
    id: str
    text: str
    prompt: str
    chosen: str  # Ground Truth (CoT + JSON)
    rejected: str = ""  # 将由 DS-CNS 动态填充
    event_types: List[str] = field(default_factory=list)  # 新增：事件类型列表，便于 DS-CNS 使用
    events: List[Dict] = field(default_factory=list)  # 原始事件列表 (Ground Truth)


class DuEEFinAdapter:
    """
    百度 DuEE-Fin 数据集适配器
    
    重构亮点：
    1. 完全中文化的 CoT 模板
    2. 强化 JSON 格式约束
    3. 支持 Few-Shot 示例注入
    4. 保留事件类型信息用于 DS-CNS 负采样
    """
    
    def __init__(
        self, 
        data_path: str, 
        schema_path: str,
        max_text_length: int = 3500,
        use_oneshot: bool = False
    ):
        """
        初始化适配器
        
        Args:
            data_path: 数据文件目录
            schema_path: Schema 文件路径
            max_text_length: 文本最大长度（截断阈值）
            use_oneshot: 是否使用 One-Shot 示例（会增加 token 消耗）
        """
        self.data_path = data_path
        self.schema_path = schema_path
        self.max_text_length = max_text_length
        self.use_oneshot = use_oneshot
        self.schema = self._load_schema()
        self.prompt_builder = ChinesePromptBuilder()

    def _load_schema(self) -> Dict[str, List[str]]:
        """
        加载事件 Schema
        
        Returns:
            {event_type: [role1, role2, ...]} 形式的字典
        """
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema 文件不存在: {self.schema_path}")
        
        schema_dict = {}
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    event_type = item.get('event_type', '')
                    roles = [r['role'] for r in item.get('role_list', [])]
                    if event_type:
                        schema_dict[event_type] = roles
                except json.JSONDecodeError as e:
                    logger.warning(f"Schema 解析错误: {e}")
                    continue
        
        logger.info(f"加载 Schema 成功，包含 {len(schema_dict)} 种事件类型")
        return schema_dict

    def get_event_types(self) -> List[str]:
        """获取所有事件类型"""
        return list(self.schema.keys())

    def get_roles_for_event(self, event_type: str) -> List[str]:
        """获取指定事件类型的角色列表"""
        return self.schema.get(event_type, [])

    def _build_prompt(self, text: str) -> str:
        """
        构建用户输入 Prompt
        
        Args:
            text: 原始文档文本
        
        Returns:
            格式化的用户 Prompt
        """
        # 截断过长文本
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "...[文本已截断]"
        
        return self.prompt_builder.build_user_prompt(text)

    def _build_chosen_response(self, event_list: List[Dict]) -> str:
        """
        构建标准答案响应（包含中文 CoT）
        
        Args:
            event_list: 标注的事件列表
        
        Returns:
            包含思维链和 JSON 的完整响应
        """
        return self.prompt_builder.build_cot_response(event_list, self.schema)

    def load_data(self, split: str = "train") -> List[DUEE_FIN_EESample]:
        """
        加载数据集
        
        Args:
            split: 数据集划分 (train/dev/test)
        
        Returns:
            样本列表
        """
        file_path = os.path.join(self.data_path, f"duee_fin_{split}.json")
        
        if not os.path.exists(file_path):
            logger.error(f"数据文件不存在: {file_path}")
            return []

        samples = []
        error_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    error_count += 1
                    logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                    continue
                
                text = entry.get('text', '')
                if not text:
                    continue
                
                # 截断过长文本
                original_length = len(text)
                if original_length > self.max_text_length:
                    text = text[:self.max_text_length] + "..."
                
                # 获取事件列表
                event_list = entry.get('event_list', [])
                
                # 提取事件类型（用于 DS-CNS）
                event_types = list(set([
                    e.get('event_type', '') 
                    for e in event_list 
                    if e.get('event_type')
                ]))
                
                # 构建 Prompt（中文化）
                prompt = self._build_prompt(text)
                
                # 构建 Chosen 响应（中文 CoT + JSON）
                chosen_response = self._build_chosen_response(event_list)
                
                samples.append(DUEE_FIN_EESample(
                    id=entry.get('id', f'sample_{line_num}'),
                    text=text,
                    prompt=prompt,
                    chosen=chosen_response,
                    event_types=event_types,
                    events=event_list  # 存储原始事件列表
                ))
        
        if error_count > 0:
            logger.warning(f"共 {error_count} 条数据解析失败")
        
        logger.info(f"从 {split} 集加载 {len(samples)} 条样本")
        return samples

    def load_data_with_schema_validation(self, split: str = "train") -> List[DUEE_FIN_EESample]:
        """
        加载数据并进行 Schema 验证
        
        Args:
            split: 数据集划分
        
        Returns:
            经过验证的样本列表
        """
        samples = self.load_data(split)
        
        validated_samples = []
        invalid_count = 0
        
        for sample in samples:
            # 解析 chosen 中的 JSON
            try:
                # 提取 JSON 部分
                json_start = sample.chosen.find("```json")
                json_end = sample.chosen.rfind("```")
                
                if json_start == -1 or json_end == -1:
                    validated_samples.append(sample)
                    continue
                
                json_str = sample.chosen[json_start + 7:json_end].strip()
                events = json.loads(json_str)
                
                # 验证事件类型是否在 Schema 中
                valid = True
                for event in events:
                    etype = event.get('event_type', '')
                    if etype and etype not in self.schema:
                        logger.warning(f"样本 {sample.id} 包含未知事件类型: {etype}")
                        valid = False
                        break
                
                if valid:
                    validated_samples.append(sample)
                else:
                    invalid_count += 1
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"样本 {sample.id} 验证失败: {e}")
                validated_samples.append(sample)  # 仍然保留，让训练时处理
        
        if invalid_count > 0:
            logger.warning(f"共 {invalid_count} 条样本未通过 Schema 验证")
        
        return validated_samples


def create_adapter(data_dir: str, dataset_name: str = "DuEE-Fin") -> DuEEFinAdapter:
    """
    工厂函数：创建数据适配器
    
    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
    
    Returns:
        DuEEFinAdapter 实例
    """
    dataset_name_lower = dataset_name.lower().replace("-", "_")
    schema_path = os.path.join(data_dir, f"{dataset_name_lower}_event_schema.json")
    
    return DuEEFinAdapter(
        data_path=data_dir,
        schema_path=schema_path
    )
