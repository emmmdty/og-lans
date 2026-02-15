"""
OG-LANS: Ontology-Graph Loss-Aware Adaptive Negative Sampling
本体图感知的损失自适应负采样

============================================================================
学术创新点（与 DA-DPO、Hard Negative DPO 的核心差异）:
============================================================================
1. 【本体图语义距离】: 基于领域知识图谱的可解释难度控制（vs DA-DPO 的黑盒难度估计）
2. 【在线能力评估】: 基于 DPO 损失的实时能力追踪（vs Hard Negative DPO 的离线验证器）
3. 【对比梯度放大 (CGA)】: 新增对比学习信号增强机制
4. 【多粒度负样本策略】: 支持事件级、论元级、数值级三层扰动

核心算法:
- Taxonomic Distance: 本体图最短路径距离
- Root Pacing Function: 根式起搏课程控制
- LANS: Loss-Aware Adaptive Negative Sampling
- CGA: Contrastive Gradient Amplification (2026 新增)

论文定位: ACL 2026 Findings / EMNLP 2026 / FinNLP Workshop
============================================================================
"""

import networkx as nx
import json
import random
import numpy as np
import logging
import re
import os
import threading
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import defaultdict

logger = logging.getLogger("OGLANS")


# ============================================================================
# LANS: Loss-Aware Adaptive Negative Sampling
# 学术创新核心模块 - 解决静态课程学习的局限性
# ============================================================================

class LANSScheduler:
    """
    Loss-Aware Adaptive Negative Sampling Scheduler
    基于损失的自适应负采样调度器 (OG-LANS 核心组件)
    
    数学公式:
    - 能力评估: C(t) = EMA(σ(α - L_DPO))
    - 自适应阈值: λ(C) = D_max - (D_max - D_min) · C
    - 对比梯度放大: w(C) = 1 + β_cga · (1 - C)
    """
    
    def __init__(
        self,
        d_max: float,
        d_min: float = 1.0,
        ema_decay: float = 0.95,
        loss_baseline: float = 0.5,
        warmup_steps: int = 100,
        competence_floor: float = 0.05,
        competence_ceiling: float = 0.95,
        warmup_target: float = 0.25,
        use_ema: bool = True,  # 【消融实验开关】是否使用 EMA 平滑
        cga_beta: float = 0.5,  # 对比梯度放大系数
        use_cga: bool = True,   # 【消融实验开关】是否使用 CGA
        granularity_weights: Optional[Dict[str, float]] = None,  # 多粒度权重
        easy_ratio: float = 0.7,
        hard_ratio: float = 0.4
    ):
        """
        初始化 LANS 调度器

        Args:
            d_max: 分类学图谱的最大直径（从 DSCNSampler 获取）
            d_min: 最小距离阈值（通常为 1）
            ema_decay: 能力值指数移动平均的衰减系数（越大越平滑）
            loss_baseline: DPO 损失基准线 α（期望收敛值，IPO 默认 0.5）
            warmup_steps: 预热步数，预热期间使用渐进式能力提升
            competence_floor: 能力下限（避免极端困难采样）
            competence_ceiling: 能力上限（保留探索空间）
            warmup_target: 预热结束时的目标能力值（0-1之间）
            use_ema: 是否使用 EMA 平滑（消融实验 A4: w/o EMA 时设为 False）
            cga_beta: 对比梯度放大系数（2026 新增，默认 0.5）
            use_cga: 是否使用对比梯度放大（消融实验开关）
            granularity_weights: 多粒度扰动权重 {event_level, argument_level, value_level}
            easy_ratio: EASY 策略的距离比例阈值 (默认 0.7)
            hard_ratio: HARD 策略的距离比例阈值 (默认 0.4)
        """
        self.d_max = d_max
        self.d_min = d_min
        self.ema_decay = ema_decay
        self.loss_baseline = loss_baseline
        self.warmup_steps = warmup_steps
        self.competence_floor = competence_floor
        self.competence_ceiling = competence_ceiling
        self.warmup_target = warmup_target
        self.use_ema = use_ema  # 【消融实验】EMA 开关

        # 策略阈值
        self.easy_ratio = easy_ratio
        self.hard_ratio = hard_ratio
        
        # 对比梯度放大 (Contrastive Gradient Amplification)
        self.cga_beta = cga_beta
        self.use_cga = use_cga
        
        # 多粒度扰动权重（支持配置化）
        if granularity_weights is not None:
            self.granularity_weights = {
                "EVENT_LEVEL": granularity_weights.get("event_level", 0.3),
                "ARGUMENT_LEVEL": granularity_weights.get("argument_level", 0.5),
                "VALUE_LEVEL": granularity_weights.get("value_level", 0.2)
            }
        else:
            # 默认权重
            self.granularity_weights = {
                "EVENT_LEVEL": 0.3,
                "ARGUMENT_LEVEL": 0.5,
                "VALUE_LEVEL": 0.2
            }
        
        # 状态变量（线程安全）
        # 使用 RLock 而不是 Lock，允许可重入，避免 export_history 死锁
        self._lock = threading.RLock()
        self._competence: float = competence_floor  # 初始能力 = 下限
        self._step_count: int = 0
        self._loss_history: List[float] = []
        self._competence_history: List[float] = []
        self._cga_weight_history: List[float] = []  # CGA 权重历史
        self._max_history_size: int = 10000  # 限制历史记录大小
        
        # 策略分布统计
        self._strategy_counts: Dict[str, int] = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
        self._recent_strategies: List[str] = []  # 【新增】记录最近的策略选择（滑动窗口）
        self._max_recent_strategies: int = 200   # 滑动窗口大小
        
        # 多粒度策略统计
        self._granularity_counts: Dict[str, int] = {
            "EVENT_LEVEL": 0,    # 事件级扰动
            "ARGUMENT_LEVEL": 0, # 论元级扰动
            "VALUE_LEVEL": 0     # 数值级扰动
        }
        
        logger.info(
            f"OG-LANS 调度器初始化: "
            f"D_max={d_max:.2f}, D_min={d_min:.2f}, "
            f"EMA_decay={ema_decay}, Loss_baseline={loss_baseline}, "
            f"Warmup={warmup_steps}, CGA_beta={cga_beta}, "
            f"Granularity_weights={self.granularity_weights}"
        )
    
    @property
    def competence(self) -> float:
        """当前能力值（线程安全读取）"""
        with self._lock:
            return self._competence
    
    @property
    def current_threshold(self) -> float:
        """当前距离阈值 λ(C)"""
        return self._compute_threshold(self.competence)
    
    @property
    def cga_weight(self) -> float:
        """
        对比梯度放大权重 w(C)
        
        公式: w(C) = 1 + β_cga · (1 - C)
        
        解释:
        - C ≈ 0 (能力弱): w ≈ 1 + β_cga → 放大困难样本的梯度
        - C ≈ 1 (能力强): w ≈ 1 → 正常梯度
        
        这确保模型在能力不足时，从困难负样本获得更强的学习信号
        """
        if not self.use_cga:
            return 1.0
        return 1.0 + self.cga_beta * (1.0 - self.competence)
    
    def _compute_threshold(self, competence: float) -> float:
        """
        自适应起搏函数: λ(C) = D_max - (D_max - D_min) · C
        
        解释:
        - C ≈ 0: 模型能力弱 → λ ≈ D_max → 采样远距离（简单）负样本
        - C ≈ 1: 模型能力强 → λ ≈ D_min → 采样近距离（困难）负样本
        """
        return self.d_max - (self.d_max - self.d_min) * competence
    
    def update_competence(self, batch_loss: float) -> float:
        """
        根据 Batch 损失更新能力估计（核心创新点）
        
        公式: C(t) = EMA(σ(α - L))
        
        解释:
        - 当 L < α（损失低于基准）: σ(α - L) > 0.5 → 能力提升
        - 当 L > α（损失高于基准）: σ(α - L) < 0.5 → 能力下降
        - EMA 确保能力变化平滑，避免单个异常 Batch 导致策略剧烈震荡
        
        Args:
            batch_loss: 当前 Batch 的 DPO 损失值
        
        Returns:
            更新后的能力值
        """
        with self._lock:
            self._step_count += 1
            self._loss_history.append(batch_loss)
            
            # 预热期：使用线性渐进能力，避免冷启动问题
            if self._step_count <= self.warmup_steps:
                warmup_progress = self._step_count / self.warmup_steps
                # 预热期能力从 floor 线性增长到 floor + warmup_target
                warmup_competence = min(
                    self.competence_floor + self.warmup_target * warmup_progress,
                    self.competence_ceiling
                )
                self._competence = warmup_competence
                self._competence_history.append(self._competence)
                return self._competence
            
            # 正常期：基于损失的自适应更新
            # Step 1: 计算即时能力信号
            #   σ(α - L) ∈ (0, 1)
            #   当 L = α 时，σ(0) = 0.5
            #   当 L < α 时，σ(α - L) > 0.5（表现好，能力高）
            #   当 L > α 时，σ(α - L) < 0.5（表现差，能力低）
            instant_signal = self._sigmoid(self.loss_baseline - batch_loss)
            
            # Step 2: EMA 平滑更新 或 直接使用即时信号（消融实验 A4）
            #   use_ema=True:  C_new = γ · C_old + (1 - γ) · instant_signal
            #   use_ema=False: C_new = instant_signal（无平滑，用于消融实验）
            if self.use_ema:
                new_competence = (
                    self.ema_decay * self._competence + 
                    (1 - self.ema_decay) * instant_signal
                )
            else:
                # 【消融实验 A4】直接使用即时信号，无 EMA 平滑
                new_competence = instant_signal
            
            # Step 3: Clamp 到合法范围
            self._competence = max(
                self.competence_floor,
                min(self.competence_ceiling, new_competence)
            )
            
            # 记录 CGA 权重
            cga_w = 1.0 + self.cga_beta * (1.0 - self._competence) if self.use_cga else 1.0
            self._cga_weight_history.append(cga_w)
            
            # 限制历史记录大小，防止内存泄漏
            self._competence_history.append(self._competence)
            if len(self._competence_history) > self._max_history_size:
                self._competence_history = self._competence_history[-self._max_history_size:]
            if len(self._loss_history) > self._max_history_size:
                self._loss_history = self._loss_history[-self._max_history_size:]
            if len(self._cga_weight_history) > self._max_history_size:
                self._cga_weight_history = self._cga_weight_history[-self._max_history_size:]
            
            # 周期性日志
            if self._step_count % 50 == 0:
                logger.info(
                    f"OG-LANS 更新 [Step {self._step_count}]: "
                    f"Loss={batch_loss:.4f}, Competence={self._competence:.4f}, "
                    f"Threshold={self.current_threshold:.2f}, CGA_w={cga_w:.3f}"
                )
            
            return self._competence
    
    def get_strategy(self) -> str:
        """
        根据当前能力值和多粒度权重确定负采样策略
        
        融合能力阈值与 granularity_weights 配置:
        1. 根据能力值确定基础概率分布
        2. 使用 granularity_weights 调整最终分布
        
        策略映射:
        - EASY → EVENT_LEVEL (事件类型替换)
        - MEDIUM → ARGUMENT_LEVEL (角色错位)
        - HARD → VALUE_LEVEL (数值微扰)
        """
        with self._lock:
            threshold = self._compute_threshold(self._competence)
            
            # Step 1: 基于能力阈值计算基础概率
            threshold_range = self.d_max - self.d_min
            easy_boundary = self.d_min + threshold_range * self.easy_ratio
            hard_boundary = self.d_min + threshold_range * self.hard_ratio
            
            if threshold > easy_boundary:
                # 能力弱：倾向简单样本
                base_probs = {"EASY": 0.6, "MEDIUM": 0.3, "HARD": 0.1}
            elif threshold > hard_boundary:
                # 能力中等：均衡分布
                base_probs = {"EASY": 0.2, "MEDIUM": 0.5, "HARD": 0.3}
            else:
                # 能力强：倾向困难样本
                base_probs = {"EASY": 0.1, "MEDIUM": 0.2, "HARD": 0.7}
            
            # Step 2: 使用 granularity_weights 调整概率
            # 权重映射: EVENT_LEVEL→EASY, ARGUMENT_LEVEL→MEDIUM, VALUE_LEVEL→HARD
            weight_easy = self.granularity_weights.get("EVENT_LEVEL", 0.3)
            weight_medium = self.granularity_weights.get("ARGUMENT_LEVEL", 0.5)
            weight_hard = self.granularity_weights.get("VALUE_LEVEL", 0.2)
            
            # 融合: adjusted = base_prob * weight (然后归一化)
            adjusted_probs = {
                "EASY": base_probs["EASY"] * weight_easy,
                "MEDIUM": base_probs["MEDIUM"] * weight_medium,
                "HARD": base_probs["HARD"] * weight_hard
            }
            
            # 归一化
            total = sum(adjusted_probs.values())
            if total > 0:
                normalized_probs = {k: v / total for k, v in adjusted_probs.items()}
            else:
                normalized_probs = {"EASY": 0.33, "MEDIUM": 0.34, "HARD": 0.33}
            
            # Step 3: 加权随机采样
            strategies = list(normalized_probs.keys())
            weights = list(normalized_probs.values())
            strategy = random.choices(strategies, weights=weights, k=1)[0]

            self._strategy_counts[strategy] += 1

            # 【新增】记录到滑动窗口（用于瞬时统计）
            self._recent_strategies.append(strategy)
            if len(self._recent_strategies) > self._max_recent_strategies:
                self._recent_strategies = self._recent_strategies[-self._max_recent_strategies:]
        
        return strategy
    
    def get_granularity(self, strategy: str) -> str:
        """
        根据策略返回对应的扰动粒度（用于统计记录）
        
        映射关系:
        - EASY → EVENT_LEVEL (事件类型替换)
        - MEDIUM → ARGUMENT_LEVEL (论元角色交换)
        - HARD → VALUE_LEVEL (数值/实体微扰)
        
        注意: 多粒度权重已在 get_strategy() 中应用，此方法仅用于统计
        """
        with self._lock:
            # 简单的策略到粒度映射
            granularity_map = {
                "EASY": "EVENT_LEVEL",
                "MEDIUM": "ARGUMENT_LEVEL",
                "HARD": "VALUE_LEVEL"
            }
            granularity = granularity_map.get(strategy, "ARGUMENT_LEVEL")
            self._granularity_counts[granularity] += 1
            return granularity
    
    def get_statistics(self) -> Dict:
        """获取调度器统计信息（用于 TensorBoard 记录）"""
        with self._lock:
            total = sum(self._strategy_counts.values()) or 1
            granularity_total = sum(self._granularity_counts.values()) or 1
            # 安全计算平均损失
            recent_losses = self._loss_history[-50:] if self._loss_history else []
            recent_avg = float(np.mean(recent_losses)) if recent_losses else 0.0

            # CGA 统计
            recent_cga = self._cga_weight_history[-50:] if self._cga_weight_history else [1.0]
            avg_cga = float(np.mean(recent_cga))

            # 【修复】计算瞬时策略分布（基于滑动窗口）
            # 使用最近 100 个样本的策略分布，而非累积比例
            recent_window_size = 100
            recent_strategy_counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
            if hasattr(self, '_recent_strategies'):
                for s in self._recent_strategies[-recent_window_size:]:
                    recent_strategy_counts[s] = recent_strategy_counts.get(s, 0) + 1
            recent_total = sum(recent_strategy_counts.values()) or 1

            return {
                "competence": self._competence,
                "threshold": self._compute_threshold(self._competence),
                "cga_weight": 1.0 + self.cga_beta * (1.0 - self._competence) if self.use_cga else 1.0,
                "avg_cga_weight": avg_cga,
                "step_count": self._step_count,
                "strategy_distribution": {
                    k: v / recent_total for k, v in recent_strategy_counts.items()
                },
                "cumulative_strategy_distribution": {
                    k: v / total for k, v in self._strategy_counts.items()
                },
                "granularity_distribution": {
                    k: v / granularity_total for k, v in self._granularity_counts.items()
                },
                "loss_history_len": len(self._loss_history),
                "recent_avg_loss": recent_avg
            }
    
    def reset_epoch(self):
        """
        重置 Epoch 相关状态（可选）
        
        注意：OG-LANS 的设计允许跨 Epoch 保持能力记忆
        调用此方法会"部分重置"，保留一定能力记忆
        """
        with self._lock:
            # 保留 50% 的历史能力作为下一 Epoch 的起点
            # 这允许模型在新 Epoch 中更快达到高难度
            self._competence = max(
                self.competence_floor,
                self._competence * 0.5
            )
            self._strategy_counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
            self._granularity_counts = {
                "EVENT_LEVEL": 0,
                "ARGUMENT_LEVEL": 0,
                "VALUE_LEVEL": 0
            }
            logger.info(
                f"OG-LANS Epoch 重置: 保留能力 {self._competence:.4f}"
            )
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid"""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)
    
    def export_history(self) -> Dict:
        """导出训练历史（用于可视化分析和论文图表）"""
        # 使用 RLock 后可以安全嵌套，但仍然分步读取以提高性能
        with self._lock:
            loss_history = list(self._loss_history)
            competence_history = list(self._competence_history)
            cga_weight_history = list(self._cga_weight_history)
            config = {
                "d_max": self.d_max,
                "d_min": self.d_min,
                "ema_decay": self.ema_decay,
                "loss_baseline": self.loss_baseline,
                "warmup_steps": self.warmup_steps,
                "cga_beta": self.cga_beta,
                "use_cga": self.use_cga,
                "use_ema": self.use_ema
            }
        
        # 在锁外调用 get_statistics，避免死锁（虽然 RLock 已修复，保持良好实践）
        stats = self.get_statistics()
        
        return {
            "loss_history": loss_history,
            "competence_history": competence_history,
            "cga_weight_history": cga_weight_history,
            "final_statistics": stats,
            "config": config
        }


class OntologyGraphAnalyzer:
    """
    本体图分析器
    
    提供本体图的深度分析功能，用于论文中的可视化和可解释性分析
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._event_types = [
            n for n, d in graph.nodes(data=True) 
            if d.get('type') == 'event_type'
        ]
    
    def compute_semantic_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        计算事件类型间的语义相似度矩阵
        
        相似度定义: sim(t1, t2) = 1 - d(t1, t2) / D_max
        """
        try:
            d_max = nx.diameter(self.graph)
        except:
            d_max = 4
        
        similarity_matrix = {}
        for t1 in self._event_types:
            similarity_matrix[t1] = {}
            for t2 in self._event_types:
                if t1 == t2:
                    similarity_matrix[t1][t2] = 1.0
                else:
                    try:
                        dist = nx.shortest_path_length(self.graph, t1, t2)
                        similarity_matrix[t1][t2] = 1.0 - dist / d_max
                    except:
                        similarity_matrix[t1][t2] = 0.0
        
        return similarity_matrix
    
    def get_sibling_types(self, event_type: str) -> List[str]:
        """获取兄弟事件类型（同一父节点下的类型）"""
        siblings = []
        for t in self._event_types:
            if t != event_type:
                try:
                    dist = nx.shortest_path_length(self.graph, event_type, t)
                    if dist == 2:  # 经过共同父节点
                        siblings.append(t)
                except:
                    continue
        return siblings
    
    def analyze_graph_structure(self) -> Dict:
        """分析图谱结构，用于论文 Appendix"""
        return {
            "num_event_types": len(self._event_types),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_connected": nx.is_connected(self.graph),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else -1,
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }


class DSCNSampler:
    """
    OG-CNS: Ontology-Graph Contrastive Negative Sampler
    本体图对比负采样器 (2026 Final Version)
    
    ============================================================================
    学术创新声明:
    ============================================================================
    本采样器是 OG-LANS 框架的核心组件，实现了基于领域本体图的动态负样本生成。
    
    与现有方法的关键区别:
    1. vs Random Negative: 使用本体图语义距离而非随机选择
    2. vs DA-DPO: 基于可解释的图距离而非黑盒难度估计
    3. vs Hard Negative DPO: 无需额外训练验证器模型
    
    核心特性:
    - 本体图构建: 从 Schema 自动构建事件类型关系图
    - 语义距离计算: 基于最短路径的分类学距离
    - 多粒度扰动: 事件级 / 论元级 / 数值级三层策略
    - 自适应课程: 与 LANS 调度器深度集成
    ============================================================================
    """
    
    def __init__(
        self, 
        schema_path: str, 
        c0: float = 0.1, 
        graph_cache_path: Optional[str] = None,
        use_ontology_distance: bool = True,  # 是否使用本体图语义距离
        static_easy_ratio: float = 0.7,
        static_hard_ratio: float = 0.4
    ):
        """
        初始化 DS-CNS 采样器
        
        Args:
            schema_path: Schema 文件路径
            c0: 初始能力值 (Initial Competence, 范围 0-1)
            graph_cache_path: 图谱缓存路径（可选）
            use_ontology_distance: 是否使用本体图语义距离（False 时退化为随机采样，用于消融实验 A6）
        """
        self.c0 = max(0.01, min(c0, 1.0))  # 确保 c0 在合理范围
        self.schema_path = schema_path
        self.use_ontology_distance = use_ontology_distance  # 【消融实验 A6 开关】
        self.static_easy_ratio = static_easy_ratio
        self.static_hard_ratio = static_hard_ratio
        
        # 加载或构建图谱
        if graph_cache_path and os.path.exists(graph_cache_path):
            try:
                logger.info(f"从缓存加载分类学图谱: {graph_cache_path}")
                self.graph = nx.read_gml(graph_cache_path)
            except Exception as e:
                logger.warning(f"缓存加载失败，重新构建: {e}")
                self.graph = self._build_taxonomy_graph(schema_path)
        else:
            self.graph = self._build_taxonomy_graph(schema_path)
        
        # 提取事件类型和角色信息
        self.event_types = [
            n for n, d in self.graph.nodes(data=True) 
            if d.get('type') == 'event_type'
        ]
        
        # 存储每个事件类型的角色
        self.event_roles: Dict[str, List[str]] = {}
        for n, d in self.graph.nodes(data=True):
            if d.get('type') == 'role':
                # 节点 ID 格式: "事件类型::角色名"
                parts = n.split("::")
                if len(parts) == 2:
                    etype, role = parts
                    if etype not in self.event_roles:
                        self.event_roles[etype] = []
                    self.event_roles[etype].append(role)
        
        # 计算图谱直径（最大距离）
        try:
            if nx.is_connected(self.graph):
                self.max_diameter = nx.diameter(self.graph)
            else:
                # 非连通图，取最大连通分量的直径
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                self.max_diameter = nx.diameter(subgraph)
        except Exception:
            self.max_diameter = 4  # 默认值
        
        self.min_dist = 1
        
        # 【LANS 集成】可选的自适应调度器（将在启用 LANS 时初始化）
        self.lans_scheduler: Optional[LANSScheduler] = None
        self._use_lans: bool = False
        
        # 【注意】OntologyGraphAnalyzer 类保留但暂不实例化
        # 该分析器用于未来扩展：关键路径分析、瓶颈检测等
        # 如需启用，取消下行注释:
        # self.graph_analyzer = OntologyGraphAnalyzer(self.graph)
        
        logger.info(
            f"OG-CNS 初始化完成: "
            f"{len(self.event_types)} 种事件类型, "
            f"图谱直径={self.max_diameter}, "
            f"c0={self.c0}"
        )
    
    def enable_lans(
        self,
        ema_decay: float = 0.95,
        loss_baseline: float = 0.5,
        warmup_steps: int = 100,
        competence_floor: float = 0.05,
        competence_ceiling: float = 0.95,
        warmup_target: float = 0.25,
        use_ema: bool = True,
        cga_beta: float = 0.5,
        use_cga: bool = True,
        granularity_weights: Optional[Dict[str, float]] = None,
        easy_ratio: float = 0.7,
        hard_ratio: float = 0.4
    ) -> LANSScheduler:
        """
        启用 OG-LANS 自适应调度器（替代静态课程学习）
        
        Args:
            ema_decay: EMA 衰减系数
            loss_baseline: 损失基准线（IPO 损失默认 0.5）
            warmup_steps: 预热步数
            competence_floor: 能力下限
            competence_ceiling: 能力上限
            warmup_target: 预热结束时的目标能力增量
            use_ema: 是否使用 EMA 平滑（消融实验 A4: w/o EMA 时设为 False）
            cga_beta: 对比梯度放大系数（2026 新增）
            use_cga: 是否使用对比梯度放大（消融实验开关）
            granularity_weights: 多粒度扰动权重配置 {event_level, argument_level, value_level}
        
        Returns:
            LANSScheduler 实例
        """
        self.lans_scheduler = LANSScheduler(
            d_max=self.max_diameter,
            d_min=self.min_dist,
            ema_decay=ema_decay,
            loss_baseline=loss_baseline,
            warmup_steps=warmup_steps,
            competence_floor=competence_floor,
            competence_ceiling=competence_ceiling,
            warmup_target=warmup_target,
            use_ema=use_ema,
            cga_beta=cga_beta,
            use_cga=use_cga,
            granularity_weights=granularity_weights,  # 传递多粒度权重配置
            easy_ratio=easy_ratio,
            hard_ratio=hard_ratio
        )
        self._use_lans = True
        logger.info("✅ OG-LANS 自适应调度器已启用")
        return self.lans_scheduler
    
    def get_negative_strategy_adaptive(self) -> str:
        """
        获取负采样策略（LANS 自适应模式）
        
        注意：调用此方法前需先调用 enable_lans()
        
        Returns:
            策略名称: "EASY" / "MEDIUM" / "HARD"
        """
        if not self._use_lans or self.lans_scheduler is None:
            raise RuntimeError(
                "LANS 未启用。请先调用 enable_lans() 或使用 get_negative_strategy()"
            )
        return self.lans_scheduler.get_strategy()
    
    def _build_taxonomy_graph(self, schema_path: str) -> nx.Graph:
        """
        构建事件分类学图谱 G=(V, E)
        
        图谱结构:
        ROOT -> EventType1 -> Role1_1, Role1_2, ...
             -> EventType2 -> Role2_1, Role2_2, ...
             ...
        
        Returns:
            NetworkX 图对象
        """
        G = nx.Graph()
        G.add_node("ROOT", type="root")
        
        if not os.path.exists(schema_path):
            logger.error(f"Schema 文件不存在: {schema_path}")
            return G
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    schema = json.loads(line)
                    etype = schema.get('event_type', '')
                    
                    if not etype:
                        continue
                    
                    # 添加事件类型节点
                    G.add_node(etype, type="event_type")
                    G.add_edge("ROOT", etype, weight=1)
                    
                    # 添加角色节点
                    for role_obj in schema.get('role_list', []):
                        role_name = role_obj.get('role', '')
                        if role_name:
                            # 使用 "事件类型::角色" 作为唯一 ID
                            node_id = f"{etype}::{role_name}"
                            G.add_node(node_id, type="role", role_name=role_name)
                            G.add_edge(etype, node_id, weight=1)
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Schema 行解析失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Schema 构建失败: {e}")
        
        logger.info(f"图谱构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        return G
    
    def get_taxonomic_distance(self, type1: str, type2: str) -> int:
        """
        计算两个事件类型之间的分类学距离
        
        Args:
            type1: 事件类型 1
            type2: 事件类型 2
        
        Returns:
            最短路径长度（若不可达则返回 max_diameter）
        """
        if type1 == type2:
            return 0
        
        if type1 not in self.graph or type2 not in self.graph:
            return int(self.max_diameter)
        
        try:
            return int(nx.shortest_path_length(self.graph, type1, type2))
        except nx.NetworkXNoPath:
            return int(self.max_diameter)
    
    def compute_pacing_threshold(self, current_step: int, total_steps: int) -> float:
        """
        根式起搏函数 (Root Pacing Function)
        
        公式: λ(t) = D_max - (D_max - D_min) * sqrt(t/T * (1 - c0²) + c0²)
        
        含义:
        - t=0 时: λ ≈ D_max (采样远距离/简单样本)
        - t=T 时: λ ≈ D_min (采样近距离/困难样本)
        
        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        
        Returns:
            当前的距离阈值 λ(t)
        """
        if total_steps <= 0:
            total_steps = 1
        
        # 归一化进度 t/T
        progress = min(current_step / total_steps, 1.0)
        
        # 计算起搏值
        c0_sq = self.c0 ** 2
        pacing_value = np.sqrt(progress * (1 - c0_sq) + c0_sq)
        
        # 映射到距离阈值 [D_min, D_max]
        # 注意: pacing_value 从 c0 增长到 1.0
        # 我们希望阈值从 D_max 下降到 D_min
        threshold = self.max_diameter - (self.max_diameter - self.min_dist) * pacing_value
        
        return threshold
    
    def get_negative_strategy(self, current_step: int, total_steps: int) -> str:
        """
        根据训练进度确定负采样策略
        
        策略映射:
        - λ > D_max * 0.7: EASY (类型替换)
        - D_max * 0.4 < λ <= D_max * 0.7: MEDIUM (角色错位)
        - λ <= D_max * 0.4: HARD (数值微扰)
        
        Returns:
            策略名称: "EASY" / "MEDIUM" / "HARD"
        """
        if total_steps <= 0:
            total_steps = 1
        
        # 计算当前距离阈值
        threshold = self.compute_pacing_threshold(current_step, total_steps)
        
        # 基于阈值确定策略
        easy_boundary = self.max_diameter * self.static_easy_ratio
        hard_boundary = self.max_diameter * self.static_hard_ratio
        
        # 添加随机性以增加多样性
        rand = random.random()
        
        if threshold > easy_boundary:
            # 初期阶段：主要 EASY，少量 MEDIUM
            if rand < 0.7:
                return "EASY"
            elif rand < 0.9:
                return "MEDIUM"
            else:
                return "HARD"
        
        elif threshold > hard_boundary:
            # 中期阶段：均衡分布
            if rand < 0.2:
                return "EASY"
            elif rand < 0.6:
                return "MEDIUM"
            else:
                return "HARD"
        
        else:
            # 后期阶段：主要 HARD
            if rand < 0.1:
                return "EASY"
            elif rand < 0.25:
                return "MEDIUM"
            else:
                return "HARD"
    
    def select_confusing_event_type(
        self, 
        original_type: str, 
        strategy: str,
        current_step: int = 0,
        total_steps: int = 1
    ) -> str:
        """
        根据分类学距离选择混淆事件类型
        
        Args:
            original_type: 原始事件类型
            strategy: 采样策略
            current_step: 当前步数
            total_steps: 总步数
        
        Returns:
            选中的混淆事件类型
        """
        if not self.event_types:
            return original_type
        
        candidates = [t for t in self.event_types if t != original_type]
        if not candidates:
            return original_type
        
        # 【消融实验 A6】禁用本体图距离时，直接随机选择
        if not self.use_ontology_distance:
            return random.choice(candidates)
        
        # 计算当前阈值
        threshold = self.compute_pacing_threshold(current_step, total_steps)
        
        if strategy == "EASY":
            # 选择距离较远的类型
            # 计算所有候选的距离
            candidate_dists = {
                t: self.get_taxonomic_distance(original_type, t) 
                for t in candidates
            }
            
            # 优先选择距离 >= threshold 的
            far_candidates = [
                t for t, d in candidate_dists.items() 
                if d >= threshold
            ]
            
            # 【关键修复】如果没有满足阈值的，退化为选择"当前最远距离的节点"
            # 避免完全随机退化破坏课程学习
            if not far_candidates:
                max_dist = max(candidate_dists.values())
                far_candidates = [
                    t for t, d in candidate_dists.items() 
                    if d == max_dist
                ]
                logger.debug(
                    f"DS-CNS EASY: 无节点满足 distance >= {threshold:.2f}，"
                    f"退化为最远距离 {max_dist} 的节点"
                )
            
            if far_candidates:
                return random.choice(far_candidates)
        
        elif strategy == "HARD":
            # 选择距离较近的类型（兄弟类型）
            # 计算所有候选的距离
            candidate_dists = {
                t: self.get_taxonomic_distance(original_type, t) 
                for t in candidates
            }
            
            # 优先选择距离 <= threshold 的
            near_candidates = [
                t for t, d in candidate_dists.items() 
                if d <= threshold
            ]
            
            # 【关键修复】如果没有满足阈值的，退化为选择"当前最近距离的节点"
            if not near_candidates:
                min_dist = min(candidate_dists.values())
                near_candidates = [
                    t for t, d in candidate_dists.items() 
                    if d == min_dist
                ]
                logger.debug(
                    f"DS-CNS HARD: 无节点满足 distance <= {threshold:.2f}，"
                    f"退化为最近距离 {min_dist} 的节点"
                )
            
            if near_candidates:
                return random.choice(near_candidates)
        
        # 默认随机选择
        return random.choice(candidates)
    
    def generate_negative_json(
        self, 
        correct_json_str: str, 
        strategy: str,
        current_step: int = 0,
        total_steps: int = 1
    ) -> str:
        """
        根据策略生成负样本 JSON
        
        Args:
            correct_json_str: 正确答案的 JSON 字符串（可能包含 CoT）
            strategy: 采样策略 (EASY/MEDIUM/HARD)
            current_step: 当前训练步数
            total_steps: 总训练步数
        
        Returns:
            负样本的 JSON 字符串
        """
        try:
            # 提取 JSON 部分
            json_start = correct_json_str.find("```json")
            json_end = correct_json_str.rfind("```")
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                content = correct_json_str[json_start + 7:json_end].strip()
            else:
                content = correct_json_str.strip()
            
            events = json.loads(content)
            
        except (json.JSONDecodeError, Exception):
            # 解析失败，返回虚假事件
            fake_type = random.choice(self.event_types) if self.event_types else "未知事件"
            return json.dumps([{
                "event_type": fake_type, 
                "trigger": "未知",
                "arguments": []
            }], ensure_ascii=False)
        
        if not events:
            # 原文无事件，生成一个虚假事件
            fake_type = random.choice(self.event_types) if self.event_types else "未知事件"
            return json.dumps([{
                "event_type": fake_type,
                "trigger": "虚假触发",
                "arguments": [{"role": "未知角色", "argument": "虚假论元"}]
            }], ensure_ascii=False)
        
        neg_events = []
        
        for event in events:
            neg_event = {
                "event_type": event.get("event_type", ""),
                "trigger": event.get("trigger", ""),
                "arguments": [arg.copy() for arg in event.get("arguments", [])]
            }
            
            original_type = event.get("event_type", "")
            
            if strategy == "EASY":
                # 策略 A: 事件类型替换
                neg_event["event_type"] = self.select_confusing_event_type(
                    original_type, strategy, current_step, total_steps
                )
            
            elif strategy == "MEDIUM":
                # 策略 B: 角色错位 (Role Swapping)
                args = neg_event["arguments"]
                if len(args) >= 2:
                    # 随机交换两个论元的值
                    idx1, idx2 = random.sample(range(len(args)), 2)
                    args[idx1]["argument"], args[idx2]["argument"] = \
                        args[idx2]["argument"], args[idx1]["argument"]
                elif len(args) == 1:
                    # 只有一个论元，添加噪声后缀
                    args[0]["argument"] = args[0]["argument"] + "（相关）"
            
            elif strategy == "HARD":
                # 策略 C: 数值/实体微扰 (Perturbation)
                mutated = False
                
                for arg in neg_event["arguments"]:
                    val = str(arg.get("argument", ""))
                    
                    # 尝试识别并扰动数字
                    num_match = re.search(r'(\d+(?:\.\d+)?)', val)
                    if num_match:
                        origin_num_str = num_match.group(1)
                        try:
                            if '.' in origin_num_str:
                                # 浮点数：微调 1-5%
                                origin_num = float(origin_num_str)
                                perturbation = random.uniform(-0.05, 0.05)
                                new_num = origin_num * (1 + perturbation)
                                # 【关键修复】限制只替换第一个匹配，避免全局替换污染
                                new_val = val.replace(origin_num_str, f"{new_num:.2f}", 1)
                            else:
                                # 整数：加减随机值
                                origin_num = int(origin_num_str)
                                perturbation = random.choice([1, 10, 100, -1, -10])
                                new_num = max(0, origin_num + perturbation)
                                # 【关键修复】限制只替换第一个匹配，避免全局替换污染
                                new_val = val.replace(origin_num_str, str(new_num), 1)
                            
                            arg["argument"] = new_val
                            mutated = True
                            break  # 只修改一个，保持高相似度
                            
                        except (ValueError, Exception):
                            continue
                
                if not mutated and neg_event["arguments"]:
                    # 无数字可扰动，尝试文本截断
                    target = random.choice(neg_event["arguments"])
                    original = target["argument"]
                    if len(original) > 2:
                        # 随机截断 1-2 个字符
                        cut_len = random.randint(1, min(2, len(original) - 1))
                        target["argument"] = original[:-cut_len]
            
            neg_events.append(neg_event)
        
        return json.dumps(neg_events, ensure_ascii=False)
    
    def get_strategy_distribution(
        self, 
        current_step: int, 
        total_steps: int,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        获取当前训练阶段的策略分布（用于调试）
        
        Args:
            current_step: 当前步数
            total_steps: 总步数
            num_samples: 采样次数
        
        Returns:
            各策略的比例
        """
        counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
        
        for _ in range(num_samples):
            strategy = self.get_negative_strategy(current_step, total_steps)
            counts[strategy] += 1
        
        return {k: v / num_samples for k, v in counts.items()}
