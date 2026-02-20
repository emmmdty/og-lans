"""
Unsloth DPO Trainer Wrapper (v4.0 OG-LANS Final)
集成 DS-CNS 动态负采样、SCV 语义校验和 CGA 梯度放大的 DPO 训练器

学术创新点:
1. LANS: 基于损失的自适应负采样调度
2. CGA: 对比梯度放大机制 (Contrastive Gradient Amplification)
3. 多粒度扰动策略集成
"""

import unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer

# Patch DPOTrainer globally before importing TRL to ensure patching works for subclasses
PatchDPOTrainer()

import torch
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
import json
import os
import logging
import time
import hashlib
from collections import OrderedDict
# Features, Value 用于显式定义数据集 Schema
from datasets import Dataset, Features, Value
from trl import DPOTrainer, DPOConfig
try:
    from trl import DPODataCollatorWithPadding
except ImportError:
    # Fallback for older trl versions or when not exported
    from dataclasses import dataclass
    from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq
    from typing import Any, Dict, List, Optional, Union

    @dataclass
    class DPODataCollatorWithPadding:
        """
        DPODataCollatorWithPadding 兼容性回退实现
        """
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        label_pad_token_id: int = -100

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # 使用 DataCollatorForSeq2Seq 辅助填充
            base_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                label_pad_token_id=self.label_pad_token_id
            )

            batch = {}

            # 定义需要处理的 DPO 字段组
            # 格式: {目标前缀: {原始字段名: 临时字段名}}
            groups = {
                "chosen": {
                    "chosen_input_ids": "input_ids",
                    "chosen_attention_mask": "attention_mask",
                    "chosen_labels": "labels"
                },
                "rejected": {
                    "rejected_input_ids": "input_ids",
                    "rejected_attention_mask": "attention_mask",
                    "rejected_labels": "labels"
                },
                "prompt": {
                    "prompt_input_ids": "input_ids",
                    "prompt_attention_mask": "attention_mask"
                }
            }

            for group_name, mapping in groups.items():
                # 提取子 batch
                sub_features = []
                for f in features:
                    item = {}
                    has_data = False
                    for orig, temp in mapping.items():
                        if orig in f:
                            item[temp] = f[orig]
                            has_data = True
                    if has_data:
                        sub_features.append(item)

                if not sub_features:
                    continue

                # 填充
                padded = base_collator(sub_features)

                # 还原字段名并合并到 batch
                for orig, temp in mapping.items():
                    if temp in padded:
                        batch[orig] = padded[temp]

            # 复制其他非 Tensor 字段 (如 prompt 文本等)
            if features:
                for k in features[0].keys():
                    if k not in batch:
                        batch[k] = [f[k] for f in features]

            return batch

from ..utils.ds_cns import DSCNSampler, LANSScheduler
from ..utils.scv import SemanticConsistencyVerifier
from ..data.prompt_builder import ChinesePromptBuilder, build_inference_prompt
import random
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("OGLANS")


# ============================================================================
# CGA-Enhanced DPO Trainer: 对比梯度放大机制
# ============================================================================

class CGADPOTrainer(DPOTrainer):
    """
    CGA-Enhanced DPO Trainer (OG-LANS 核心创新组件)
    
    Contrastive Gradient Amplification (CGA) 对比梯度放大:
    - 根据模型当前能力动态调整损失权重
    - 能力弱时放大梯度，增强对比学习信号
    - 能力强时恢复正常，避免过拟合
    
    数学公式: 
        loss_weighted = loss × (1 + β_cga × (1 - C))
    其中 C 是当前能力估计值 (0~1)
    """
    
    def __init__(
        self, 
        *args, 
        lans_scheduler: Optional[LANSScheduler] = None,
        rpo_alpha: float = 0.0,
        rpo_warmup_steps: int = 0,
        aux_log_interval: int = 50,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lans_scheduler = lans_scheduler
        self.rpo_alpha = max(0.0, float(rpo_alpha))
        self.rpo_warmup_steps = max(0, int(rpo_warmup_steps))
        self.aux_log_interval = max(1, int(aux_log_interval))
        self._cga_applied_count = 0
        self._rpo_warning_emitted = False
        self._last_aux_metrics: Dict[str, float] = {}
        
        if self.lans_scheduler is not None:
            logger.info(
                f"🚀 CGADPOTrainer 初始化: "
                f"CGA_enabled={self.lans_scheduler.use_cga}, "
                f"CGA_beta={self.lans_scheduler.cga_beta}"
            )
        if self.rpo_alpha > 0:
            logger.info(
                f"🎯 RPO 混合损失已启用: alpha={self.rpo_alpha:.4f}, "
                f"warmup_steps={self.rpo_warmup_steps}"
            )

    def _compute_rpo_weight(self) -> float:
        """RPO 权重调度：预热后达到目标 alpha。"""
        if self.rpo_alpha <= 0:
            return 0.0
        if self.rpo_warmup_steps <= 0:
            return self.rpo_alpha
        step = int(getattr(self.state, "global_step", 0))
        scale = min(1.0, max(0.0, step / max(self.rpo_warmup_steps, 1)))
        return self.rpo_alpha * scale

    @staticmethod
    def _compute_chosen_sft_loss(
        model: Any,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Optional[torch.Tensor]:
        """计算 chosen 序列的 NLL（SFT anchor）。"""
        chosen_input_ids = inputs.get("chosen_input_ids")
        chosen_attention_mask = inputs.get("chosen_attention_mask")
        chosen_labels = inputs.get("chosen_labels")

        if chosen_input_ids is None or chosen_labels is None:
            return None
        if not isinstance(chosen_labels, torch.Tensor):
            return None
        if not torch.any(chosen_labels != -100):
            return None

        outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_labels,
            use_cache=False,
            return_dict=True,
        )
        sft_loss = getattr(outputs, "loss", None)
        if sft_loss is None:
            return None
        if not torch.isfinite(sft_loss):
            return None
        return sft_loss

    def get_aux_metrics_snapshot(self) -> Dict[str, float]:
        return dict(self._last_aux_metrics)
    
    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        重写损失计算方法，应用 CGA 梯度放大
        
        这是 OG-LANS 与 DA-DPO/Hard Negative DPO 的核心差异:
        - DA-DPO: 使用 VLM 置信度调整采样
        - Hard Neg DPO: 使用验证器筛选样本
        - OG-LANS: 使用本体图距离 + CGA 动态梯度放大
        """
        # 调用父类计算原始损失
        if return_outputs:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
            outputs = None

        pref_loss_raw = loss
        cga_weight = 1.0

        # 应用 CGA 梯度放大（只作用于偏好损失项）
        if self.lans_scheduler is not None and self.lans_scheduler.use_cga:
            cga_weight = self.lans_scheduler.cga_weight
            if self.lans_scheduler._step_count > self.lans_scheduler.warmup_steps:
                pref_loss_raw = pref_loss_raw * cga_weight
                self._cga_applied_count += 1
                if self._cga_applied_count % 100 == 0:
                    logger.debug(
                        f"CGA 梯度放大: weight={cga_weight:.4f}, "
                        f"competence={self.lans_scheduler.competence:.4f}"
                    )

        # RPO: 偏好损失 + alpha * chosen-SFT 锚定项
        rpo_weight = self._compute_rpo_weight()
        sft_loss: Optional[torch.Tensor] = None
        if rpo_weight > 0.0:
            try:
                sft_loss = self._compute_chosen_sft_loss(model, inputs)
            except Exception as exc:
                if not self._rpo_warning_emitted:
                    logger.warning(f"RPO 计算失败，已退化为纯 DPO/IPO: {exc}")
                    self._rpo_warning_emitted = True

        final_loss = pref_loss_raw
        if sft_loss is not None and rpo_weight > 0.0:
            final_loss = pref_loss_raw + rpo_weight * sft_loss

        self._last_aux_metrics = {
            "pref_loss_raw": float(loss.detach().float().item()),
            "pref_loss_weighted": float(pref_loss_raw.detach().float().item()),
            "cga_weight": float(cga_weight),
            "rpo_weight": float(rpo_weight),
            "rpo_sft_loss": float(sft_loss.detach().float().item()) if sft_loss is not None else 0.0,
            "combined_loss": float(final_loss.detach().float().item()),
        }
        if getattr(self.state, "global_step", 0) % self.aux_log_interval == 0:
            logger.debug(
                "loss_components: "
                f"pref={self._last_aux_metrics['pref_loss_weighted']:.4f}, "
                f"rpo_w={self._last_aux_metrics['rpo_weight']:.4f}, "
                f"sft={self._last_aux_metrics['rpo_sft_loss']:.4f}, "
                f"combined={self._last_aux_metrics['combined_loss']:.4f}"
            )

        if return_outputs:
            return final_loss, outputs
        return final_loss


# ============================================================================
# LANS Callback: 训练过程中动态更新能力评估
# ============================================================================

class LANSCallback(TrainerCallback):
    """
    Loss-Aware Adaptive Negative Sampling Callback

    核心功能：
    1. 每个 Step 结束时更新能力值 (Competence)
    2. 每个 Epoch 开始时按配置刷新负样本（动态课程学习）
    3. 记录瞬时策略分布到 TensorBoard
    """

    def __init__(
        self,
        lans_scheduler: LANSScheduler,
        log_interval: int = 10,
        lans_sampler: Optional['LANSNegativeSampler'] = None,
        lans_dataset: Optional['LANSIterableDataset'] = None,
        logging_dir: Optional[str] = None,
        trainer_ref: Optional[Any] = None,  # 【新增】Trainer 引用，用于动态更新数据集
        base_samples: Optional[List[Dict]] = None,  # 【新增】原始样本数据
        tokenizer: Optional[Any] = None,  # 【新增】Tokenizer 引用
        refresh_start_epoch: int = 1,
        refresh_log_interval: int = 100,
        refresh_log_seconds: float = 30.0,
    ):
        self.lans_scheduler = lans_scheduler
        self.log_interval = log_interval
        self.lans_sampler = lans_sampler
        self.lans_dataset = lans_dataset
        self._global_step = 0
        self._writer = SummaryWriter(log_dir=logging_dir) if logging_dir else None
        self.trainer_ref = trainer_ref
        self.base_samples = base_samples
        self.tokenizer = tokenizer
        self._current_epoch = -1  # 跟踪当前 Epoch
        self.refresh_start_epoch = max(0, int(refresh_start_epoch))
        self.refresh_log_interval = max(1, int(refresh_log_interval))
        self.refresh_log_seconds = max(5.0, float(refresh_log_seconds))
        self._refresh_warning_emitted = False

    def on_epoch_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Epoch 开始时动态重新生成负样本（核心创新点）

        基于当前 Competence 刷新整个数据集的 rejected 字段，
        实现真正的课程学习：随着模型能力提升，负样本难度逐渐增加。
        """
        current_epoch = int(state.epoch) if state.epoch else 0

        # 避免重复触发（某些 Trainer 实现可能多次调用）
        if current_epoch == self._current_epoch:
            return
        self._current_epoch = current_epoch

        if self.lans_sampler is not None:
            self.lans_sampler.set_epoch(current_epoch)

        current_comp = self.lans_scheduler.competence
        current_threshold = self.lans_scheduler.current_threshold

        logger.info(
            f"📅 Epoch {current_epoch} 开始: "
            f"能力值={current_comp:.4f}, 阈值={current_threshold:.2f}"
        )

        if current_epoch < self.refresh_start_epoch:
            logger.info(
                f"⏭️ Epoch {current_epoch}: 跳过负样本刷新 "
                f"(refresh_start_epoch={self.refresh_start_epoch})，沿用初始负样本"
            )
            return

        # 【核心】动态重新生成负样本
        if self.trainer_ref is not None and self.base_samples is not None and self.lans_sampler is not None:
            logger.info(f"🔄 Epoch {current_epoch}: 基于当前能力值重新生成负样本...")
            refresh_start_ts = time.perf_counter()

            # 【修复】清空滑动窗口，确保统计仅反映当前 Epoch 的策略分布
            if hasattr(self.lans_scheduler, '_recent_strategies'):
                self.lans_scheduler._recent_strategies = []

            # 重新生成所有样本的 rejected
            new_prompts, new_chosens, new_rejecteds = [], [], []
            new_texts, new_event_types = [], []

            total_samples = len(self.base_samples)
            last_log_ts = refresh_start_ts
            for idx, sample in enumerate(self.base_samples, start=1):
                result = self.lans_sampler.generate_rejected(sample)
                new_prompts.append(result["prompt"])
                new_chosens.append(result["chosen"])
                new_rejecteds.append(result["rejected"])
                new_texts.append(sample.get("text", ""))
                new_event_types.append(sample.get("event_types", []))

                now = time.perf_counter()
                should_log = (
                    idx % self.refresh_log_interval == 0
                    or idx == total_samples
                    or (now - last_log_ts) >= self.refresh_log_seconds
                )
                if should_log:
                    elapsed = max(time.perf_counter() - refresh_start_ts, 1e-6)
                    speed = idx / elapsed
                    eta = max(total_samples - idx, 0) / max(speed, 1e-6)
                    logger.info(
                        f"   ⏳ Epoch {current_epoch} 负样本刷新进度: "
                        f"{idx}/{total_samples} ({idx / max(total_samples, 1):.1%}), "
                        f"{speed:.2f} samples/s, ETA {eta:.1f}s"
                    )
                    last_log_ts = now

            # 创建新数据集
            new_dataset = Dataset.from_dict({
                "prompt": new_prompts,
                "chosen": new_chosens,
                "rejected": new_rejecteds,
                "text": new_texts,
                "event_types": new_event_types
            })

            # 更新 Trainer 的数据集
            self.trainer_ref.train_dataset = new_dataset

            # 重建 DataLoader（关键步骤）
            if hasattr(self.trainer_ref, '_train_dataloader'):
                self.trainer_ref._train_dataloader = None
            if hasattr(self.trainer_ref, 'accelerator') and self.trainer_ref.accelerator is not None:
                # 对于使用 accelerator 的情况，需要重新准备数据集
                pass  # accelerator 会在下次迭代时自动处理
            if not self._refresh_warning_emitted:
                logger.warning(
                    "⚠️ 动态刷新依赖 trainer.train_dataset/_train_dataloader 的内部行为；"
                    "升级 transformers/trl 后请做回归验证。"
                )
                self._refresh_warning_emitted = True

            # 【修复】重新生成后再获取统计，确保反映当前 Epoch 的策略分布
            new_stats = self.lans_scheduler.get_statistics()
            refresh_elapsed = time.perf_counter() - refresh_start_ts
            logger.info(
                f"✅ Epoch {current_epoch}: 已重新生成 {len(new_dataset)} 条负样本 "
                f"(耗时 {refresh_elapsed:.1f}s, 本轮策略分布: {new_stats['strategy_distribution']})"
            )
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """每个 Step 结束时更新 LANS 能力值"""
        # [修复 T6] 更新 LANS 采样器的训练进度
        if self.lans_sampler is not None:
            total_steps = state.max_steps if state.max_steps > 0 else 1
            self.lans_sampler.set_training_progress(state.global_step, total_steps)

        if state.log_history:
            latest_log = state.log_history[-1]
            loss = latest_log.get("loss") or latest_log.get("train/loss") or latest_log.get("train_loss")
            
            if loss is not None:
                self._global_step += 1
                new_competence = self.lans_scheduler.update_competence(float(loss))
                
                if self._writer is not None:
                    stats = self.lans_scheduler.get_statistics()
                    self._writer.add_scalar("train/loss", float(loss), self._global_step)
                    self._writer.add_scalar("lans/competence", stats["competence"], self._global_step)
                    self._writer.add_scalar("lans/threshold", stats["threshold"], self._global_step)
                    self._writer.add_scalar("lans/recent_avg_loss", stats["recent_avg_loss"], self._global_step)
                    dist = stats["strategy_distribution"]
                    self._writer.add_scalar("lans/strategy_easy", dist.get("EASY", 0.0), self._global_step)
                    self._writer.add_scalar("lans/strategy_medium", dist.get("MEDIUM", 0.0), self._global_step)
                    self._writer.add_scalar("lans/strategy_hard", dist.get("HARD", 0.0), self._global_step)
                    if self.trainer_ref is not None and hasattr(self.trainer_ref, "get_aux_metrics_snapshot"):
                        aux_metrics = self.trainer_ref.get_aux_metrics_snapshot()
                        if aux_metrics:
                            self._writer.add_scalar(
                                "train/pref_loss_weighted",
                                float(aux_metrics.get("pref_loss_weighted", 0.0)),
                                self._global_step,
                            )
                            self._writer.add_scalar(
                                "train/rpo_weight",
                                float(aux_metrics.get("rpo_weight", 0.0)),
                                self._global_step,
                            )
                            self._writer.add_scalar(
                                "train/rpo_sft_loss",
                                float(aux_metrics.get("rpo_sft_loss", 0.0)),
                                self._global_step,
                            )
                            self._writer.add_scalar(
                                "train/loss_combined",
                                float(aux_metrics.get("combined_loss", 0.0)),
                                self._global_step,
                            )

                if self._global_step % 50 == 0:
                    stats = self.lans_scheduler.get_statistics()
                    aux_info = ""
                    if self.trainer_ref is not None and hasattr(self.trainer_ref, "get_aux_metrics_snapshot"):
                        aux_metrics = self.trainer_ref.get_aux_metrics_snapshot()
                        if aux_metrics:
                            aux_info = (
                                f", RPO_w={aux_metrics.get('rpo_weight', 0.0):.4f}, "
                                f"SFT={aux_metrics.get('rpo_sft_loss', 0.0):.4f}"
                            )
                    logger.debug(
                        f"LANS [Step {self._global_step}]: "
                        f"Loss={loss:.4f}, C={new_competence:.4f}, "
                        f"策略分布={stats['strategy_distribution']}{aux_info}"
                    )
    
    def on_train_end(
        self, 
        args, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """训练结束时保存历史"""
        logger.info("🔄 on_train_end 回调开始...")
        
        # TensorBoard writer 关闭（添加超时保护）
        if self._writer is not None:
            try:
                logger.info("  📊 正在关闭 TensorBoard writer...")
                self._writer.flush()
                self._writer.close()
                self._writer = None  # 防止重复关闭
                logger.info("  ✅ TensorBoard writer 已关闭")
            except Exception as e:
                logger.warning(f"  ⚠️ TensorBoard writer 关闭失败: {e}")

        # 保存 LANS 历史
        if self.lans_scheduler:
            try:
                logger.info("  📝 正在导出 LANS 历史...")
                history = self.lans_scheduler.export_history()
                output_path = os.path.join(args.output_dir, "lans_history.json")
                
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                logger.info(f"  ✅ LANS 训练历史已保存: {output_path}")
            except Exception as e:
                logger.warning(f"  ❌ 保存 LANS 历史失败: {e}")
        
        # 导出 LANS 生成的负样本和 SCV 过滤样本
        if self.lans_sampler:
            try:
                logger.info("  📦 正在导出 LANS 负样本...")
                export_result = self.lans_sampler.export_samples()
                if export_result:
                    for key, path in export_result.items():
                        logger.info(f"     ✅ {key}: {path}")
                
                # 输出统计摘要
                stats = self.lans_sampler.get_statistics()
                logger.info(f"  📊 LANS 采样统计:")
                logger.info(f"     生成总数: {stats['total_generated']}")
                logger.info(f"     SCV 过滤: {stats['scv_filtered_count']} ({stats['scv_filter_rate']:.2%})")
                if "scv_cache_hit_rate" in stats:
                    logger.info(
                        f"     SCV 缓存: hits={stats.get('scv_cache_hits', 0)}, "
                        f"misses={stats.get('scv_cache_misses', 0)}, "
                        f"hit_rate={stats.get('scv_cache_hit_rate', 0.0):.2%}"
                    )
            except Exception as e:
                logger.warning(f"  ❌ 导出 LANS 样本失败: {e}")
        
        logger.info("🔄 on_train_end 回调完成")


# ============================================================================
# LANS 在线负采样生成器
# ============================================================================

class LANSNegativeSampler:
    """
    LANS 负样本生成器
    支持负样本导出和 SCV 过滤样本记录
    """
    
    def __init__(
        self,
        ds_cns: DSCNSampler,
        scv: Optional[SemanticConsistencyVerifier] = None,
        export_dir: Optional[str] = None,  # 导出目录
        scv_cache_enabled: bool = True,
        scv_cache_max_entries: int = 50000,
        scv_max_retries: int = 1,
    ):
        self.ds_cns = ds_cns
        self.scv = scv
        self.epoch = 0
        self.export_dir = export_dir
        self.scv_cache_enabled = bool(scv_cache_enabled)
        self.scv_cache_max_entries = max(1000, int(scv_cache_max_entries))
        self.scv_max_retries = max(0, int(scv_max_retries))
        
        # 记录生成的负样本和 SCV 过滤样本
        self._generated_samples: List[Dict] = []
        self._scv_filtered_samples: List[Dict] = []
        self._sample_counter = 0
        # [修复 T6] 添加训练进度追踪
        self._current_step = 0
        self._total_steps = 1
        self._scv_cache: "OrderedDict[str, bool]" = OrderedDict()
        self._scv_cache_hits = 0
        self._scv_cache_misses = 0

    @staticmethod
    def _stable_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def _get_scv_cache_key(self, text: str, neg_json: str) -> str:
        return self._stable_hash(f"{text}\n<SEP>\n{neg_json}")

    def _is_false_negative(self, text: str, neg_json: str) -> bool:
        if self.scv is None:
            return False

        if not self.scv_cache_enabled:
            return bool(self.scv.is_false_negative(text, neg_json))

        key = self._get_scv_cache_key(text, neg_json)
        cached = self._scv_cache.get(key)
        if cached is not None:
            self._scv_cache_hits += 1
            self._scv_cache.move_to_end(key)
            return bool(cached)

        self._scv_cache_misses += 1
        result = bool(self.scv.is_false_negative(text, neg_json))
        self._scv_cache[key] = result
        self._scv_cache.move_to_end(key)

        if len(self._scv_cache) > self.scv_cache_max_entries:
            self._scv_cache.popitem(last=False)
        return result

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_training_progress(self, current_step: int, total_steps: int):
        """设置当前训练进度，用于 LANS 自适应阈值计算"""
        self._current_step = current_step
        self._total_steps = max(1, total_steps)

    def generate_rejected(self, example: Dict) -> Dict:
        chosen = example["chosen"]
        text = example.get("text", "")
        event_types = example.get("event_types") or []
        prompt = example.get("prompt", "")

        if self.ds_cns._use_lans:
            strategy = self.ds_cns.get_negative_strategy_adaptive()
        else:
            strategy = "MEDIUM"

        strategy_fallback = {"HARD": "MEDIUM", "MEDIUM": "EASY", "EASY": "EASY"}
        scv_filtered = False
        scv_retry_count = 0
        neg_json = ""
        current_strategy = strategy

        for attempt in range(self.scv_max_retries + 1):
            # [修复 T6] 传递实际的训练步数，使 LANS 阈值生效
            neg_json = self.ds_cns.generate_negative_json(
                chosen, current_strategy, self._current_step, self._total_steps
            )

            if self.scv is None:
                break

            try:
                is_false_neg = self._is_false_negative(text, neg_json)
            except Exception:
                is_false_neg = False

            if not is_false_neg:
                break

            scv_filtered = True
            scv_retry_count += 1
            self._scv_filtered_samples.append({
                "sample_id": self._sample_counter,
                "text_preview": text[:200] if text else "",
                "chosen": chosen[:500] if chosen else "",
                "filtered_rejected": neg_json[:500] if neg_json else "",
                "strategy": current_strategy,
                "attempt": attempt,
                "reason": "SCV detected false negative"
            })

            if attempt >= self.scv_max_retries:
                break

            current_strategy = strategy_fallback.get(current_strategy, "EASY")

        strategy = current_strategy

        rejected_cot = ChinesePromptBuilder.build_incorrect_cot_response(
            neg_json, strategy, original_types=event_types
        )
        
        # 记录生成的负样本
        self._generated_samples.append({
            "sample_id": self._sample_counter,
            "prompt_preview": prompt[:200] if prompt else "",
            "chosen_preview": chosen[:300] if chosen else "",
            "rejected_preview": rejected_cot[:300] if rejected_cot else "",
            "strategy": strategy,
            "scv_filtered": scv_filtered,
            "scv_retry_count": scv_retry_count,
            "epoch": self.epoch
        })
        self._sample_counter += 1
        
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": rejected_cot
        }
    
    def export_samples(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        导出生成的负样本和 SCV 过滤样本
        
        Args:
            output_dir: 输出目录，默认使用初始化时的 export_dir
        
        Returns:
            导出文件路径字典
        """
        export_path = output_dir or self.export_dir
        if not export_path:
            logger.warning("未指定导出目录，跳过样本导出")
            return {}
        
        os.makedirs(export_path, exist_ok=True)
        result = {}
        
        # 导出 LANS 生成的负样本
        if self._generated_samples:
            neg_samples_file = os.path.join(export_path, "lans_generated_samples.jsonl")
            with open(neg_samples_file, 'w', encoding='utf-8') as f:
                for sample in self._generated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            result["generated_samples"] = neg_samples_file
            logger.info(f"📦 导出 LANS 负样本: {len(self._generated_samples)} 条 -> {neg_samples_file}")
        
        # 导出 SCV 过滤样本
        if self._scv_filtered_samples:
            scv_filtered_file = os.path.join(export_path, "scv_filtered_samples.jsonl")
            with open(scv_filtered_file, 'w', encoding='utf-8') as f:
                for sample in self._scv_filtered_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            result["scv_filtered"] = scv_filtered_file
            logger.info(f"🔍 导出 SCV 过滤样本: {len(self._scv_filtered_samples)} 条 -> {scv_filtered_file}")
        
        # 导出统计摘要
        summary = {
            "total_generated": len(self._generated_samples),
            "scv_filtered_count": len(self._scv_filtered_samples),
            "scv_filter_rate": len(self._scv_filtered_samples) / max(1, len(self._generated_samples)),
            "scv_cache": {
                "enabled": self.scv_cache_enabled,
                "max_entries": self.scv_cache_max_entries,
                "size": len(self._scv_cache),
                "hits": self._scv_cache_hits,
                "misses": self._scv_cache_misses,
                "hit_rate": self._scv_cache_hits / max(1, self._scv_cache_hits + self._scv_cache_misses),
            },
            "strategy_distribution": {}
        }
        
        # 统计策略分布
        for sample in self._generated_samples:
            strategy = sample.get("strategy", "UNKNOWN")
            summary["strategy_distribution"][strategy] = summary["strategy_distribution"].get(strategy, 0) + 1
        
        summary_file = os.path.join(export_path, "lans_sampling_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        result["summary"] = summary_file
        
        return result
    
    def get_statistics(self) -> Dict:
        """获取采样统计信息"""
        return {
            "total_generated": len(self._generated_samples),
            "scv_filtered_count": len(self._scv_filtered_samples),
            "scv_filter_rate": len(self._scv_filtered_samples) / max(1, len(self._generated_samples)),
            "scv_cache_hits": self._scv_cache_hits,
            "scv_cache_misses": self._scv_cache_misses,
            "scv_cache_hit_rate": self._scv_cache_hits / max(1, self._scv_cache_hits + self._scv_cache_misses),
        }


class LANSIterableDataset(IterableDataset):
    """
    在线 LANS 负样本数据集（真正的 Online Adaptive 采样）
    每次迭代都会读取最新的 competence 并动态生成 rejected。

    ⚠️ 警告：此类不支持 num_workers > 0，因为 Worker 进程无法同步主进程的能力值更新。
    请确保 DPOConfig 中设置 dataloader_num_workers=0。
    """

    def __init__(self, base_samples: List[Dict], lans_sampler: LANSNegativeSampler, seed: int = 3407):
        import warnings
        warnings.warn(
            "LANSIterableDataset 仅支持 dataloader_num_workers=0。"
            "多进程模式下 LANS 能力值无法同步，动态采样将失效。",
            UserWarning
        )
        self.base_samples = base_samples
        self.lans_sampler = lans_sampler
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.base_samples)))
        rng.shuffle(indices)

        for idx in indices:
            item = self.base_samples[idx]
            yield self.lans_sampler.generate_rejected(item)


class LANSDataCollator:
    """
    [Legacy] LANS 动态负采样 Collator（默认训练路径未启用）

    历史实现：在 DataLoader 批次生成时动态调用 LANS 采样器生成负样本。
    当前默认主路径采用“初始全量 + 每 Epoch 刷新”的训练机制，不启用 per-batch collator。
    该类仅保留用于对照实验与向后兼容。
    """

    def __init__(self, tokenizer, lans_sampler: LANSNegativeSampler, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.lans_sampler = lans_sampler
        self.max_length = max_length
        # 使用 DPODataCollatorWithPadding 处理最终的 Padding
        self.base_collator = DPODataCollatorWithPadding(tokenizer, max_length=max_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理一个 Batch 的数据：
        1. 提取原始信息 (chosen, prompt, etc)
        2. 调用 lans_sampler 生成动态 rejected
        3. Tokenize 新的 rejected
        4. 构造 labels (mask prompt)
        5. 调用 base_collator 进行 Padding
        """
        # 预先计算 pad_token_id (如果 base_collator 需要，虽然后面调用了它)

        for feature in features:
            # 确保有必要的字段 (DPOTrainer remove_unused_columns=False 应保留这些)
            if "prompt" not in feature or "chosen" not in feature:
                continue

            # 1. 构造采样所需的样本字典
            sample = {
                "chosen": feature["chosen"],
                "text": feature.get("text", ""),
                "event_types": feature.get("event_types", []),
                "prompt": feature.get("prompt", "")
            }

            # 2. 动态生成 Negative (基于当前 LANS Competence)
            # 注意：generate_rejected 内部会读取 lans_scheduler 的最新状态
            result = self.lans_sampler.generate_rejected(sample)
            rejected_content = result["rejected"]

            # 3. Tokenize Prompt + Rejected
            # prompt 已经包含 Chat Template 格式
            prompt = feature["prompt"]
            full_rejected_text = prompt + rejected_content + self.tokenizer.eos_token

            tokenized_rejected = self.tokenizer(
                full_rejected_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False  # Prompt 已含特殊 token
            )

            rejected_input_ids = tokenized_rejected["input_ids"]
            rejected_attention_mask = tokenized_rejected["attention_mask"]

            # 4. 构造 Rejected Labels (Mask Prompt 部分)
            # 为了准确 Mask，我们需要知道 Prompt 的长度
            # Tokenize Prompt 单独获取长度
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False
            )["input_ids"]
            prompt_len = len(prompt_tokens)

            rejected_labels = list(rejected_input_ids)
            # 将 Prompt 部分设为 -100 (Ignore Index)
            for i in range(min(prompt_len, len(rejected_labels))):
                rejected_labels[i] = -100

            # 5. 更新 feature (覆盖 DPOTrainer 预处理的 dummy rejected)
            feature["rejected_input_ids"] = rejected_input_ids
            feature["rejected_attention_mask"] = rejected_attention_mask
            feature["rejected_labels"] = rejected_labels

            # chosen_input_ids 等字段保持不变 (由 DPOTrainer 预处理好)

        # 6. 交给 Base Collator 进行 Padding 和 Tensor 转换
        return self.base_collator(features)


class UnslothDPOTrainerWrapper:
    def __init__(self, config: dict, data_samples: list):
        init_start_ts = time.perf_counter()
        self.config = config
        self.samples = data_samples
        self.runtime_stats: Dict[str, Any] = {
            "phase_timings_seconds": {},
        }
        
        ds_cns_cfg = config['algorithms']['ds_cns']
        schema_path = ds_cns_cfg['taxonomy_path']
        static_c0 = ds_cns_cfg.get('static_mode_c0', 0.1)
        use_ontology_distance = ds_cns_cfg.get('use_ontology_distance', True)  # 【消融实验 A6】
        ds_cns_start_ts = time.perf_counter()
        self.ds_cns = DSCNSampler(
            schema_path, 
            c0=static_c0,
            use_ontology_distance=use_ontology_distance
        )
        self.runtime_stats["phase_timings_seconds"]["ds_cns_init"] = round(
            time.perf_counter() - ds_cns_start_ts, 4
        )
        
        self.scv_cfg = config.get('algorithms', {}).get('scv', {})
        self.scv = None
        if config['algorithms']['scv']['enabled']:
            scv_start_ts = time.perf_counter()
            self.scv = SemanticConsistencyVerifier(
                self.scv_cfg['nli_model'],
                self.scv_cfg['nli_threshold'],
                progress_log_interval=self.scv_cfg.get('progress_log_interval', 200),
                progress_log_seconds=self.scv_cfg.get('progress_log_seconds', 30),
            )
            self.runtime_stats["phase_timings_seconds"]["scv_init"] = round(
                time.perf_counter() - scv_start_ts, 4
            )
        else:
            self.runtime_stats["phase_timings_seconds"]["scv_init"] = 0.0
        
        self.lans_scheduler = None
        self.lans_callback = None
        self.lans_sampler = None
        self.runtime_stats["phase_timings_seconds"]["trainer_init"] = round(
            time.perf_counter() - init_start_ts, 4
        )

    def load_model(self) -> None:
        load_start_ts = time.perf_counter()
        m_cfg = self.config['model']
        l_cfg = self.config['lora']
        model_name_or_path = m_cfg['base_model']
        
        if m_cfg.get('source', 'huggingface') == 'modelscope':
            try:
                from modelscope import snapshot_download
                model_name_or_path = snapshot_download(model_name_or_path, cache_dir='./models')
            except Exception:
                pass
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_or_path,
            max_seq_length = m_cfg['max_seq_length'],
            dtype = None,
            load_in_4bit = m_cfg['load_in_4bit'],
        )
        self.model.gradient_checkpointing_enable()
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = l_cfg['r'],
            target_modules = l_cfg['target_modules'],
            lora_alpha = l_cfg['lora_alpha'],
            lora_dropout = l_cfg['lora_dropout'],
            bias = l_cfg['bias'],
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # [修复] 显式检查 EOS token（Qwen3 通常使用 <|im_end|>）
        expected_eos = "<|im_end|>"
        if self.tokenizer.eos_token is None or self.tokenizer.eos_token == "":
            self.tokenizer.eos_token = expected_eos
        elif self.tokenizer.eos_token != expected_eos:
            logger.warning(
                f"EOS token 为 {self.tokenizer.eos_token}，期望 {expected_eos}。将保留当前设置。"
            )
        logger.info(
            f"EOS Token: {self.tokenizer.eos_token} | EOS Token ID: {self.tokenizer.eos_token_id}"
        )
        load_elapsed = time.perf_counter() - load_start_ts
        self.runtime_stats["phase_timings_seconds"]["model_load"] = round(load_elapsed, 4)
        logger.info(f"⏱️ 模型加载阶段耗时: {load_elapsed:.1f}s")

    def _apply_chat_template(self, raw_text: str) -> str:
        """[关键] 应用 Chat Template，统一训练与评估的 prompt 构建"""
        return build_inference_prompt(
            text=raw_text,
            tokenizer=self.tokenizer,
            use_oneshot=False
        )

    def prepare_dpo_dataset(self, use_lans: bool = True) -> Any:
        seed = self.config['project']['seed']
        random.seed(seed)
        np.random.seed(seed)
        
        verified_samples = []
        for s in self.samples:
            if all(t in self.ds_cns.graph for t in (s.event_types or [])):
                verified_samples.append(s)
        self.samples = verified_samples
        total_steps = len(self.samples)
        
        if use_lans:
            logger.info("🚀 启用 LANS (在线模式)")
            lans_cfg = self.config['algorithms'].get('lans', {})

            # 【修复】读取多粒度权重配置
            granularity_weights = lans_cfg.get('granularity_weights', None)
            loss_baseline = lans_cfg.get("loss_baseline")
            if loss_baseline is None and "lans_alpha" in lans_cfg:
                # 向后兼容旧配置键，避免历史实验配置直接失效
                loss_baseline = lans_cfg.get("lans_alpha")
                logger.warning(
                    "配置键 algorithms.lans.lans_alpha 已弃用，请改用 algorithms.lans.loss_baseline。"
                )
            if loss_baseline is None:
                loss_baseline = 0.5

            self.lans_scheduler = self.ds_cns.enable_lans(
                ema_decay=lans_cfg.get('ema_decay', 0.95),
                loss_baseline=loss_baseline,
                warmup_steps=lans_cfg.get('warmup_steps', 100),
                competence_floor=lans_cfg.get('competence_floor', 0.05),
                competence_ceiling=lans_cfg.get('competence_ceiling', 0.95),
                warmup_target=lans_cfg.get('warmup_target', 0.25),
                use_ema=lans_cfg.get('use_ema', True),
                cga_beta=lans_cfg.get('cga_beta', 0.1),
                use_cga=lans_cfg.get('use_cga', True),
                granularity_weights=granularity_weights,
                easy_ratio=lans_cfg.get('strategies', {}).get('easy_ratio', 0.7),
                hard_ratio=lans_cfg.get('strategies', {}).get('hard_ratio', 0.4),
                hard_floor_prob=lans_cfg.get('strategies', {}).get('hard_floor_prob', 0.0),
                hard_floor_after_warmup=lans_cfg.get('strategies', {}).get('hard_floor_after_warmup'),
                medium_floor_prob=lans_cfg.get('strategies', {}).get('medium_floor_prob', 0.0),
            )

            # 传递导出目录到 LANSNegativeSampler
            export_dir = self.config['project'].get('debug_data_dir')
            self.lans_sampler = LANSNegativeSampler(
                ds_cns=self.ds_cns,
                scv=self.scv,
                export_dir=export_dir,
                scv_cache_enabled=self.scv_cfg.get("cache_enabled", True),
                scv_cache_max_entries=self.scv_cfg.get("cache_max_entries", 50000),
                scv_max_retries=self.scv_cfg.get("max_retries", 1),
            )

            samples_data = []
            for sample in self.samples:
                formatted_prompt = self._apply_chat_template(sample.text) # 修复 Loss
                samples_data.append({
                    "prompt": formatted_prompt,
                    "chosen": sample.chosen,
                    "rejected": "",  # 【关键修改】提供空 Rejected，由 LANSDataCollator 动态生成
                    "text": sample.text,
                    "event_types": sample.event_types or []
                })
            return samples_data

        else:
            logger.info("📦 使用静态课程学习模式")
            cache_dir = os.path.join(self.config['project']['dataset_cache_dir'], "dpo_dataset_cache")
            
            # 优先从缓存加载
            if os.path.exists(cache_dir):
                logger.info(f"   📂 从缓存加载数据集: {cache_dir}")
                return Dataset.load_from_disk(cache_dir)
            
            logger.info("   🔧 首次运行，生成静态数据集...")
            prompts, chosens, rejecteds = [], [], []
            static_samples_log = []  # 记录静态样本用于导出
            
            for idx, sample in enumerate(self.samples):
                formatted_prompt = self._apply_chat_template(sample.text) # 修复 Loss
                strategy = self.ds_cns.get_negative_strategy(idx, total_steps)
                neg_json = self.ds_cns.generate_negative_json(sample.chosen, strategy, idx, total_steps)
                rejected_cot = ChinesePromptBuilder.build_incorrect_cot_response(
                    neg_json, strategy, sample.event_types
                )
                prompts.append(formatted_prompt)
                chosens.append(sample.chosen)
                rejecteds.append(rejected_cot)
                
                # 记录用于导出
                static_samples_log.append({
                    "sample_id": idx,
                    "strategy": strategy,
                    "prompt_preview": formatted_prompt[:200],
                    "chosen_preview": sample.chosen[:300],
                    "rejected_preview": rejected_cot[:300]
                })
            
            dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})
            
            # 保存缓存到磁盘
            os.makedirs(cache_dir, exist_ok=True)
            dataset.save_to_disk(cache_dir)
            logger.info(f"   💾 数据集已缓存: {cache_dir}")
            
            # 导出静态样本日志
            debug_dir = self.config['project'].get('debug_data_dir')
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                static_log_file = os.path.join(debug_dir, "static_dpo_samples.jsonl")
                with open(static_log_file, 'w', encoding='utf-8') as f:
                    for sample in static_samples_log:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                logger.info(f"   📦 静态样本日志已导出: {static_log_file}")
            
            return dataset

    def train(self, use_lans: bool = True) -> None:
        transformers.set_seed(self.config['project']['seed'])
        
        base_dataset = self.prepare_dpo_dataset(use_lans=use_lans)
        t_cfg = self.config['training']
        
        # 从模型配置获取 max_seq_length
        m_cfg = self.config['model']
        max_seq_len = m_cfg.get('max_seq_length', 4096)
        
        use_online_lans = bool(use_lans and self.lans_scheduler is not None)
        fast_io = bool(t_cfg.get('fast_io', False))
        dataloader_num_workers = int(t_cfg.get('dataloader_num_workers', 0))
        dataloader_pin_memory = bool(t_cfg.get('dataloader_pin_memory', False))
        gradient_checkpointing = bool(t_cfg.get('gradient_checkpointing', True))

        if fast_io and not use_online_lans and dataloader_num_workers <= 0:
            dataloader_num_workers = 2
        if fast_io:
            dataloader_pin_memory = True
        if use_online_lans and dataloader_num_workers != 0:
            logger.warning(
                "LANS 动态采样模式下强制 dataloader_num_workers=0，避免多进程状态不同步。"
            )
            dataloader_num_workers = 0

        dpo_config = DPOConfig(
            output_dir = self.config['project']['output_dir'],
            logging_dir = self.config['project']['logging_dir'],
            beta = t_cfg['beta'],
            loss_type = t_cfg['loss_type'],
            per_device_train_batch_size = t_cfg['per_device_train_batch_size'],
            gradient_accumulation_steps = t_cfg['gradient_accumulation_steps'],
            learning_rate = t_cfg['learning_rate'],
            lr_scheduler_type = t_cfg.get('lr_scheduler_type', 'cosine'),
            warmup_steps = t_cfg.get('warmup_steps', 0),
            num_train_epochs = t_cfg['num_train_epochs'],
            max_steps = t_cfg.get('max_steps', -1),
            logging_steps = t_cfg['logging_steps'],
            bf16 = t_cfg['bf16'],
            fp16 = False,  # 【修复】bf16 和 fp16 互斥，明确禁用 fp16
            optim = t_cfg['optim'],
            weight_decay = t_cfg.get('weight_decay', 0.0),
            max_grad_norm = t_cfg.get('max_grad_norm', 1.0),
            # 【关键修复】调整长度参数，与模型配置一致
            max_prompt_length = min(2048, max_seq_len // 2),  # prompt 占一半
            max_length = max_seq_len,  # 总长度与模型一致
            max_completion_length = max_seq_len // 2,  # 限制响应长度
            gradient_checkpointing = gradient_checkpointing,
            report_to = ["tensorboard"],
            remove_unused_columns = False,
            # 【修复】正确处理保存策略
            save_strategy = t_cfg.get('save_strategy', 'no'),  # 默认不自动保存
            save_steps = t_cfg.get('save_steps', 500) if t_cfg.get('save_strategy', 'no') != 'no' else 500,
            save_total_limit = t_cfg.get('save_total_limit', 2),
            # 【修复】现在使用普通 Dataset，禁用 precompute 以防止 OOM
            precompute_ref_log_probs = False,
            # 【修复】显式设置 dataloader 参数
            dataloader_num_workers = dataloader_num_workers,
            dataloader_pin_memory = dataloader_pin_memory,
        )
        
        save_info = f"Save={dpo_config.save_strategy}" if dpo_config.save_strategy != 'no' else "Save=manual"
        print(
            f"\n🚀 Training Config: Steps={dpo_config.max_steps}, {save_info}, "
            f"GC={dpo_config.gradient_checkpointing}, workers={dpo_config.dataloader_num_workers}, "
            f"pin_memory={dpo_config.dataloader_pin_memory}"
        )
        if dpo_config.gradient_checkpointing:
            logger.info("ℹ️ gradient_checkpointing 已启用：更省显存，但训练吞吐通常会下降。")

        # RPO mixed objective: loss = preference_loss + alpha * SFT(chosen)
        rpo_cfg = t_cfg.get("rpo", {})
        rpo_alpha = float(rpo_cfg.get("alpha", t_cfg.get("rpo_alpha", 0.0)))
        rpo_warmup_steps = int(rpo_cfg.get("warmup_steps", t_cfg.get("rpo_warmup_steps", 0)))
        aux_log_interval = int(rpo_cfg.get("log_interval", t_cfg.get("aux_log_interval", 50)))
        if rpo_alpha > 0:
            logger.info(
                f"🎯 训练目标: Preference + RPO(SFT), alpha={rpo_alpha:.4f}, "
                f"warmup_steps={rpo_warmup_steps}"
            )
        
        callbacks = []
        # 默认采用 Epoch 级刷新，不走 per-batch DataCollator 动态采样
        data_collator = None
        lans_cfg = self.config.get('algorithms', {}).get('lans', {})

        if use_online_lans:
            logger.info("🔄 启用 LANS 自适应采样模式 (Epoch 级别动态刷新)")
            logger.info("ℹ️ 当前不启用 LANSDataCollator，负样本由初始化阶段与每 Epoch 刷新生成")

            # 【重要】先使用初始能力值生成第一批负样本
            logger.info("📦 生成初始负样本 (Epoch 0)...")
            prompts, chosens, rejecteds = [], [], []
            texts, event_types_list = [], []

            total_samples = len(base_dataset)
            refresh_log_interval = max(1, int(lans_cfg.get("refresh_log_interval", 100)))
            refresh_log_seconds = max(5.0, float(lans_cfg.get("refresh_log_seconds", 30)))
            init_start_ts = time.perf_counter()
            last_log_ts = init_start_ts
            for idx, sample in enumerate(base_dataset):
                current = idx + 1
                now = time.perf_counter()
                should_log = (
                    current % refresh_log_interval == 0
                    or current == total_samples
                    or (now - last_log_ts) >= refresh_log_seconds
                )
                if should_log:
                    elapsed = max(now - init_start_ts, 1e-6)
                    speed = current / elapsed
                    eta = max(total_samples - current, 0) / max(speed, 1e-6)
                    logger.info(
                        f"   ⏳ 初始负样本进度: {current}/{total_samples} "
                        f"({current / max(total_samples, 1):.1%}), {speed:.2f} samples/s, ETA {eta:.1f}s"
                    )
                    last_log_ts = now

                result = self.lans_sampler.generate_rejected(sample)
                prompts.append(result["prompt"])
                chosens.append(result["chosen"])
                rejecteds.append(result["rejected"])
                texts.append(sample.get("text", ""))
                event_types_list.append(sample.get("event_types", []))

            dataset = Dataset.from_dict({
                "prompt": prompts,
                "chosen": chosens,
                "rejected": rejecteds,
                "text": texts,
                "event_types": event_types_list
            })

            init_elapsed = time.perf_counter() - init_start_ts
            init_speed = len(dataset) / max(init_elapsed, 1e-6)
            logger.info(
                f"✅ 生成 {len(dataset)} 条初始负样本 "
                f"(耗时 {init_elapsed:.1f}s, 吞吐 {init_speed:.2f} samples/s)"
            )
            self.runtime_stats["phase_timings_seconds"]["initial_negative_generation"] = round(
                init_elapsed, 4
            )
            if self.scv is not None:
                self.runtime_stats["scv_runtime"] = {
                    "calls": int(getattr(self.scv, "_calls", 0)),
                    "total_windows": int(getattr(self.scv, "_total_windows", 0)),
                    "total_time_seconds": round(float(getattr(self.scv, "_total_time_seconds", 0.0)), 4),
                }
                logger.info(
                    "🔎 SCV 阶段汇总: "
                    f"calls={self.runtime_stats['scv_runtime']['calls']}, "
                    f"windows={self.runtime_stats['scv_runtime']['total_windows']}, "
                    f"time={self.runtime_stats['scv_runtime']['total_time_seconds']:.1f}s"
                )

            # 导出初始样本用于调试
            if self.lans_sampler:
                self.lans_sampler.export_samples()

        else:
            dataset = base_dataset
            self.runtime_stats["phase_timings_seconds"]["initial_negative_generation"] = 0.0

        # 【关键修复】使用 CGADPOTrainer 替代标准 DPOTrainer，启用对比梯度放大
        # [修复 T7] 使用 processing_class 替代 tokenizer（TRL >= 0.9.0）
        trainer = CGADPOTrainer(
            model = self.model,
            ref_model = None,
            processing_class = self.tokenizer,
            train_dataset = dataset,
            data_collator = data_collator,
            args = dpo_config,
            callbacks = [],  # 先传空，后面添加
            lans_scheduler = self.lans_scheduler if use_online_lans else None,
            rpo_alpha = rpo_alpha,
            rpo_warmup_steps = rpo_warmup_steps,
            aux_log_interval = aux_log_interval,
        )

        # 【核心】创建 LANS Callback 并传递 trainer 引用
        if use_online_lans:
            self.lans_callback = LANSCallback(
                self.lans_scheduler,
                lans_sampler=self.lans_sampler,
                lans_dataset=None,
                logging_dir=dpo_config.logging_dir,
                trainer_ref=trainer,  # 【新增】传递 Trainer 引用
                base_samples=base_dataset,  # 【新增】传递原始样本
                tokenizer=self.tokenizer,  # 【新增】传递 Tokenizer
                refresh_start_epoch=lans_cfg.get("refresh_start_epoch", 1),
                refresh_log_interval=lans_cfg.get("refresh_log_interval", 100),
                refresh_log_seconds=lans_cfg.get("refresh_log_seconds", 30),
            )
            trainer.add_callback(self.lans_callback)
        
        logger.info(f"Starting Training...")
        train_loop_start_ts = time.perf_counter()
        try:
            trainer.train()
            logger.info("✅ trainer.train() 完成")
            self.runtime_stats["phase_timings_seconds"]["train_loop"] = round(
                time.perf_counter() - train_loop_start_ts, 4
            )

        except Exception as e:
            logger.error(f"Training interrupted: {e}")
            raise e
        
        # [优化] 显式清理缓存，防止保存时 OOM 或卡顿
        logger.info("🧹 Training finished. Cleaning up memory before saving...")
        import gc
        gc.collect()
        
        # CUDA 同步，确保所有操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("  ✅ 内存清理完成")

        logger.info(f"💾 Saving model to {dpo_config.output_dir}...")
        save_start_ts = time.perf_counter()
        try:
            # 确保目录存在
            os.makedirs(dpo_config.output_dir, exist_ok=True)
            
            self.model.save_pretrained(dpo_config.output_dir)
            logger.info("  ✅ 模型权重已保存")
            
            self.tokenizer.save_pretrained(dpo_config.output_dir)
            logger.info("  ✅ Tokenizer 已保存")
            
        except Exception as e:
            logger.error(f"❌ 保存模型失败: {e}")
            raise e
            
        self.runtime_stats["phase_timings_seconds"]["save_artifacts"] = round(
            time.perf_counter() - save_start_ts, 4
        )
        timings = self.runtime_stats.get("phase_timings_seconds", {})
        logger.info(
            "⏱️ 训练阶段耗时汇总(s): "
            + ", ".join([f"{k}={v}" for k, v in timings.items()])
        )
        logger.info("✅ Model saved successfully.")

    def get_runtime_stats(self) -> Dict[str, Any]:
        return dict(self.runtime_stats)
