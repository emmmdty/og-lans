# DuEE-Fin `0.75+` Hybrid 方案设计

## 目标

本方案采用双目标：

1. 在 DuEE-Fin 的 legacy comparable track 上，把本地 4B 体系做到 `legacy_dueefin_overall_f1 >= 0.75`
2. 继续保留 record-aware 轨道，用于衡量真正的文档级记录组装能力：
   - `doc_role_micro_f1`
   - `doc_instance_micro_f1`
   - `doc_combination_micro_f1`
   - `doc_event_type_micro_f1`

这里的关键判断是：文献里 DuEE-Fin `0.75+` 的成绩多数对应 legacy overall F1，而不是当前仓库更严格的 record-aware 指标。两条轨道必须同时保留，不能混写。

## 设计判断

当前纯生成式 `Qwen3-4B + OG-LANS + record_corrector+cat_lite` 可以带来稳定增益，但离 API baseline 和 `0.75+` legacy 目标还有明显距离。问题不在于评估体系错误，而在于当前主线更擅长：

- 降低幻觉
- 提高 schema 合法性
- 修正局部角色错误

它还不擅长：

- 稳定拆分多主体事件
- 补齐高频缺失 role
- 直接生成完整 record 集合

因此正式方案不再把“纯生成式偏好训练”当成唯一主线，而是切到混合范式。

## 正式方案

### A. 双轨评估

- `legacy_dueefin_overall_f1`
- `doc_role_micro_f1`
- `doc_instance_micro_f1`
- `doc_combination_micro_f1`
- `doc_event_type_micro_f1`

所有结果必须同时报告两条轨道，legacy 用于对齐旧式 DuEE-Fin 文献，record-aware 用于解释真实记录级能力。

### B. 主模型路线

近期不直接替换 backbone，也不再优先扩展 `two_stage`。主线分三层：

1. `single_pass` 抽取器
2. `record_corrector_v2`
3. `cat_lite`

这是当前最稳的共享推理骨架，适用于：

- API baseline
- 本地 Qwen3-4B base-only
- 训练后 checkpoint

### C. Teacher silver

Teacher silver 用于补足 4B 模型在高质量结构化 supervision 上的不足，但不进入正式主表。

来源：

1. 最强 API baseline
2. 最强本地 checkpoint + shared postprocess

保留条件：

- parse_success
- 事件类型集合一致
- role overlap 超过固定阈值
- 来源文本必须可恢复，不能只剩截断 preview

teacher silver 仅用于训练增强，不改变最终评估协议。

### D. 下一阶段真正要做的结构化主线

本仓库下一阶段最值得新增的不是更长 prompt，而是结构化 extractor：

1. event-specific probes
2. argument library / role prototype
3. set-based record decoding
4. 最后再接 `record_corrector_v2 -> cat_lite`

这条结构化主线的目标是直接打 record assembly，而不是继续依赖 JSON 自回归把完整记录“写出来”。

## 本轮已落地能力

本轮已经实现：

- legacy DuEE-Fin 可比指标轨道
- teacher silver JSONL builder
- teacher silver 训练入口接线
- teacher silver provenance 写入 run manifest

本轮没有硬做：

- 结构化 extractor 本体
- event probes / argument library 训练代码
- set decoder / matching loss

这些内容保留为下一阶段正式实现范围。

## 实验纪律

- `train_fit`：训练与 few-shot exemplar pool
- `train_tune`：方法选择
- `dev_final`：冻结后最终报告

teacher silver 若来自 train split 的评测输出，必须继续受 `train_fit` 过滤，不允许把 tune/dev 的 gold 信息混回训练。

## 成功标准

### 近期

- `record_corrector_v2` 进一步提高 `doc_instance_micro_f1`
- teacher silver 能带来 checkpoint 的净增益
- legacy 轨道结果进入正式 summary / suite

### 中期

- 本地 4B 体系在 legacy 轨道上逼近 `0.75`
- record-aware 轨道继续提升，而不是只提高 legacy

## 不该做的事

- 不要再把 `two_stage` 当作当前 4B 主线
- 不要继续用更长 few-shot prompt 压小模型
- 不要把 API teacher 直接当正式主表模型
- 不要把 legacy `0.75` 强行要求到 `doc_role_micro_f1`
