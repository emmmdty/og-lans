# OG-LANS 代码修改任务分解与检查机制

**用途**: 将修改拆分为可独立完成的小任务，每个任务包含验收检查，确保 AI 能完整修复所有问题。

**执行原则**
1. 严格按任务顺序执行，完成一个任务再进行下一个。
2. 每个任务必须通过“检查机制”后才能标记完成。
3. 不要引入未在任务中声明的改动。
4. 所有修改应保持 Python 3.10 兼容。

---

## 全局完成检查（总开关）

完成所有任务后，逐项确认：
- [ ] T1 已完成并通过检查
- [ ] T2 已完成并通过检查
- [ ] T3 已完成并通过检查
- [ ] T4 已完成并通过检查
- [ ] T5 已完成并通过检查
- [ ] T6 已完成并通过检查
- [ ] T7 已完成并通过检查
- [ ] T8 已完成并通过检查
- [ ] T9 已完成并通过检查
- [ ] T10 已完成并通过检查

---

## T1. 修复 Unsloth DPO patch 顺序

**目标**: 确保 `PatchDPOTrainer()` 在任何 `trl` 导入之前执行。

**涉及文件**: `src/oglans/trainer/unsloth_trainer.py`

**操作步骤**
1. 移动 `PatchDPOTrainer()` 到 `from trl import DPOTrainer, DPOConfig` 之前。
2. 保持其他逻辑不变。

**检查机制**
- 手动检查或 `rg -n "PatchDPOTrainer\(\)|from trl" src/oglans/trainer/unsloth_trainer.py`，确保 `PatchDPOTrainer()` 出现在所有 `trl` 导入之前。
- 若顺序正确，勾选：
  - [ ] T1 通过

---

## T2. 评估默认使用确定性解码

**目标**: 评估默认 greedy，采样式评估需显式开关。

**涉及文件**: `evaluate.py`

**操作步骤**
1. 新增 CLI 参数 `--do_sample`（默认 `False`）。
2. 当 `do_sample=False` 时，`model.generate` 不传 temperature/top_p/top_k 且 `do_sample=False`。
3. 当 `do_sample=True` 时，允许使用 `config['inference']` 的采样参数。
4. 打印解码策略（greedy/采样）。

**检查机制**
- `rg -n "do_sample" evaluate.py`，确认新增参数与分支逻辑。
- 确认默认路径不包含 `temperature/top_p/top_k`。
- 若完成，勾选：
  - [ ] T2 通过

---

## T3. 修复 LoRA adapter 加载顺序

**目标**: 确保推理模式切换发生在 adapter 加载之后。

**涉及文件**: `evaluate.py`

**操作步骤**
1. 调整顺序为：`from_pretrained` → `load_adapter` → `for_inference` → `model.eval()`。
2. 保持 tokenizer padding_side 与 pad_token 逻辑不变。

**检查机制**
- 搜索 `load_adapter` 和 `for_inference` 的顺序，确认 `load_adapter` 在前。
- 确认新增 `model.eval()`。
- 若完成，勾选：
  - [ ] T3 通过

---

## T4. Gold events 使用已解析结构

**目标**: 优先使用 `sample.events` 作为 gold，避免解析失败影响指标。

**涉及文件**: `evaluate.py`

**操作步骤**
1. 在评估循环中优先使用 `sample.events`。
2. 若为空再回退解析 `sample.chosen`。

**检查机制**
- `rg -n "gold_events" evaluate.py`，确认存在 `sample.events` 优先逻辑。
- 若完成，勾选：
  - [ ] T4 通过

---

## T5. 统一训练/评估 Prompt 构建

**目标**: 训练与评估使用同一 Prompt 构建入口。

**涉及文件**: 
- `src/oglans/data/prompt_builder.py`
- `evaluate.py`
- `src/oglans/trainer/unsloth_trainer.py`

**操作步骤**
1. 在 `prompt_builder.py` 增加 `build_inference_prompt(tokenizer, text, use_oneshot=False)`。
2. `evaluate.py` 改用该函数构造 prompt。
3. 训练侧如可安全替换，也使用该函数，保持输出一致。

**检查机制**
- 确认 `prompt_builder.py` 新增函数。
- 确认 `evaluate.py` 不再自行拼接 `apply_chat_template`。
- 若完成，勾选：
  - [ ] T5 通过

---

## T6. 让 LANS 自适应阈值参与距离采样

**目标**: 负采样距离能随能力/进度变化。

**涉及文件**: 
- `src/oglans/trainer/unsloth_trainer.py`
- `src/oglans/utils/ds_cns.py`

**操作步骤**
1. 在 `LANSNegativeSampler.generate_rejected()` 获取当前能力或阈值。
2. 将动态值传入 `generate_negative_json()` / `select_confusing_event_type()`，替代固定 `current_step=0,total_steps=1`。
3. 如需新增可选参数，保持向后兼容。

**检查机制**
- 搜索 `generate_negative_json(` 调用处，确认不再使用固定 `0,1`。
- 若完成，勾选：
  - [ ] T6 通过

---

## T7. TRL 参数对齐

**目标**: DPOTrainer 使用 `processing_class` 传 tokenizer。

**涉及文件**: `src/oglans/trainer/unsloth_trainer.py`

**操作步骤**
1. 将 `tokenizer=self.tokenizer` 改为 `processing_class=self.tokenizer`。

**检查机制**
- `rg -n "processing_class|tokenizer=" src/oglans/trainer/unsloth_trainer.py`，确认替换完成。
- 若完成，勾选：
  - [ ] T7 通过

---

## T8. Debug 配置字段统一

**目标**: Debug 与正式配置字段一致。

**涉及文件**:
- `configs/config_debug.yaml`
- `src/oglans/trainer/unsloth_trainer.py`

**操作步骤**
1. 统一使用 `lans_alpha` 字段（推荐）。
2. 确保训练读取字段一致。

**检查机制**
- `rg -n "loss_baseline|lans_alpha" configs/config_debug.yaml src/oglans/trainer/unsloth_trainer.py`
- 若完成，勾选：
  - [ ] T8 通过

---

## T9. JSON 解析器中文单引号映射修正

**目标**: 规范中文单引号映射。

**涉及文件**: `src/oglans/utils/json_parser.py`

**操作步骤**
1. 将 `‘` 与 `’` 正确映射到 `'`。
2. 保持其余映射不变。

**检查机制**
- 打印 `PUNCTUATION_MAP`，确认包含 `‘` 和 `’`。
- 若完成，勾选：
  - [ ] T9 通过

---

## T10. API Key 去明文

**目标**: 不再在仓库中存储明文 key。

**涉及文件**:
- `configs/config.yaml`
- `evaluate_api.py`

**操作步骤**
1. 将 `api_key` 设为 `null` 或注释掉。
2. `evaluate_api.py` 中若环境变量缺失，则显式报错并退出。

**检查机制**
- `rg -n "api_key" configs/config.yaml evaluate_api.py`，确认无明文。
- 若完成，勾选：
  - [ ] T10 通过

---

## 完成确认

当所有任务通过后，在总开关处将全部勾选为完成。

