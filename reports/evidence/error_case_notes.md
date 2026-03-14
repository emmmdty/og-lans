# Error Case Notes

## Scope
- 本文件只基于现有静态产物做样例级审计，不训练、不重跑、不回算总指标。
- 重点对象是 `academic_full_local` 与 `academic_no_scv_local`。
- 次级参照对象是 `base_qwen4b_local`、`api_zeroshot_deepseek_chat`、`api_fewshot_deepseek_chat`。
- 样例主键统一使用 `id`；本轮未发现独立 `doc_id` 字段。

## JSONL Field Structure
- local/base JSONL 共同字段：
  - `id`, `text_preview`, `ground_truth`, `prediction`, `prediction_canonical`, `raw_response`, `parse_success`, `parse_method`, `repair_steps`, `canonical_role_rewrites`
- API JSONL 共同字段：
  - `id`, `sample_idx`, `text`, `gold`, `pred`, `pred_canonical`, `response`, `usage`, `parse_success`, `parse_error`, `parse_method`, `repair_steps`, `response_meta`
- 统一比较视图映射：
  - `sample_id <- id`
  - `text_for_review <- text_preview | text`
  - `gold_events <- ground_truth | gold`
  - `pred_events <- prediction | pred`
  - `pred_events_canonical <- prediction_canonical | pred_canonical`
  - `response_text <- raw_response | response`
  - `parse_success`, `parse_method`, `repair_steps`, `canonical_role_rewrites` 直接保留
  - `usage`, `response_meta` 仅 API 有值，本地三组记为 `NA`
- 对齐情况：
  - 5 个对象均为 `1171` 行
  - `academic_full_local` 与 `academic_no_scv_local` 的 `id` 交集为 `1171/1171`

## Comparison Heuristic
- 样例级主比较口径与 run-level 主指标保持一致，使用 strict triplet：
  - `(event_type, role, normalized_argument)`
- 归一化规则跟随 `evaluate.py`：
  - 去空白
  - 统一 `（ ） ， 。`
  - 英文小写
- `prediction_canonical` 只用于观察 canonicalization 是否改变样例结论，不用于回写原始错误类别。

## Typical Cases

### Full Correct, No-SCV Wrong
- `sample_id=2b9ed764ea681a7254207b22e39df35a`
  - `academic_full_local` 只抽到金标所需的 `企业破产/破产公司=誉衡集团`。
  - `academic_no_scv_local` 额外多报 `债权人=誉衡集团债权人`，形成 precision 型错误。
  - `base_qwen4b_local` 与 `api_zeroshot_deepseek_chat` 跟 `full` 一致；`api_fewshot_deepseek_chat` 也多报了 `债权人`。
  - 错误摘要：`no_scv` 在单事件样例上更容易多报附属角色。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:1054`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:1054`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:1054`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:195`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:195`

- `sample_id=66a55a53915fbdf74bcd5e14f863d40d`
  - `academic_full_local` 命中金标 `企业收购(聚美优品, 街电)`。
  - `academic_no_scv_local` 将同一事实改写成 `企业融资(投资方=聚美优品, 被投资方=街电)`。
  - `base_qwen4b_local` 与 `api_zeroshot_deepseek_chat` 站在 `full` 一侧；`api_fewshot_deepseek_chat` 同时输出收购和融资，表现为过预测。
  - 错误摘要：`no_scv` 在“收购/融资”相邻语义上出现事件类型翻转。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:614`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:614`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:614`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:483`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:483`

- `sample_id=389655d8f75c394dbd62f80eced5ad00`
  - `academic_full_local` 命中 `企业融资` 全部 5 个金标论元。
  - `academic_no_scv_local` 唯一漏掉 `披露时间=5月3日`。
  - `base_qwen4b_local` 与两条 API 线都保住了时间或只在 `Pre-A/Pre-A轮` 上有轻微表述差异。
  - 错误摘要：`no_scv` 在简单单事件样例上丢失显式披露时间。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:413`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:413`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:413`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:257`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:257`

### No-SCV Correct, Full Wrong
- `sample_id=58567406e01800c4d1ff10835ffac74d`
  - `academic_full_local` 原始预测把 `股票简称=宁波银行` 写成了 `被减持方=宁波银行`。
  - `academic_no_scv_local` 直接输出金标角色。
  - `base_qwen4b_local` 与 `academic_full_local` 原始预测同错，但二者 `prediction_canonical` 都可修正为 `股票简称`。
  - 两条 API 线在原始预测层面已直接正确。
  - 错误摘要：这是最清晰的 role alias / canonicalization 样例之一。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:663`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:663`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:663`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:411`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:411`

- `sample_id=7de944ac12541a2b3021d3dbe264ecca`
  - `academic_full_local` 用 `破产时间=2019年9月28日` 替代了金标 `披露时间=2019年9月28日`。
  - `academic_no_scv_local` 角色槽位正确。
  - `base_qwen4b_local` 与两条 API 线也都出现时间槽位混用，且额外引入了 `2019年10月24日` 这一披露节点。
  - 错误摘要：`no_scv` 在这个样例上比其他四条线都更贴近金标口径。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:356`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:356`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:356`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:593`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:593`

- `sample_id=76e991541d5ba919309b396785ea136e`
  - `academic_full_local` 漏掉 `披露时间=9月3日`。
  - `academic_no_scv_local` 命中 4 个金标论元。
  - `base_qwen4b_local` 与 API 两线都能保住披露时间，但更容易额外生出 `收购标的` 或把金额扩写为 `9.5亿美元现金`。
  - 错误摘要：`no_scv` 在简单显式时间抽取上优于 `full`。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:232`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:232`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:232`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:553`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:553`

### Both Wrong, But Differently
- `sample_id=178cba7840f03f81daa8cc99e736e944`
  - 金标只要 `收购方=LVMH` 与 `被收购方=Tiffany`。
  - `academic_full_local` 额外引入 `监管机构=欧盟委员会` 与 `审查结果=通过`。
  - `academic_no_scv_local` 不再多报这两个角色，但仍把 `LVMH` 扩写为 `LVMH集团`。
  - `base_qwen4b_local` 跟 `full` 更像；API 两线则主要卡在别名扩写和 `Tiffany(蒂芙尼)` 上。
  - 错误摘要：两者都错，但 `full` 更偏向附加监管语义，`no_scv` 更接近“少报/别名扩写”。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:906`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:906`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:906`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:101`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:101`

- `sample_id=a413e8c72bba3359cc9a5cf6cea4fc8b`
  - 金标是单一 `企业破产(OneWeb, 今年3月)`。
  - `academic_full_local` 只错在 `破产时间` 写成 `披露时间=3月`。
  - `academic_no_scv_local` 除了同类时间槽位问题，还额外生成一条 `企业收购` 事件，把竞购背景当成主事件。
  - `base_qwen4b_local` 把这类“破产 + 多家竞购”混合得更重；API 两线只保留 `企业破产`，但仍多报 `披露时间=5月8日`。
  - 错误摘要：多主题新闻里，`no_scv` 更容易把背景竞购信息抬升为主事件。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:147`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:147`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:147`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:755`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:755`

- `sample_id=81bb2a40ae7ec7af28e95ce40abc970e`
  - 金标同时包含 `企业收购` 和 `质押` 两类事件。
  - `academic_full_local` 直接 `no_json_found`，整条样例失守。
  - `academic_no_scv_local` 能解析出两类事件，但把 `2019年7月5日/2019年8月9日/2019年8月8日` 等时间链条混在一起，并把若干角色改写成 `收购比例/收购方式/贷款金额/事件时间`。
  - 两条 API 线也能解析两类事件，但收购披露时间被漂移到 `2019年8月9日`。
  - 错误摘要：这是典型的长距离、多日期、多事件交错样例。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:327`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:327`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:327`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:613`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:613`

### Parse Divergence
- `sample_id=019e72e570e8082c4ac9f537e221a12d`
  - `academic_full_local` `parse_success=false`, `parse_method=no_json_found`。
  - 检查 `raw_response` 后可见其停在长段 `<thought>` 推理末尾，没有进入可解析 JSON。
  - `academic_no_scv_local` 虽然成功输出 JSON，但把聚合统计背景也抽成了额外 `股东减持` 事件。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:637`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:637`

- `sample_id=fc489b7b91f05e3eaee453063154d7ff`
  - `academic_no_scv_local` `parse_success=false`, `parse_method=no_json_found`。
  - 该条 `raw_response` 是纯文本抄录式摘要，没有 JSON 边界。
  - `academic_full_local` 至少输出了结构化 `股份回购`，虽然仍有 `117,600股` / `5.71元/股` / `交易完成时间` 等 strict 不匹配。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:665`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:665`

- `sample_id=81bb2a40ae7ec7af28e95ce40abc970e`
  - `academic_full_local` 再次出现 `no_json_found`。
  - `academic_no_scv_local` 能输出两事件，但 role schema 与时间链条都偏离金标。
  - 错误摘要：parse 是否成功，直接决定后续“多事件文本”还能不能进入 role-level 比较。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:327`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:327`

### Canonicalization-Sensitive Cases
- `sample_id=0a85c1c8ec6d4042be78784839cb4116`
  - `academic_full_local` 原始预测把 `回购完成时间` 写成了 `事件时间`，`prediction_canonical` 可修正为金标。
  - `academic_no_scv_local` 同样发生 1 次 rewrite，但还额外生出一条无关 `企业收购` 事件，所以 canonical 后仍不对。
  - `base_qwen4b_local` 与两条 API 线也都体现了“`事件时间 -> 回购完成时间`”这一 role rewrite 模式。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:385`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:385`, `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results.jsonl:385`, `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl:39`, `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl:39`

- `sample_id=58567406e01800c4d1ff10835ffac74d`
  - `academic_full_local` 的 `被减持方 -> 股票简称` 需要 1 次 canonical rewrite。
  - `academic_no_scv_local` 原始预测就已对齐金标。
  - 这是“canonicalization 能修，但 raw output 仍错”的典型样例。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:663`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:663`

- `sample_id=df0ae19741f364a0a8430dc98e44f6da`
  - `academic_full_local` 与 `academic_no_scv_local` 都把 `股票简称=绝味食品` 写成了 `被增持方=绝味食品`。
  - 两者 `prediction_canonical` 都能回到金标。
  - 该样例说明 canonicalization 对本地两条线都是实质性补丁，而不是只作用于单一变体。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:273`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:273`

### Schema-Violation Candidates
- `sample_id=58567406e01800c4d1ff10835ffac74d`
  - `academic_full_local` 原始输出 `股东减持|被减持方=宁波银行`。
  - 该 `(event_type, role)` 组合与 full run summary 中的 `invalid_role:股东减持|被减持方` 一致。
  - 说明：这里仅标为 candidate；样例级 schema 结论仍以 run summary breakdown 为参照，不宣称穷尽复核。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:663`, `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_summary.json`

- `sample_id=0a85c1c8ec6d4042be78784839cb4116`
  - `academic_full_local` 原始输出 `股份回购|事件时间=2020年11月11日`。
  - 该组合与 run summary 中的 `invalid_role:股份回购|事件时间` 高度对齐。
  - `prediction_canonical` 会把它改成 `回购完成时间`，说明这类错误确实落在 role canonicalization 覆盖面内。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:385`, `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_summary.json`

- `sample_id=c45668926f8b7a09a9fff66254420147`
  - `academic_full_local` 与 `academic_no_scv_local` 都原始输出了 `股份回购|事件时间=2020年9月8日`，canonical 后才回到 `回购完成时间`。
  - 该样例不是 full/no_scv 的分歧样例，但能证明此类 invalid role 在两条本地线都真实存在。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:7`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:7`

### Long-Distance / Cross-Sentence Candidates
- `sample_id=81bb2a40ae7ec7af28e95ce40abc970e`
  - 同一篇文档里交织 `企业收购` 与 `质押`，还带两组日期。
  - `academic_no_scv_local` 能解析结构，但在 `2019年7月5日 / 2019年8月9日 / 2019年8月8日` 之间发生跨句错配。
  - `academic_full_local` 则直接 parse fail。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:327`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:327`

- `sample_id=a413e8c72bba3359cc9a5cf6cea4fc8b`
  - 破产主线之外，还有“多家公司竞购 OneWeb”的背景段落。
  - `academic_no_scv_local` 把背景竞购提升成了新的 `企业收购` 事件。
  - `academic_full_local` 只保留了破产主线，但时间角色仍错。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl:147`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:147`

- `sample_id=0a85c1c8ec6d4042be78784839cb4116`
  - `text_preview` 中主事件是 `艾德韦宣集团回购`，但后半段还拼接了 `*ST美讯` 的股权出售新闻。
  - `academic_no_scv_local` 额外生成了无关 `企业收购` 事件，说明文本拼接/尾部干扰对它更敏感。
  - Evidence: `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl:385`

## Hallucination Note
- 本地三组 JSONL 只有 `text_preview`，没有完整 `text`。
- 因此，本轮不输出“样例级 hallucination 定案列表”。
- 只能保留两个保守结论：
  - run-level `hallucination_breakdown` 已在 summary 中存在，可作为后续筛样方向。
  - 真正的样例级 hallucination 复核，需要本地 run 对应的完整原文，而不仅是 `text_preview`。

## Best Case Study Candidates
- `66a55a53915fbdf74bcd5e14f863d40d`
  - 事件类型翻转非常干净，full/base/api_zeroshot 一致，no_scv 单独偏到 `企业融资`。
- `58567406e01800c4d1ff10835ffac74d`
  - role alias / canonicalization 样例最清楚，且 full、base、API 之间能形成层次对照。
- `019e72e570e8082c4ac9f537e221a12d`
  - 直接展示 parse fail 与“解析成功但过泛化”是两种不同风险。
- `81bb2a40ae7ec7af28e95ce40abc970e`
  - 多事件、多日期、跨句信息整合失败非常典型。
- `0a85c1c8ec6d4042be78784839cb4116`
  - 一条样例同时覆盖 canonicalization、schema-role candidate、文本拼接干扰三类现象。
