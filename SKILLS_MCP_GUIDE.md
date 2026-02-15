# Skills 与 MCP 使用指南（LLM 学术研究全流程）

本指南面向当前仓库的研究流程：
1. 课题可行性判断与联网分析
2. 按投稿要求设计 Python 实验
3. 分析实验结果并改写为投稿表达
4. 撰写论文
5. 每一步满足投稿规范（可复现、统计显著性、伦理与数据声明）

## 1. 当前已就绪能力

### 1.1 Skills（已安装）
- `academic-researcher`
- `deep-research`
- `arxiv-search`
- `read-arxiv-paper`
- `research`
- `research-engineer`
- `ml-paper-writing`
- `latex-paper-en`
- `mlflow-python`
- `statistical-hypothesis-testing`
- `statistical-analysis`
- `python-performance-optimization`
- `find-skills`

说明：原计划中的 `anton-abyzov/specweave@experiment-tracker` 在当前时间点不可安装，已用 `mlflow-python` 作为实验追踪替代。

### 1.2 MCP（已配置）
当前在 Codex 侧已配置并可用的 MCP：
- `fetch`：网页内容抓取（`uvx mcp-server-fetch`）
- `github`：代码仓库与 issue 证据检索（`mcp-server-github`，需 `GITHUB_PERSONAL_ACCESS_TOKEN`）
- `memory`：跨会话研究记忆（`mcp-server-memory`）
- `openalex`：开放学术元数据与引用网络（`openalex-mcp`，免申请 API key）
- `crossref`：DOI/期刊论文元数据检索（`crossref-mcp`，免申请 API key）
- `arxiv`：arXiv 论文检索与阅读（`uvx arxiv-mcp-server`）
- `sqlite`：本地实验数据库查询（`uvx mcp-server-sqlite`）
- `ripgrep`：代码与文本高速检索（`mcp-ripgrep`）
- `chrome-devtools`：网页调试与性能分析（`chrome-devtools-mcp --no-usage-statistics`）

配置文件路径（Codex）：
- `C:\Users\zimmmtly\.codex\config.toml`

## 2. 快速开始

1. 重启 Codex 客户端，使 MCP 配置生效。
2. 若使用 `github`，配置环境变量 `GITHUB_PERSONAL_ACCESS_TOKEN`（推荐）。
4. 先跑一个最小任务验证：
- `openalex` 查一篇目标论文
- `crossref` 用 DOI 回查元数据
- `fetch` 抓论文页面
- `github` 查实现仓库
- `deep-research` 输出可行性结论

## 3. 按研究流程使用（推荐模板）

### 3.1 课题可行性判断 + 联网分析
推荐组合：`deep-research` + `academic-researcher` + MCP(`openalex`,`crossref`,`fetch`,`github`)

可直接给助手的指令模板：
- `请使用 $deep-research 和 openalex/crossref/fetch/github，对课题「<topic>」做可行性分析，输出：问题定义、相关工作分层、SOTA对比、可复现风险、最小可行实验方案。`
- `请用 $academic-researcher 给出该课题的 novelty claim 与潜在 rebuttal 点。`

产出要求：
- 明确研究假设（Hypothesis）
- 至少 3 条可验证研究问题（RQ）
- 至少 2 条高风险失败模式与规避策略

### 3.2 设计符合投稿要求的 Python 实验
推荐组合：`research-engineer` + `mlflow-python` + `statistical-hypothesis-testing`

模板：
- `请使用 $research-engineer 把研究方案转成实验设计，包含数据划分、基线、消融、超参搜索、计算预算、复现实验脚本。`
- `请使用 $mlflow-python 设计实验追踪字段（run_name, seed, config_hash, metrics, artifacts）。`
- `请使用 $statistical-hypothesis-testing 设计显著性检验（多seed、bootstrap CI、paired test）。`

产出要求：
- 每个主结论对应一个可复现实验
- 指标定义、统计检验、停止准则都可审稿复查

### 3.3 结果分析与论文化改写
推荐组合：`statistical-hypothesis-testing` + `statistical-analysis` + `ml-paper-writing`

模板：
- `请分析实验结果，输出主结果、消融解释、误差分析、失败案例，并标注统计显著性。`
- `请使用 $ml-paper-writing 将结果改写成 Results + Analysis 小节，风格对齐 ACL/EMNLP。`

产出要求：
- 不只报最优值，必须报方差/区间
- 每个结论有证据句和图表引用句

### 3.4 论文撰写
推荐组合：`ml-paper-writing` + `latex-paper-en` + `read-arxiv-paper`

模板：
- `请使用 $ml-paper-writing 生成论文初稿结构：Abstract, Intro, Related Work, Method, Experiments, Limitations。`
- `请使用 $latex-paper-en 生成 LaTeX 段落与表格模板（含 caption 风格）。`
- `请使用 $read-arxiv-paper 抽取关键参考论文的可引用贡献点。`

产出要求：
- 摘要包含：问题、方法、结果、贡献
- 局限性与伦理声明明确独立小节

### 3.5 投稿一致性检查（每一步都做）
推荐组合：`research-engineer` + 仓库文档

固定检查清单：
- 可复现：对齐 `REPRODUCIBILITY_CHECKLIST.md`
- 评测规范：对齐 `ACADEMIC_EVALUATION_PROTOCOL.md`
- 数据声明：对齐 `DATA_STATEMENT.md`
- 伦理局限：对齐 `ETHICS_AND_LIMITATIONS.md`
- 指标说明：对齐 `ACADEMIC_METRICS_GUIDE.md`

## 4. 推荐工作流（单轮到端）

1. 用 `deep-research` 产出研究方案 V1。
2. 用 `research-engineer` 把 V1 转成实验计划与脚本清单。
3. 用 `mlflow-python` 定义记录规范并开始实验。
4. 用 `statistical-hypothesis-testing` 做显著性分析。
5. 用 `ml-paper-writing` + `latex-paper-en` 出可投稿草稿。
6. 用仓库 checklist 做提交前核对。

## 5. 常见问题与排查

- 问题：MCP 不生效。
- 处理：重启客户端；检查配置文件语法；确认命令在本机可执行。

- 问题：`github` 查询受限或频繁限流。
- 处理：配置 `GITHUB_PERSONAL_ACCESS_TOKEN`。

- 问题：`openalex` / `crossref` 结果不完整。
- 处理：交叉使用 `arxiv` + `fetch` 补全文献，并在提示词中加年份、领域关键词与 DOI 约束。

- 问题：无法申请 Semantic Scholar API key。
- 处理：使用 `openalex + crossref + arxiv` 组合作为主检索通道，通常已满足论文调研与引用链分析需求。

- 问题：某 skill 无法安装或仓库无有效 `SKILL.md`。
- 处理：用 `$find-skills` 搜索可替代 skill，再安装替代项。

## 6. 维护建议

- 每月执行一次：`npx skills check`
- 定期升级：`npx skills update`
- 新增技能前先记录用途与替代关系，避免能力重叠
- 重要配置变更前先备份 `config.toml` 与 `claude_desktop_config.json`

---

如需，我可以在下一步把这份指南再拆成两个版本：
- `SKILLS_QUICKSTART.md`（5分钟上手）
- `MCP_OPERATIONS.md`（配置/排错/安全）
