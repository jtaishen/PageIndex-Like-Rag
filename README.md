# PageIndex-Like RAG MVP

这是一个基于 OpenCode 的 PageIndex-like 论文知识库智能体 MVP。当前目标是跑通论文入库、结构化工件、章节树检索和 evidence packet 的核心闭环：

```text
parse -> normalize -> tree -> artifacts -> indexes -> evidence packet -> CLI / MCP 工具
```

## 当前能力

- 递归扫描目录，支持增量同步。
- 支持 Markdown、TXT、DOCX、HTML 基础解析。
- PDF 解析通过可选依赖 `pypdf` 支持，并可用 `KB_PDF_PARSER` 或 `--pdf-parser` 选择 `auto`、`pypdf`、`docling`、`grobid`。
- 解析后生成 `raw_text.txt`、`body.md`、`structured.json`、`metadata.json`、`references.json`、`parse_report.json`、`tree.json`、`node_index.jsonl`、`doc_card.json` 等工件。
- 对中文论文常见结构做规则识别，包括摘要、关键词、第 X 章、`1.1`/`1.1.1` 小节、结论、参考文献、图和表。
- 将文档保存为 `documents` 和 `doc_nodes`。
- 使用 SQLite FTS5 做全文检索，并支持本地 embedding + hybrid rerank。
- 返回带 `doc_id`、`node_id`、`node_path`、页码和 excerpt 的 evidence packet。
- 提供 CLI 命令，包括 `card`、`artifacts`、`quality`、`parse-report`、`embed`、`search-report`、`eval-search`、`eval-review`、`eval-memory`、`query-log`、`query-stats`、`extract`、`innovations`、`citations`。
- 支持跨论文比较和综述规划任务工件，生成比较矩阵、综述大纲、章节证据表和下一步行动。
- 支持长期 memory 写入门控、任务进度记忆、任务恢复和任务进度压缩。
- 提供可选 MCP server，供 OpenCode 调用。

## 快速开始

```bash
python3 -m kb_agent.cli sync ./papers
python3 -m kb_agent.cli list
python3 -m kb_agent.cli search "agent memory"
python3 -m kb_agent.cli ask "这些文章里哪些提到了 memory compaction?"
```

默认数据库在：

```text
data/kb.sqlite
```

也可以指定数据库：

```bash
python3 -m kb_agent.cli --db /tmp/kb.sqlite sync ./papers
```

## v0.3 论文结构演示

同步真实 PDF 并强制使用当前解析器版本重建工件：

```bash
uv run --extra pdf python -m kb_agent.cli sync articles --force
```

查看文档、章节树、工件和解析质量：

```bash
uv run python -m kb_agent.cli list
uv run python -m kb_agent.cli tree <doc_id>
uv run python -m kb_agent.cli card <doc_id>
uv run python -m kb_agent.cli artifacts <doc_id>
uv run python -m kb_agent.cli quality <doc_id>
```

只看证据、不调用模型：

```bash
uv run python -m kb_agent.cli ask "这篇论文的主要研究内容是什么？" --no-llm
```

## v0.4 论文理解工件抽取

生成或刷新单篇论文的创新点与引用关系工件：

```bash
uv run python -m kb_agent.cli extract <doc_id>
```

如果只使用规则抽取、不调用 DeepSeek：

```bash
uv run python -m kb_agent.cli extract <doc_id> --no-llm --force
```

查看抽取结果：

```bash
uv run python -m kb_agent.cli innovations <doc_id>
uv run python -m kb_agent.cli citations <doc_id>
```

## v0.5 跨论文比较与综述任务工件

v0.5 将单篇论文工件升级为多论文任务工件。默认数据库下，任务运行态文件会写入项目根目录的 `.kb_state/`，该目录不提交到 Git。

先同步真实论文并抽取单篇理解工件：

```bash
uv run --extra pdf python -m kb_agent.cli sync articles --force
uv run python -m kb_agent.cli list
uv run python -m kb_agent.cli extract <doc_id> --no-llm --force
```

生成跨论文比较矩阵：

```bash
uv run python -m kb_agent.cli compare "服务机器人与多智能体任务规划方法对比" --no-llm
```

生成综述规划工件：

```bash
uv run python -m kb_agent.cli generate-review "任务规划方法研究综述" --no-llm
```

查看任务工件：

```bash
uv run python -m kb_agent.cli task-artifact <task_id> selected_papers.json
uv run python -m kb_agent.cli task-artifact <task_id> comparison_matrix.json
uv run python -m kb_agent.cli task-artifact <task_id> review_outline.json
uv run python -m kb_agent.cli task-artifact <task_id> section_evidence/background_problem.json
```

如果已配置 DeepSeek，可以去掉 `--no-llm`，让模型基于 evidence packet 生成结构化比较和综述大纲；如果必须调用模型成功，则添加 `--require-llm`。

## v0.6 综述正文草稿与引用检查

v0.6 会基于 v0.5 的 `review_outline.json` 和 `section_evidence/*.json` 逐节生成综述草稿，并写入同一个 `.kb_state/<task_id>/` 任务目录。

生成章节草稿、总稿和质量报告：

```bash
uv run python -m kb_agent.cli draft-review <task_id> --no-llm
```

只生成某个章节：

```bash
uv run python -m kb_agent.cli draft-review <task_id> --section-id background_problem
```

重新组装已有章节草稿：

```bash
uv run python -m kb_agent.cli assemble-review <task_id>
```

检查正文中的 `[E1]`、`[E2]` 等证据编号是否能映射到章节证据：

```bash
uv run python -m kb_agent.cli check-review <task_id>
```

查看 v0.6 工件：

```bash
uv run python -m kb_agent.cli task-artifact <task_id> section_drafts/background_problem.json
uv run python -m kb_agent.cli task-artifact <task_id> review_draft.md
uv run python -m kb_agent.cli task-artifact <task_id> citation_check.json
uv run python -m kb_agent.cli task-artifact <task_id> review_report.json
```

## v0.7 记忆写入门控与任务恢复

长期 memory 只保存用户偏好、项目规则和跨 session 任务进度，不保存论文正文、检索命中、evidence packet 或大段综述草稿。

安全写入长期记忆：

```bash
uv run python -m kb_agent.cli memory-put project preference citation_style "GB/T 7714"
```

写入带 TTL 或来源的记忆：

```bash
uv run python -m kb_agent.cli memory-put project task_progress current_review "已生成综述草稿，下一步检查引用。" --ttl-days 30 --refs ".kb_state/<task_id>"
```

将当前任务压缩为长期任务进度：

```bash
uv run python -m kb_agent.cli remember-task <task_id>
```

恢复最近任务并查看建议命令：

```bash
uv run python -m kb_agent.cli resume-task
```

压缩重复任务进度：

```bash
uv run python -m kb_agent.cli memory-compact --scope project
```

## v0.8 高质量 PDF 解析与质量诊断

v0.8 保留 `pypdf` 作为稳定兜底，同时支持可选 Docling 和 GROBID 增强。默认 `auto` 会先尝试 Docling，本地不可用或失败时回退到 `pypdf`；配置 `GROBID_URL` 后会增强元数据和参考文献。

强制使用 pypdf：

```bash
uv run --extra pdf python -m kb_agent.cli sync articles --force --pdf-parser pypdf
```

安装并使用 Docling：

```bash
uv sync --extra pdf --extra docling
uv run --extra pdf --extra docling python -m kb_agent.cli sync articles --force --pdf-parser docling
```

配置 GROBID 服务增强元数据和参考文献：

```bash
GROBID_URL=http://localhost:8070 uv run --extra pdf python -m kb_agent.cli sync articles --force --pdf-parser auto
```

查看解析质量和解析链诊断：

```bash
uv run python -m kb_agent.cli quality <doc_id>
uv run python -m kb_agent.cli parse-report <doc_id>
```

`parse_report.json` 会记录 `parser_chain`、`fallback_used`、外部解析器失败原因和 adapter 状态；Marker 本轮只作为可检测占位，不作为默认解析器。`parse_quality` 会输出 `metadata_score`、`structure_score`、`reference_score`、`warning_count` 和 `quality_level`。

## v0.9 混合检索、重排与评测

v0.9 在 FTS5 基础上增加本地 embedding、混合检索和搜索评测。默认 provider 是离线可用的 `hash`，不会下载模型；如果没有构建 embedding，`hybrid` 会自动降级为 FTS。

构建语义索引：

```bash
uv run python -m kb_agent.cli embed --provider hash --force
```

同步后立即构建默认 embedding：

```bash
uv run --extra pdf python -m kb_agent.cli sync articles --force --pdf-parser pypdf --build-embeddings
```

查看混合检索候选、融合分数和降级原因：

```bash
uv run python -m kb_agent.cli search-report "多智能体任务规划的主要方法"
```

问答、比较和综述规划都可以选择检索模式：

```bash
uv run python -m kb_agent.cli ask "这两篇论文的任务规划方法有什么区别？" --no-llm --search-mode hybrid
uv run python -m kb_agent.cli compare "服务机器人与多智能体任务规划方法对比" --no-llm --search-mode hybrid
uv run python -m kb_agent.cli generate-review "任务规划方法研究综述" --no-llm --search-mode hybrid
```

运行搜索评测：

```bash
uv run python -m kb_agent.cli eval-search tests/fixtures/search_eval_queries.json --search-mode hybrid
```

如果需要 sentence-transformers：

```bash
uv sync --extra embeddings
KB_EMBEDDING_PROVIDER=sentence-transformers uv run python -m kb_agent.cli embed --force
```

## v0.10 LLM Tree Search 与可解释树检索

v0.10 在 hybrid/FTS 候选基础上增加论文内部树检索：先识别查询意图，再用 value function 为章节、段落、图表、参考文献节点打分，最后返回可解释的 `tree_search_trace`。默认 `ask/search/compare/generate-review` 仍保持 hybrid；需要树搜索时显式指定 `--search-mode tree`。

查看查询意图：

```bash
uv run python -m kb_agent.cli classify-query "这篇论文的方法设计是什么？" --no-llm
```

在单篇论文内运行树搜索：

```bash
uv run python -m kb_agent.cli tree-search <doc_id> "这篇论文的方法设计是什么？" --no-llm
```

让问答、比较和综述规划使用树搜索证据：

```bash
uv run python -m kb_agent.cli ask "这两篇论文的任务规划方法有什么区别？" --no-llm --search-mode tree
uv run python -m kb_agent.cli compare "服务机器人与多智能体任务规划方法对比" --no-llm --search-mode tree
uv run python -m kb_agent.cli generate-review "任务规划方法研究综述" --no-llm --search-mode tree
```

`tree_search_trace` 会记录 query profile、评分分解、展开路径、最终 evidence、LLM fallback 原因和读取到的长期偏好摘要。树搜索只读取允许的 memory 偏好，不会写入论文正文、检索命中或 evidence packet。

## v0.11 评测闭环、查询日志与 OpenCode 观测

v0.11 增加统一 query log、搜索多模式评测、综述任务评测、memory 卫生评测和 OpenCode observer hook。日志只保存 query、doc/node ID、指标和 warning，不保存论文正文、长 excerpt 或 evidence packet。

对比 hybrid、tree、fts 三种检索模式：

```bash
uv run python -m kb_agent.cli eval-search tests/fixtures/search_eval_queries.json --compare-modes hybrid,tree,fts
```

评测综述任务的引用覆盖和未支撑段落：

```bash
uv run python -m kb_agent.cli eval-review <task_id>
```

检查长期 memory 是否存在过期、重复或疑似论文资产污染：

```bash
uv run python -m kb_agent.cli eval-memory
```

查看最近查询日志和聚合统计：

```bash
uv run python -m kb_agent.cli query-log --limit 10
uv run python -m kb_agent.cli query-stats --since-days 7
```

OpenCode 已配置 `.opencode/plugins/kb-observer/index.mjs`。它会在 `kb_tree_search`、`kb_compare`、`kb_generate_review`、`kb_check_review_citations` 和评测工具运行后，把任务状态和质量告警摘要写入 `.kb_state/opencode_observer/`，并在会话压缩时注入短上下文；不会写入论文正文或 evidence。

## PDF 和 MCP 可选依赖

如果要解析 PDF：

```bash
uv sync --extra pdf
```

如果要启用 Docling PDF 增强：

```bash
uv sync --extra pdf --extra docling
```

如果要启用 sentence-transformers embedding 增强：

```bash
uv sync --extra embeddings
```

如果要启用 OpenCode MCP server：

```bash
uv sync --extra mcp --extra pdf
uv run --extra mcp python -m kb_agent.mcp_server
```

## DeepSeek 配置

`kb ask` 会优先读取本地 `.env` 或系统环境变量中的 DeepSeek 配置。不要把真实 API key 提交到 GitHub。

```bash
cp .env.example .env
```

然后编辑 `.env`：

```text
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

配置后运行：

```bash
python3 -m kb_agent.cli ask "这些文章关于 memory 的核心观点是什么？"
```

如果只想看证据、不调用模型：

```bash
python3 -m kb_agent.cli ask "memory write gate 是什么？" --no-llm
```

OpenCode 配置见 [opencode.json](/Users/jtai/Desktop/PageIndex-Like-Rag/opencode.json)。
项目默认模型已设置为：

```text
deepseek/deepseek-v4-pro
```

DeepSeek 官方 OpenCode 接入方式：

1. 在项目目录执行 `opencode`。
2. 在 OpenCode 输入框输入 `/connect`。
3. 输入并选择 `deepseek` provider。
4. 粘贴 DeepSeek API key。
5. 选择 `DeepSeek-V4-Pro` 模型。

## OpenCode 使用方式

`.opencode/skills/paper-qa/SKILL.md` 约束 agent 先检索文档、再检索树节点、最后只基于 evidence packet 回答。

推荐工具调用顺序：

```text
kb_sync -> kb_build_semantic_index -> kb_search_docs -> kb_get_doc_card -> kb_get_parse_quality -> kb_get_parse_report -> kb_extract_doc_insights -> kb_get_innovations -> kb_get_citation_map -> kb_classify_query -> kb_tree_search -> kb_search_tree -> kb_get_evidence -> kb_answer -> kb_compare -> kb_generate_review -> kb_draft_review -> kb_check_review_citations -> kb_assemble_review -> kb_eval_search -> kb_eval_review -> kb_eval_memory -> kb_get_query_stats -> memory_remember_task -> memory_resume_task -> kb_get_task_artifact
```

## 测试

```bash
uv run python -m unittest discover -s tests
```

## 后续阶段

- 增加人工反馈闭环、评测集管理、查询日志可视化和更完整的 OpenCode 工作流自动化。
