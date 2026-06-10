# PageIndex-Like RAG MVP

这是一个基于 OpenCode 的 PageIndex-like 论文知识库智能体 MVP。当前目标是跑通论文入库、结构化工件、章节树检索和 evidence packet 的核心闭环：

```text
parse -> normalize -> tree -> artifacts -> indexes -> evidence packet -> CLI / MCP 工具
```

## 当前能力

- 递归扫描目录，支持增量同步。
- 支持 Markdown、TXT、DOCX、HTML 基础解析。
- PDF 解析通过可选依赖 `pypdf` 支持，并可用 `KB_PDF_PARSER` 或 `--pdf-parser` 选择 `auto`、`pypdf`、`docling`、`grobid`。
- 解析后生成 `raw_text.txt`、`body.md`、`structured.json`、`metadata.json`、`references.json`、`layout_blocks.json`、`tables.json`、`table_content.json`、`table_summaries.json`、`figures.json`、`reference_sections.json`、`parse_report.json`、`tree.json`、`node_index.jsonl`、`doc_card.json` 等工件。
- 对中文论文常见结构做规则识别，包括摘要、关键词、第 X 章、`1.1`/`1.1.1` 小节、结论、参考文献、图和表。
- 将文档保存为 `documents` 和 `doc_nodes`。
- 使用 SQLite FTS5 做全文检索，并支持本地 embedding + hybrid rerank。
- 返回带 `doc_id`、`node_id`、`node_path`、页码和 excerpt 的 evidence packet。
- 提供 CLI 命令，包括 `card`、`artifacts`、`quality`、`parse-report`、`layout`、`tables`、`table-content`、`table-summaries`、`embed`、`search-report`、`eval-search`、`eval-review`、`eval-memory`、`eval-facts`、`audit-facts`、`fact-conflicts`、`graph-build`、`graph-neighborhood`、`graph-export`、`graph-report`、`quality-baseline`、`latest-quality-baseline`、`eval-suite`、`benchmark`、`analyze-failures`、`case-study`、`query-log`、`query-stats`、`feedback-put`、`feedback-to-eval`、`eval-dashboard`、`tune-search`、`search-profile`、`extract`、`innovations`、`citations`、`extract-facts`、`claims`、`entities`、`relations`、`fact-search`。
- 支持跨论文比较和综述规划任务工件，生成比较矩阵、综述大纲、章节证据表和下一步行动。
- 支持长期 memory 写入门控、任务进度记忆、任务恢复和任务进度压缩。
- 支持人工反馈闭环，可将用户评分、期望 doc/node/keyword 转为搜索评测集。
- 支持基于评测集生成 search profile，并通过显式 `--search-mode auto` 使用调优策略。
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

## v0.12 人工反馈闭环与评测集管理

v0.12 将人工判断沉淀为结构化反馈，再转换成可复跑的搜索评测集。反馈只保存 query、评分、标签、期望 doc/node/keyword 和短评论；如果评论包含 `node_id=...`、`page_range=...`、`excerpt=...` 这类论文资产内容，会拒绝保存评论正文并留下 warning。

记录一次反馈：

```bash
uv run python -m kb_agent.cli feedback-put "动态角色任务规划" \
  --operation ask \
  --rating 5 \
  --label good \
  --expected-doc-id <doc_id> \
  --expected-node-id <node_id> \
  --expected-keyword 动态角色 \
  --preferred-search-mode tree \
  --comment "树搜索命中了方法章节"
```

查看反馈并转成评测集：

```bash
uv run python -m kb_agent.cli feedback-list --limit 10
uv run python -m kb_agent.cli feedback-to-eval --min-rating 4
```

对反馈生成的评测集比较多种检索模式：

```bash
uv run python -m kb_agent.cli eval-search data/eval_sets/<feedback_eval>.json --compare-modes hybrid,tree,fts
```

生成静态复盘报告：

```bash
uv run python -m kb_agent.cli eval-dashboard --since-days 7
```

`query-stats` 现在也会汇总反馈数量、平均评分、低评分数量、反馈标签分布和各 search mode 的反馈分布。OpenCode observer 会在证据不足、fallback 或综述引用覆盖低时提示使用 `kb_put_feedback -> kb_build_eval_set_from_feedback -> kb_eval_search` 做复盘。

## v0.13 真实 Embedding 与评测驱动检索调优

v0.13 增强 `sentence-transformers` provider，并用评测集生成本地 search profile。默认检索模式仍是 `hybrid`；只有显式传入 `--search-mode auto` 时，系统才会读取 active profile，根据查询意图选择 `hybrid`、`tree` 或 `fts`。

查看 embedding 状态：

```bash
uv run python -m kb_agent.cli embed --status --provider hash
uv run python -m kb_agent.cli embed --status --provider sentence-transformers --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

构建真实 embedding：

```bash
uv sync --extra embeddings
uv run python -m kb_agent.cli embed --provider sentence-transformers --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --batch-size 16 --force
```

基于评测集调优检索策略并保存 profile：

```bash
uv run python -m kb_agent.cli tune-search data/eval_sets/<feedback_eval>.json --compare-modes hybrid,tree,fts --save-profile paper-v1
uv run python -m kb_agent.cli search-profile list
uv run python -m kb_agent.cli search-profile apply paper-v1
```

显式使用 auto 策略：

```bash
uv run python -m kb_agent.cli search "动态角色任务规划" --search-mode auto
uv run python -m kb_agent.cli ask "这篇论文的方法设计是什么？" --no-llm --search-mode auto
uv run python -m kb_agent.cli compare "服务机器人与多智能体任务规划方法对比" --no-llm --search-mode auto
uv run python -m kb_agent.cli generate-review "任务规划方法研究综述" --no-llm --search-mode auto
```

生成 HTML 仪表盘：

```bash
uv run python -m kb_agent.cli eval-dashboard --format html --since-days 7
```

dashboard 会展示 query log、feedback、eval report、search tuning 和 active profile 摘要，但不会展示论文正文、长 excerpt、evidence packet 或综述草稿正文。

## v0.14 复杂 PDF 版面解析与图表结构增强

v0.14 将 PDF 解析结果统一为 `layout_block.v1`，并新增图表、表格和参考文献区域工件。`pypdf` 仍是稳定兜底，但会先做页眉页脚、页码、DOI/版权等噪声清理，再按摘要、关键词、章节、图题、表题、公式样式、参考文献条目拆分版面块。Docling 可用时会优先融合其结构块、bbox、表格、图片和 caption；GROBID 可用时继续增强元数据和参考文献结构。

重建真实 PDF 工件：

```bash
uv run --extra pdf python -m kb_agent.cli sync articles --force --pdf-parser pypdf
```

查看版面、图表、树和质量：

```bash
uv run python -m kb_agent.cli parse-report <doc_id>
uv run python -m kb_agent.cli layout <doc_id>
uv run python -m kb_agent.cli figures <doc_id>
uv run python -m kb_agent.cli tables <doc_id>
uv run python -m kb_agent.cli tree <doc_id>
uv run python -m kb_agent.cli quality <doc_id>
```

如果 `quality` 显示 `page_only_tree`、`weak_layout_blocks` 或章节数量过少，优先尝试 Docling：

```bash
uv sync --extra pdf --extra docling
uv run --extra pdf --extra docling python -m kb_agent.cli sync articles --force --pdf-parser docling
```

本轮不做 OCR。扫描版 PDF 会给出 `scanned_pdf_or_empty_text` 或弱解析 warning，需要后续接入 OCR/版面模型。

## v0.15 Claims / Entities / Relations 事实层

v0.15 将单篇论文的 evidence nodes、创新点、引用关系和版面工件进一步抽取为可查询事实层。事实层会写入 SQLite schema v6 的 `paper_claims`、`paper_entities`、`paper_relations`，同时生成 `claims.json`、`entities.json`、`relations.json`、`fact_graph.json`、`fact_report.json`。

抽取事实层：

```bash
uv run python -m kb_agent.cli extract-facts <doc_id> --no-llm --force
```

查看事实工件：

```bash
uv run python -m kb_agent.cli claims <doc_id>
uv run python -m kb_agent.cli entities <doc_id>
uv run python -m kb_agent.cli relations <doc_id>
uv run python -m kb_agent.cli fact-graph <doc_id>
```

搜索 claims / entities / relations：

```bash
uv run python -m kb_agent.cli fact-search "动态角色任务规划" --type claim
uv run python -m kb_agent.cli fact-search "任务完成率" --type entity
```

`search-report` 会附带 `fact_matches` 摘要，`eval-dashboard` 会展示事实层覆盖率。事实表和日志只保存短 claim、实体名、关系摘要和 evidence ID，不保存长 excerpt 或论文正文。

## v0.16 表格内容结构化与事实层评测闭环

v0.16 将表格从 caption/layout 级别推进到保守的行列内容工件。Docling 可用时优先保留结构化 rows/cells/bbox；pypdf 和文本兜底只在表题附近识别明显多列文本，不做 OCR，也不会把弱表格伪装成高质量结果。新增工件包括 `table_content.json` 和 `table_summaries.json`，`parse_quality` 会展示 `table_content_count`、`table_parse_score` 和 `table_warning_count`。

查看表格内容：

```bash
uv run python -m kb_agent.cli table-content <doc_id>
uv run python -m kb_agent.cli table-summaries <doc_id>
uv run python -m kb_agent.cli quality <doc_id>
```

表格事实会进入 v0.15 的 facts 表，来源标记为 `table_rule` 或 `docling_table`，并在 evidence 中绑定 `table_id`、`layout_block_id`、`node_id` 和页码范围。可以按来源过滤事实：

```bash
uv run python -m kb_agent.cli extract-facts <doc_id> --no-llm --force
uv run python -m kb_agent.cli fact-search "任务完成率" --source table --min-confidence 0.5
uv run python -m kb_agent.cli eval-facts --doc-id <doc_id>
```

`eval-facts` 会生成 `fact_eval.v1` 到 `data/eval/`，统计 claim/entity/relation 数量、证据覆盖率、低置信率、重复率、无 `node_id` 事实数、表格事实覆盖率和弱解析 warning。`search-report` 的 `fact_matches` 会标出 `source_kind` 与 confidence；`eval-dashboard` 会展示最新事实评测与表格事实数量。

## v0.17 真实评测套件与 PageIndex-like 核心能力验证

v0.17 把零散评测升级为可复用的真实评测套件。评测套件只保存 query、期望 doc/node/keyword、期望事实来源和短标签；benchmark 报告只保存指标、ID、warning 和短摘要，不保存论文正文、长 excerpt 或 evidence packet。

创建评测套件：

```bash
uv run python -m kb_agent.cli eval-suite create paper-core --input-json data/eval_sets/core_queries.json
uv run python -m kb_agent.cli eval-suite create feedback-core --from-feedback --min-rating 4
uv run python -m kb_agent.cli eval-suite list
uv run python -m kb_agent.cli eval-suite show paper-core
```

比较 `fts / hybrid / tree / auto`：

```bash
uv run python -m kb_agent.cli benchmark paper-core --compare-modes fts,hybrid,tree,auto --top-k 5
uv run python -m kb_agent.cli analyze-failures <benchmark_id>
```

复盘代表性查询：

```bash
uv run python -m kb_agent.cli case-study "这篇论文的方法设计是什么？" --doc-id <doc_id> --compare-modes hybrid,tree
uv run python -m kb_agent.cli eval-dashboard --format html --since-days 7
```

`benchmark` 会生成 `benchmark_report.v1`，覆盖 doc recall、node recall、precision、MRR、表格事实命中率、tree trace 完整度、fallback 和弱解析风险。`analyze-failures` 会生成失败原因和 next actions，帮助判断下一步应该刷新 embedding、重建 PDF 工件、补充表格事实，还是调整评测样例。

## v0.18 事实一致性审计与冲突检测

v0.18 对 `claims / entities / relations / table facts` 做本地规则审计，识别重复、低置信、无证据、跨论文冲突、表格-正文不一致和引用关系缺口。审计报告只保存短摘要、ID、指标和 warning，不保存论文正文、长 excerpt 或完整 evidence packet。

运行事实审计：

```bash
uv run python -m kb_agent.cli audit-facts --doc-id <doc_id> --min-confidence 0.5
uv run python -m kb_agent.cli fact-conflicts --doc-id <doc_id> --severity high
```

审计结果会写入 `data/eval/fact_audit_<id>.json`，并在 `eval-dashboard` 中展示 latest fact audit、冲突数量、高严重度冲突、表格-正文不一致和引用缺口。`compare`、`generate-review` 和 `case-study` 会读取审计摘要，把事实冲突作为风险提示和 open questions，而不会把审计报告当成论文内容证据。

## v0.19 轻量 Claim Graph 与证据链导航

v0.19 将已有 `claims / entities / relations / table facts / fact audit conflicts` 组织成轻量 Claim Graph。图谱作为运行态工件写入 `.kb_state/graphs/<graph_id>/`，包括 `knowledge_graph.json`、`graph_index.json`、`graph_report.json`。图谱节点和边只保存短标签、ID、页码、置信度和来源，不保存论文正文、长 excerpt 或完整 evidence packet。

构建并查看图谱：

```bash
uv run python -m kb_agent.cli graph-build --doc-id <doc_id> --include-conflicts
uv run python -m kb_agent.cli graph-report <graph_id>
uv run python -m kb_agent.cli graph-neighborhood <claim_or_entity_or_conflict_id> --graph-id <graph_id> --depth 2
uv run python -m kb_agent.cli graph-export <graph_id> --format mermaid
uv run python -m kb_agent.cli graph-export <graph_id> --format html
```

`compare`、`generate-review` 和 `case-study` 会读取 Claim Graph 摘要，把共享实体、冲突事实、孤立事实和证据覆盖缺口作为风险提示。正式论文结论仍必须回到 `kb_get_evidence` 的 evidence packet。

## v0.20 真实论文集质量基线与核心能力纠偏

v0.20 不继续横向加新知识层，而是把当前系统放到真实论文集上做总体验收。`quality-baseline` 会同步语料、检查 PDF 解析质量、比较可选 parser 状态、构建 hash embedding、尝试可用的 sentence-transformers、生成 baseline eval suite、运行 `fts/hybrid/tree` benchmark、执行 tree-search、compare/review、case-study、memory eval 和 Claim Graph 风险汇总。

运行默认真实论文集基线：

```bash
uv run --extra pdf python -m kb_agent.cli quality-baseline articles
uv run python -m kb_agent.cli latest-quality-baseline
```

如果已安装 sentence-transformers，可指定真实 embedding 模型：

```bash
uv sync --extra embeddings
uv run python -m kb_agent.cli quality-baseline articles --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

报告会写入 `data/eval/quality_baseline_<id>.json/.md/.html`。HTML 只展示指标、路径、warning 和建议动作，不包含论文正文、长 excerpt、完整 evidence packet 或综述草稿正文。

## v0.21 DeepSeek 论文理解质量增强

v0.21 将 DeepSeek Chat API 用于论文理解质量复测，而不是继续等待真实 embedding API。`quality-baseline --with-llm` 会记录 LLM 状态、规则版与 LLM 版 Tree Search 对比、创新点/事实抽取成功率、compare/review warning 和 evidence coverage。报告仍不保存 API key、完整 prompt、论文正文、长 excerpt 或完整 evidence packet。

检查本地 DeepSeek 配置和连通性：

```bash
uv run python -m kb_agent.cli llm-status
uv run python -m kb_agent.cli llm-status --probe
```

运行 LLM 强制验收：

```bash
uv run python -m kb_agent.cli tree-search <doc_id> "这篇论文的方法设计是什么？" --require-llm
uv run python -m kb_agent.cli extract <doc_id> --force --require-llm
uv run python -m kb_agent.cli extract-facts <doc_id> --force --require-llm
uv run --extra pdf python -m kb_agent.cli quality-baseline articles --with-llm --top-k 3
```

如果 DeepSeek 调用失败，普通 `--with-llm` 会回退规则版并在报告写入 warning；`--require-llm` 会直接失败，避免写入伪成功工件。

## v0.22 DeepSeek 结构化输出稳定性

v0.22 针对真实基线里 `generate-review` 偶发 `DeepSeek returned invalid JSON` 的问题做稳定性增强。JSON 调用会清理 fenced JSON、截取完整 object、剥离尾部噪声，并在解析失败时自动重试一次；错误信息只保留错误类型，不保存模型原文片段。

`generate-review` 会先尝试整体 JSON。如果整体输出超时、截断或非法，会改为逐章节生成小 JSON，再组装 `review_outline.v1`。诊断信息写入 `llm_diagnostics`：

```json
{
  "mode": "full_json | section_json | fallback_rule",
  "retry_count": 0,
  "repair_used": false,
  "fallback_sections": [],
  "error_type": ""
}
```

验收命令：

```bash
uv run python -m kb_agent.cli generate-review "任务规划方法研究综述" --require-llm --search-mode tree
uv run --extra pdf python -m kb_agent.cli quality-baseline articles --with-llm --top-k 3
```

如果整体 JSON 失败但分章节恢复成功，报告会显示 `review_fallback_mode=section_json`，不会再标记为纯 `rule_based_review_plan`。

## v0.23 真实基线可信化与 Tree Search 证据质量

v0.23 聚焦真实 `articles/` 验收，不继续横向增加新模块。`quality-baseline` 会标记 `run_kind`、`corpus_fingerprint` 和 `is_real_corpus`，`latest-quality-baseline` 默认优先展示真实 `articles/` 结果，避免测试 fixture 的临时报告抢占“最新质量”判断。

查看当前真实项目质量：

```bash
uv run --extra pdf python -m kb_agent.cli quality-baseline articles --with-llm --top-k 3
uv run python -m kb_agent.cli latest-quality-baseline --real-only --limit 1
uv run python -m kb_agent.cli latest-quality-baseline --corpus articles --limit 1
```

Tree Search 报告会稳定包含 `query_profile`、`expanded_nodes`、`selected_paths`、`evidence`、`score_components` 和 `selected_reason`，用于判断为什么选中某个章节或证据节点：

```bash
uv run python -m kb_agent.cli search-report "服务机器人任务规划的方法设计" --search-mode tree
```

compare/review 会记录 `duplicate_evidence_removed`、`evidence_quality` 和 `review_partial_reasons`。如果综述仍是 `partial`，报告会区分是语料太少、缺引用关系、证据重复、缺章节证据，还是 LLM/规则降级导致。

事实层也会从 `citation_map.json` 补齐 `cites` 关系，并过滤 `No.`、`Ra`、残缺正文长句这类实体噪声；Claim Graph 报告新增 `noisy_entity_count`，默认不把噪声实体放进 `top_entities`。

## v0.24 DeepSeek 长流程稳定化

v0.24 聚焦 `quality-baseline --with-llm` 的真实可复测性。DeepSeek 调用现在有统一 per-call timeout、JSON retry 次数、baseline 总预算和阶段预算；报告会把 `llm_probe`、`llm_tree_search`、`llm_insights`、`llm_facts`、`llm_compare`、`llm_review` 分别标记为 `completed | partial | skipped | failed | timeout`。

推荐验收命令：

```bash
uv run python -m kb_agent.cli llm-status --probe
uv run --extra pdf python -m kb_agent.cli quality-baseline articles --with-llm --top-k 3 --llm-timeout-seconds 45 --llm-stage-timeout-seconds 120
uv run python -m kb_agent.cli latest-quality-baseline --real-only --limit 1
```

如果 DeepSeek 慢、超时或返回不可解析 JSON，baseline 仍会落盘；`llm_baseline.stage_summary`、`llm_timeout_count`、`llm_budget_exhausted` 和 Markdown/HTML 报告中的 “LLM Runtime” 会说明具体卡在哪个阶段。默认 `--with-llm` 只对前 2 篇论文运行 LLM 阶段，可用 `--llm-max-docs` 调整；如只想复测解析/检索而跳过 compare/review 的 LLM，可加 `--skip-llm-tasks`。

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

也可以使用 OpenAI-compatible 私有网关，例如：

```text
DEEPSEEK_BASE_URL=http://202.117.56.203:3000/v1
DEEPSEEK_MODEL=deepseek_v4
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_TEMPERATURE=0
DEEPSEEK_MAX_TOKENS=3000
DEEPSEEK_TIMEOUT_SECONDS=45
DEEPSEEK_PROBE_TIMEOUT_SECONDS=15
DEEPSEEK_JSON_RETRY_COUNT=1
KB_BASELINE_LLM_TIMEOUT_SECONDS=420
KB_BASELINE_LLM_STAGE_TIMEOUT_SECONDS=120
```

如果 `DEEPSEEK_BASE_URL` 使用 `http://`，密钥会以明文经过该网络链路；只建议在可信内网或临时科研验证环境中使用。

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
kb_sync -> kb_build_semantic_index -> kb_search_docs -> kb_get_doc_card -> kb_get_parse_quality -> kb_get_parse_report -> kb_get_layout_blocks -> kb_get_figures -> kb_get_tables -> kb_get_table_content -> kb_get_table_summaries -> kb_extract_doc_insights -> kb_get_innovations -> kb_get_citation_map -> kb_extract_facts -> kb_get_claims -> kb_get_entities -> kb_get_relations -> kb_get_fact_graph -> kb_fact_search -> kb_audit_facts -> kb_get_fact_conflicts -> kb_build_knowledge_graph -> kb_get_graph_neighborhood -> kb_run_quality_baseline -> kb_get_latest_quality_baseline -> kb_classify_query -> kb_tree_search -> kb_search_tree -> kb_get_evidence -> kb_answer -> kb_compare -> kb_generate_review -> kb_draft_review -> kb_check_review_citations -> kb_assemble_review -> kb_eval_search -> kb_eval_review -> kb_eval_memory -> kb_eval_facts -> kb_create_eval_suite -> kb_run_benchmark -> kb_analyze_failures -> kb_generate_case_study -> kb_get_query_stats -> memory_remember_task -> memory_resume_task -> kb_get_task_artifact
```

当用户明确指出某次结果好坏时，推荐追加：

```text
kb_get_query_log -> kb_put_feedback -> kb_build_eval_set_from_feedback -> kb_eval_search -> kb_eval_dashboard
```

调优检索策略时，推荐追加：

```text
kb_build_eval_set_from_feedback -> kb_eval_search -> kb_tune_search -> kb_apply_search_profile -> search_mode="auto"
```

验证 PageIndex-like 树检索是否有效时，推荐追加：

```text
kb_create_eval_suite -> kb_run_benchmark -> kb_analyze_failures -> kb_generate_case_study -> kb_eval_dashboard
```

## 测试

```bash
uv run python -m unittest discover -s tests
```

## 后续阶段

- 继续增强扫描版 OCR、更完整的图谱可视化和 OpenCode 多智能体工作流。
