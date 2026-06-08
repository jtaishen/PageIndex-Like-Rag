# PageIndex-Like RAG MVP

这是一个基于 OpenCode 的 PageIndex-like 论文知识库智能体 MVP。当前目标是跑通论文入库、结构化工件、章节树检索和 evidence packet 的核心闭环：

```text
parse -> normalize -> tree -> artifacts -> indexes -> evidence packet -> CLI / MCP 工具
```

## 当前能力

- 递归扫描目录，支持增量同步。
- 支持 Markdown、TXT、DOCX、HTML 基础解析。
- PDF 解析通过可选依赖 `pypdf` 支持。
- 解析后生成 `raw_text.txt`、`body.md`、`structured.json`、`metadata.json`、`references.json`、`parse_report.json`、`tree.json`、`node_index.jsonl`、`doc_card.json` 等工件。
- 对中文论文常见结构做规则识别，包括摘要、关键词、第 X 章、`1.1`/`1.1.1` 小节、结论、参考文献、图和表。
- 将文档保存为 `documents` 和 `doc_nodes`。
- 使用 SQLite FTS5 做全文检索。
- 返回带 `doc_id`、`node_id`、`node_path`、页码和 excerpt 的 evidence packet。
- 提供 CLI 命令，包括 `card`、`artifacts`、`quality`、`extract`、`innovations`、`citations`。
- 支持跨论文比较和综述规划任务工件，生成比较矩阵、综述大纲、章节证据表和下一步行动。
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

## PDF 和 MCP 可选依赖

如果要解析 PDF：

```bash
uv sync --extra pdf
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
kb_sync -> kb_search_docs -> kb_get_doc_card -> kb_get_parse_quality -> kb_extract_doc_insights -> kb_get_innovations -> kb_get_citation_map -> kb_search_tree -> kb_get_evidence -> kb_answer -> kb_compare -> kb_generate_review -> kb_draft_review -> kb_check_review_citations -> kb_assemble_review -> kb_get_task_artifact
```

## 测试

```bash
uv run python -m unittest discover -s tests
```

## 后续阶段

- 接入 GROBID / Docling / Marker 提升 PDF 解析质量。
- 加入 memory write gate、TTL、去重、压缩和 OpenCode plugin hook。
