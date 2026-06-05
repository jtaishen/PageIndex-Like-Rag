# PageIndex-Like RAG MVP

这是一个基于 OpenCode 的 PageIndex-like 论文知识库智能体 MVP。第一版目标是先跑通核心闭环：

```text
目录扫描 -> 文档解析 -> SQLite FTS5 索引 -> 简单文档树 -> evidence packet -> CLI / MCP 工具
```

## 当前能力

- 递归扫描目录，支持增量同步。
- 支持 Markdown、TXT、DOCX、HTML 基础解析。
- PDF 解析通过可选依赖 `pypdf` 支持。
- 将文档保存为 `documents` 和 `doc_nodes`。
- 使用 SQLite FTS5 做全文检索。
- 返回带 `doc_id`、`node_id`、`node_path`、页码和 excerpt 的 evidence packet。
- 提供 CLI 命令。
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
kb_sync -> kb_search_docs -> kb_search_tree -> kb_get_evidence -> kb_answer
```

## 测试

```bash
python3 -m unittest discover -s tests
```

## 后续阶段

- 接入 GROBID / Docling / Marker 提升 PDF 解析质量。
- 增加 doc_card、innovation.json、citation_map.json。
- 增加跨论文比较和综述任务工件。
- 加入 memory write gate、TTL、去重、压缩和 OpenCode plugin hook。
