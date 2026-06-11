---
name: project-code-quality
description: 在本项目中编写、重构、测试或提交代码时使用。要求代码风格简洁清晰、避免冗余、保持职责边界、保护运行态资产和密钥，并按中文详细提交规范进行 Git 操作。
---

# Project Code Quality

在本项目做代码改动时，优先保证代码可读、可测、可维护。不要为了“完成一个版本号”横向堆功能；先确认改动是否让真实论文知识库链路更稳、更清楚、更容易验证。

最重要的是把用户当前要求的功能点真正实现好。降级、fallback、warning 和 partial 状态只是可靠性边界，不能替代核心功能本身；不要把主要精力放在“失败后怎么兜底”而忽略“成功路径是否真的可用、效果是否足够好”。

## 工作原则

1. 先读现有实现，再动手修改。
2. 保持改动小而完整，避免顺手重构无关模块。
3. 优先复用项目已有函数、数据结构、工件格式和 CLI/MCP 风格。
4. 新增抽象必须减少真实重复、降低复杂度或明确模块边界。
5. 不把问题隐藏到更大的 helper 里；helper 名称必须说明语义边界。
6. 不为了让测试通过而伪造成功状态；失败、降级、timeout 和 partial 都要显式暴露。
7. 不运行长流程、真实 LLM baseline、外部解析器或大规模任务，除非用户明确要求。
8. 先完成主路径能力，再补失败兜底；fallback 不能成为“功能未完成”的遮羞布。

## 代码风格

- 使用简单直接的 Python 标准库实现，除非项目已有依赖或需求明确要求引入新依赖。
- 函数只做一件主要事情；复杂流程拆成有名字的阶段函数。
- 避免复制粘贴相似逻辑；抽公共函数前先确认调用方语义一致。
- 对 `read_json`、`string_list`、`excerpt`、dedupe、fallback 等宽松工具保持谨慎，不能用于需要严格失败的路径。
- 命名要表达业务含义，例如 `review_draft_skip_reason` 优于模糊的 `reason2`。
- 注释只解释不显然的业务约束、降级策略或安全边界，不重复代码字面意思。
- 输出报告、日志和 memory 时只保存 ID、指标、短摘要、warning、路径和状态，不保存论文正文、长 excerpt、完整 prompt、API key 或完整 evidence packet。

## 模块边界

- `cli.py` 只保留入口。
- `cli_parser.py` 只定义参数，不执行业务逻辑。
- `cli_handlers.py` 只做命令分发和轻量输出，不承载核心算法。
- `quality_baseline.py` 负责 baseline 编排；runtime 统计放 `baseline_runtime.py`；Markdown/HTML 输出放 `baseline_renderers.py`。
- `tasks.py` 负责 compare/review 任务工件；不要把解析、DB schema 或低层检索逻辑继续塞进去。
- `facts.py` 负责事实抽取和事实入库；事实审计、图谱、评测保持在各自模块。
- `parsers.py` 和 `ingest.py` 的边界要清楚：parser 产出规范化解析结果，ingest 负责入库、工件和索引更新。
- 如果单文件继续超过约 1200 行，优先考虑按职责拆分，而不是继续追加私有函数。

## 测试与验证

改动验证按风险选择：

- 小型纯函数或 CLI parser 改动：运行相关单测。
- 检索、事实、综述、baseline 改动：运行相关单测加回归单测。
- 跨模块行为改动：运行 `uv run python -m unittest discover -s tests`。
- 提交前至少运行 `git diff --check`。
- LLM、Docling、GROBID、真实 `articles/` baseline 属于长流程或环境依赖流程，只在用户明确要求时运行。

测试要求：

- 新增功能必须有稳定结构断言，不只断言“没有报错”。
- 测试要覆盖主成功路径的真实产出质量，而不只是 fallback、异常和空结果。
- 降级路径、失败路径、partial 状态和 warning 要有测试。
- 报告、日志、dashboard、memory 的测试必须确认不包含密钥、论文正文、长 evidence、完整 prompt 或草稿正文。
- 不把 `data/eval/`、`data/parsed/`、`.kb_state/` 的运行态输出作为常规测试 fixture 提交。

## 安全与资产边界

- `.env`、DeepSeek key、API token、私有 base URL 凭据不提交。
- 论文原文、长 excerpt、完整 evidence packet 和综述正文属于知识库或任务工件，不写入长期 memory、query log、feedback 或普通报告摘要。
- Git 中默认只提交代码、测试、文档和必要配置；运行态数据默认不提交。
- 修改 `.gitignore` 前先确认不会误纳入 `data/parsed`、`.kb_state`、`data/eval` 或本地缓存。

## Git 规范

提交前：

1. 运行 `git status --short --branch`，确认是否有用户未提交改动。
2. 只 stage 本次任务相关文件，不顺手纳入无关改动。
3. 查看 `git diff --cached`，确认没有密钥、运行态大文件或无关格式化。
4. 运行合适测试和 `git diff --check`。

提交信息必须使用中文，并尽量详细说明：

- 做了什么。
- 为什么要做。
- 对现有功能有什么影响。
- 如何验证。

推荐格式：

```text
实现/修复/重构 <简短主题>

做了什么：
- ...

为什么：
- ...

影响：
- ...

验证：
- ...
```

只有用户明确要求“提交”时才 commit；只有用户明确要求“推送”时才 push。

## 完成前自检

- 代码是否比修改前更清楚，而不是只换了位置。
- 用户要求的核心功能是否真的实现并能产出有效结果。
- 是否引入了新的重复逻辑或过宽 helper。
- 是否破坏现有工件 schema、CLI 参数或 MCP 返回结构。
- 失败和降级是否可见。
- 是否有对应测试或明确说明未运行测试的原因。
- 是否遵守 evidence-first、memory 隔离和运行态资产边界。
