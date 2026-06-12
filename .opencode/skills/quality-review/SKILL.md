---
name: quality-review
description: 知识库质量复盘 workflow；用于评测检索、baseline、facts、ClaimFrame、memory 和 dashboard。
---

# Quality Review Workflow

## 适用场景

用户要求判断系统效果、检索质量、真实论文集 baseline、embedding 效果、事实层质量或下一步优化方向时使用。

## 必调工具顺序

1. `kb_create_eval_suite` 或 `kb_get_eval_suite`：确认评测集。
2. `kb_run_benchmark`：比较 FTS、hybrid、tree、auto 或 BGE-M3 hybrid。
3. `kb_run_quality_baseline`：用户明确要求真实基线时运行。
4. `kb_eval_dashboard`：生成质量复盘报告。
5. `kb_get_latest_quality_baseline`：读取最新真实 baseline 摘要。

## 可选工具

- `kb_eval_search`：直接跑查询集评测。
- `kb_eval_facts` / `kb_audit_facts`：复盘事实层、ClaimFrame 和 citation gaps。
- `kb_get_task_artifact`：抽查 `comparison_matrix.answer_plan_summary` 或 `review_outline.answer_plan_summary` 的 answerability 分布。
- `kb_eval_memory`：检查 memory 污染、重复和任务恢复。
- `kb_get_query_stats`：查看查询日志趋势。

## 停止条件

- 真实 baseline、DeepSeek、Docling/GROBID 或大规模入库只有用户明确要求时才运行。
- 如果 baseline stale，先报告 stale 原因，不把旧结果当作当前质量。

## 输出要求

- 输出指标、baseline id、报告路径、warning 和 next actions。
- 区分硬问题和背景限制，例如缺证据、unsupported frame、answerability 冲突、small corpus、可选 parser 未启用。

## 禁止事项

- 不复述论文正文、长 evidence、完整 prompt 或综述正文。
- 不把 benchmark/case-study 当作论文内容证据。
