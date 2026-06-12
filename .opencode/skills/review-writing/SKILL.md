---
name: review-writing
description: 综述规划与草稿 workflow；用于生成综述大纲、章节证据、章节草稿、引用检查和总稿。
---

# Review Writing Workflow

## 适用场景

用户要求围绕主题生成综述提纲、章节草稿、引用检查、总稿 Markdown 或修改建议时使用。

## 必调工具顺序

1. `kb_generate_review`：生成 review task、selected papers、outline、section evidence。
2. `kb_draft_review`：逐节生成草稿；默认遵守 evidence-first。
3. `kb_check_review_citations`：检查 `[E#]` 引用、missing refs 和 unsupported paragraphs。
4. `kb_assemble_review`：引用检查可接受后再组装总稿。
5. `kb_get_task_artifact`：按需读取 `review_report.json`、`citation_check.json`、`next_actions.json`。

## 可选工具

- `kb_extract_claim_frames` / `kb_verify_claim_frames`：章节证据需要主张级支撑或 citation risk 判断时调用。
- `kb_eval_review`：用户要求复盘综述质量时调用。
- `memory_remember_task`：用户要求跨 session 保存任务进度时调用。

## 停止条件

- 如果 `kb_check_review_citations` 有 unsupported paragraphs，停止组装或明确标记需修改。
- 如果关键 ClaimFrame 的 `citation_risk` 为 `needs_more_evidence` 或 `conflicting_evidence`，停止当作强结论使用并给出修订动作。
- 如果 `review_outline.answer_plan_summary.answerability` 为 `conflicting` 或 `insufficient_evidence`，先处理 open questions，不进入强结论写作。
- 如果 section evidence 不足，保留 open questions，不让模型自由扩写。

## 输出要求

- 汇总 review task id、章节状态、citation coverage、unsupported paragraph 数、missing refs 和关键 citation risk。
- 汇总 `answerability`、strong/qualified/conflicting/insufficient claim 数。
- 给出每节 `section_revision_actions`。
- 总稿路径只作为工件路径返回，不在聊天里粘贴长正文。

## 禁止事项

- 不一次性自由生成完整综述正文。
- 不把草稿正文、长 evidence 或完整 prompt 写入 memory、feedback 或普通报告。
