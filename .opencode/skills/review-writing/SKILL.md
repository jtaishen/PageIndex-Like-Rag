---
name: review-writing
description: 综述规划与草稿 workflow；用于生成综述大纲、章节证据、章节草稿、引用检查和总稿。
---

# Review Writing Workflow

## 适用场景

用户要求围绕主题生成综述提纲、章节草稿、引用检查、总稿 Markdown 或修改建议时使用。

## 必调工具顺序

1. `kb_search_docs`：先围绕综述主题筛选候选论文，默认控制在 3-5 篇，避免直接启动重型综述生成。
2. `kb_get_doc_card`：逐篇读取标题、摘要、description、sections 和 quality warnings，先向用户列出候选文献。
3. `kb_generate_review`：用户确认候选文献或明确要求继续后，再用确认后的 doc_ids 生成 review task、selected papers、outline、section evidence；如果 MCP 长流程超时风险高，优先用 `use_llm=false` 生成可追溯任务工件，再由当前对话模型基于短工件润色。
4. `kb_get_task_artifact`：先读取 `manifest.json`、`selected_papers.json`、`review_outline.json`、`section_evidence`、`open_questions.json` 和 `next_actions.json`，确认 evidence 覆盖后再进入草稿。
5. `kb_draft_review`：用户明确需要草稿时逐节生成；默认遵守 evidence-first。
6. `kb_check_review_citations`：检查 `[E#]` 引用、missing refs 和 unsupported paragraphs。
7. `kb_assemble_review`：引用检查可接受后再组装总稿。

## 可选工具

- `kb_extract_claim_frames` / `kb_verify_claim_frames`：章节证据需要主张级支撑或 citation risk 判断时调用。
- `kb_eval_review`：用户要求复盘综述质量时调用。
- `kb_get_task_artifact`：需要复核综述结构时读取 `review_outline.json` 中的 `method_lineage`、`limitation_groups` 和 `research_gap_candidates`。
- `memory_remember_task`：用户要求跨 session 保存任务进度时调用。

## 停止条件

- 用户没有确认候选论文时，不直接调用 `kb_generate_review`。
- `kb_generate_review` 使用 LLM 超时时，只重试一次轻量路径：确认 doc_ids、`top_k_docs<=3`、`use_llm=false`；不要连续重试重型 LLM MCP 调用。
- 如果用户只是要“看看能做什么”或组会演示，先展示候选文献、outline、section evidence 和 open questions，不直接生成长草稿。
- 如果 `kb_check_review_citations` 有 unsupported paragraphs，停止组装或明确标记需修改。
- 如果关键 ClaimFrame 的 `citation_risk` 为 `needs_more_evidence` 或 `conflicting_evidence`，停止当作强结论使用并给出修订动作。
- 如果 `review_outline.answer_plan_summary.answerability` 为 `conflicting` 或 `insufficient_evidence`，先处理 open questions，不进入强结论写作。
- 如果 section evidence 不足，保留 open questions，不让模型自由扩写。

## 输出要求

- 先输出候选论文列表：doc_id、标题、摘要/description、解析质量 warning 和推荐/不推荐理由。
- 汇总 review task id、章节状态、citation coverage、unsupported paragraph 数、missing refs 和关键 citation risk。
- 汇总 `answerability`、strong/qualified/conflicting/insufficient claim 数。
- 汇总 `claim_alignment_summary`、`method_lineage`、`evidence_patterns`、`limitation_groups` 和 `research_gap_candidates`，把证据不足的对齐组优先作为研究空白候选，而不是确定结论。
- 给出每节 `section_revision_actions`。
- 总稿路径只作为工件路径返回，不在聊天里粘贴长正文。

## 禁止事项

- 不在未检索候选文献、未读取 doc card、未确认范围时直接调用重型 `kb_generate_review`。
- 不反复用 `use_llm=true` 调用同一个超时综述任务；超时后改用轻量任务工件和短上下文对话润色。
- 不一次性自由生成完整综述正文。
- 不把草稿正文、长 evidence 或完整 prompt 写入 memory、feedback 或普通报告。
