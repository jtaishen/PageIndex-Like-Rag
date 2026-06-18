---
name: compare-papers
description: 跨论文比较 workflow；用于围绕主题筛选论文、生成比较矩阵、检查证据覆盖和事实风险。
---

# Compare Papers Workflow

## 适用场景

用户要求比较多篇论文的方法、问题设定、实验、创新点、局限、证据强度或冲突时使用。

## 必调工具顺序

1. `kb_search_docs`：围绕比较主题筛选候选论文。
2. `kb_extract_facts`：确保候选论文有 claims、entities、relations；多篇候选论文默认使用轻量规则路径，避免在 MCP 内部并发 LLM 抽取。
3. `kb_extract_claim_frames`：生成主张级证据链。
4. `kb_verify_claim_frames`：检查 ClaimFrame 支撑状态。
5. `kb_compare`：生成固定维度比较矩阵和任务工件；MCP 内部 LLM 超时风险高时先用规则路径生成矩阵，再由当前对话模型基于短工件解释。
6. `kb_audit_facts`：检查重复、低置信、冲突和 citation gaps。

## 可选工具

- `kb_fact_search`：按方法、指标、数据集或局限搜索事实层。
- `kb_get_fact_conflicts`：用户明确问冲突或风险时调用。
- `kb_get_task_artifact`：读取 `comparison_matrix.json`、`open_questions.json` 或 `next_actions.json`，必要时查看 `claim_alignment` 与 `claim_relations`。

## 停止条件

- 如果候选论文少于 2 篇，停止比较并说明需要更多论文。
- 如果 `kb_extract_facts use_llm=true` 或 `kb_compare use_llm=true` 超时，停止继续重试重型 LLM MCP；改用规则 facts / compare 工件，并把 LLM 超时作为限制说明。
- 不对多篇论文并发运行 LLM facts 抽取；确实需要 LLM 增强时先只处理用户确认的关键论文。
- 如果某比较维度缺 evidence，保留 warning，不补写确定结论。

## 输出要求

- 输出六类维度：问题设定、方法范式、数据与评测、创新点重叠、局限与失败模式、证据强度。
- 每个关键比较结论绑定 evidence、EvidenceUnit 或 ClaimFrame。
- 标明比较矩阵来自规则路径、LLM 路径还是 LLM 超时后的规则回退。
- 查看 `comparison_matrix.answer_plan_summary`；只有 strong/qualified claim 可以支撑关键比较结论。
- 查看 `comparison_matrix.claim_alignment_summary`、`method_family_groups`、`conflicting_claim_groups` 和 `research_gap_candidates`；跨论文方法族先按对齐维度解释，结果冲突按 `comparability_checks` 中的 supports/contradicts/incomparable 分类解释，typed relation 只使用技术方案中的轻量关系类型。
- 复盘 claim 对齐时优先查看 `claim_align_score` 及其分项：type_match、field_overlap、subject_similarity、method_family_similarity。
- 报告事实审计 warning、冲突数和证据缺口。

## 禁止事项

- 不并发执行多篇 `kb_extract_facts use_llm=true`。
- 不反复重试超时的 `kb_compare use_llm=true`；先保留 evidence-first 的规则比较结果。
- 不把 fact audit 或 claim graph 当成最终论文证据；正式结论仍需回到 evidence。
- 不把比较矩阵中的短摘要写入长期 memory。
