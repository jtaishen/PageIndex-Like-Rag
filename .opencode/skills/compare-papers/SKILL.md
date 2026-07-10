---
name: compare-papers
description: 跨论文比较 workflow；用于围绕主题筛选论文、生成比较矩阵、检查证据覆盖和事实风险。
---

# Compare Papers Workflow

## 适用场景

用户要求比较多篇论文的方法、问题设定、实验、创新点、局限、证据强度或冲突时使用。

## 必调工具顺序

1. `kb_search_docs`：围绕比较主题筛选候选论文。
2. `kb_prepare_fact_extraction -> kb_extract_fact_batch -> kb_finalize_fact_extraction`：候选论文缺少 facts 时逐篇、逐批完成抽取，不并发调用 LLM。
3. `kb_extract_claim_frames`：生成主张级证据链。
4. `kb_verify_claim_frames`：检查 ClaimFrame 支撑状态。
5. `kb_prepare_compare`：生成固定六维的比较任务和 evidence，不调用 LLM。
6. `kb_get_workflow_status` / `kb_generate_compare_dimension`：按 pending steps 一次生成一个比较维度。
7. `kb_finalize_compare`：所有维度完成后合并 comparison matrix。
8. `kb_audit_facts`：检查重复、低置信、冲突和 citation gaps。

## 可选工具

- `kb_fact_search`：按方法、指标、数据集或局限搜索事实层。
- `kb_get_fact_conflicts`：用户明确问冲突或风险时调用。
- `kb_get_task_artifact`：读取 `comparison_matrix.json`、`open_questions.json` 或 `next_actions.json`，必要时查看 `claim_alignment` 与 `claim_relations`。

## 停止条件

- 如果候选论文少于 2 篇，停止比较并说明需要更多论文。
- 单个 fact batch 或 comparison dimension 超时时，只重试失败 step，不重跑已完成部分。
- 不对多篇论文并发运行 LLM facts 抽取；按论文依次完成 workflow。
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

- 交互式 OpenCode workflow 不直接调用旧的 `kb_extract_facts use_llm=true` 或 `kb_compare use_llm=true` 长流程。
- 不跳过失败维度后把比较矩阵标记为完整。
- 不把 fact audit 或 claim graph 当成最终论文证据；正式结论仍需回到 evidence。
- 不把比较矩阵中的短摘要写入长期 memory。
