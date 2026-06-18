---
name: paper-insight
description: 单篇论文理解 workflow；用于抽取 doc card、创新点、引用、facts、EvidenceUnit、ClaimFrame 和 verifier 状态。
---

# Paper Insight Workflow

## 适用场景

用户要求理解单篇论文、总结创新点、方法、局限、引用关系、事实层或主张证据链时使用。

## 必调工具顺序

1. `kb_get_doc_card`：读取标题、摘要、description、sections 和 quality warnings。
2. `kb_extract_doc_insights`：生成或刷新 innovation 与 citation 工件。
3. `kb_get_innovations` / `kb_get_citation_map`：读取创新点和引用关系状态。
4. `kb_extract_facts`：生成 claims、entities、relations；多篇论文、组会演示或批处理时默认用轻量规则路径，不在 MCP 内部并发跑 LLM。
5. `kb_extract_evidence_units`：从节点、图表、表格、引用工件派生 EvidenceUnit。
6. `kb_extract_claim_frames`：把 facts 和 insight 组织成 ClaimFrame。
7. `kb_verify_claim_frames`：检查 ClaimFrame 到 EvidenceUnit、node、source 的结构链路与语义支持状态。

## 可选工具

- `kb_get_parse_quality` / `kb_get_parse_report`：抽取质量差或结论高风险时调用。
- `kb_get_table_content` / `kb_get_table_summaries`：实验指标或表格事实必须调用。
- `kb_get_claims` / `kb_get_entities` / `kb_get_relations`：需要查看事实层明细时调用。

## 停止条件

- 如果 doc card 或 parse quality 显示弱解析，先报告解析风险，再给出有限结论。
- 如果 `kb_extract_facts` 或 `kb_extract_doc_insights` 的 LLM 路径超时，停止继续批量 LLM 抽取；改用规则工件和短 evidence ID 让当前对话模型辅助分析。
- 对多篇论文不要并发调用 `kb_extract_facts use_llm=true`；确实需要 LLM 抽取时一次只处理一篇，并在超时后回退轻量路径。
- 如果 ClaimFrame `support_status` 为 unsupported，或 `semantic_support_status` 为 `related_only` / `insufficient_evidence` / `contradicted`，不把它当作正式论文结论。

## 输出要求

- 输出论文名片、创新点、方法贡献、局限、引用关系和事实层状态。
- 说明 facts/insights 是规则路径、LLM 路径还是超时后回退路径产出。
- 标明 `innovation.status`、fact status、ClaimFrame `support_status`、`trace_status`、`semantic_support_status` 和 `citation_risk`。
- 对每条关键结论给出 evidence id、node id 或 EvidenceUnit id。

## 禁止事项

- 不对多篇论文并发执行 `kb_extract_facts use_llm=true` 或其他 MCP 内部长 LLM 调用。
- 不因 LLM 抽取超时而连续重试同一篇论文；超时后先保留规则事实和 evidence 链路。
- 不输出无证据支撑的创新点或实验结论。
- 不保存论文正文、长 excerpt、完整 evidence packet 或模型原文。
