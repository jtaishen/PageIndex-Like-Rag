---
name: paper-qa
description: 证据优先论文问答 workflow；用于回答单篇或多篇论文中的具体问题，并要求返回可追溯证据。
---

# Paper QA Workflow

## 适用场景

用户询问论文内容、方法、实验、局限、引用或跨论文事实时使用。若用户要生成综述、比较矩阵或质量报告，改用对应 workflow。

## 必调工具顺序

1. `kb_search_docs`：路由候选论文，默认 `search_mode="hybrid"`；用户要求树检索或证据更强时可用 `search_mode="tree"`。
2. `kb_classify_query`：判断 query intent，决定是否偏向方法、实验、局限、引用或比较。
3. `kb_tree_search`：在候选 doc 内找章节、段落、图表或表格证据节点。
4. `kb_get_evidence`：读取最终 evidence packet。
5. `kb_answer`：只基于 evidence 生成回答；证据不足时必须说明不足。

## 可选工具

- `kb_get_parse_quality` / `kb_get_parse_report`：当证据质量可疑、PDF 弱解析或用户问高风险结论时调用。
- `kb_get_claim_frames` / `kb_verify_claim_frames`：当问题需要主张级证据链、方法贡献或实验结论时调用。
- `kb_get_table_content` / `kb_get_table_summaries`：当问题涉及指标、实验表格或性能对比时调用。

## 停止条件

- 至少拿到一个带 `doc_id`、`node_id`、`node_path` 或 `page_range` 的证据来源后再回答。
- 如果没有足够证据，停止生成结论，改为说明缺口和建议下一步检索。

## 输出要求

- 简要回答用户问题。
- 列出关键证据来源：文档标题、章节路径、节点 ID、页码范围。
- 对创新点、实验结果、局限性等高风险结论标注不确定性。
- 若使用 ClaimFrame，说明 `support_status`、`trace_status` 和相关 `evidence_unit_ids`。

## 禁止事项

- 不基于聊天记忆或模型常识补写论文结论。
- 不把论文正文、长 excerpt、完整 evidence packet 或完整 prompt 写入 memory、feedback、query log 或报告摘要。
