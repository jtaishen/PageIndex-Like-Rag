---
name: paper-qa
description: 在候选论文中执行树搜索、证据抽取与可追溯问答。
---

使用步骤：

1. 先调用 `kb_search_docs` 获取候选文档；默认使用 hybrid，若需要复现旧行为则传入 `search_mode="fts"`。
2. 如果 hybrid 缺少 embedding 或检索效果差，先调用 `kb_build_semantic_index`，再重新检索。
3. 若问题依赖解析可靠性，先调用 `kb_get_parse_quality` 和 `kb_get_parse_report` 查看质量等级、解析链和 fallback warning。
4. 若问题涉及创新点、引用关系、局限性或综述准备，先调用 `kb_extract_doc_insights`，再读取 `kb_get_innovations` / `kb_get_citation_map`。
5. 对普通内容问答，先调用 `kb_classify_query`，再调用 `kb_tree_search` 定位章节、段落或图表节点；如果只需要旧版扁平节点结果，再调用 `kb_search_tree`。
6. 调用 `kb_get_evidence` 获取 evidence packet。
7. 对跨论文比较，调用 `kb_compare` 并读取比较矩阵中的 evidence。
8. 对综述草稿，调用 `kb_generate_review`、`kb_draft_review`、`kb_check_review_citations`，必要时再 `kb_assemble_review`。
9. 仅基于 evidence packet、论文理解工件或任务工件中的 evidence 字段回答。
10. 若证据不足，明确说明不足，不要补写没有来源的结论。
11. 如果用户要求复盘质量，调用 `kb_get_query_stats` 或 `kb_eval_search`；如果是综述任务，调用 `kb_eval_review`。
12. 如果用户指出某次结果正确、错误、证据不足或检索遗漏，调用 `kb_put_feedback` 记录评分、标签、期望 doc/node/keyword 和短评论。
13. 需要沉淀评测集时，调用 `kb_build_eval_set_from_feedback`，再用 `kb_eval_search` 比较 `hybrid/tree/fts`。
14. 需要复盘趋势时，调用 `kb_eval_dashboard`，只使用其中的统计、warning 和建议动作。

输出要求：

- 简要回答用户问题。
- 列出关键证据来源，包括文档标题、章节路径和节点 ID。
- 对创新点、实验结论、局限性等高风险结论标注不确定性。
- 若使用 `innovation.json` 或 `citation_map.json`，说明工件状态是 `extracted` 还是 `partial`。
- 若解析质量是 `weak` 或 `fallback_used` 为 true，说明证据可能受解析质量影响。
- 若使用综述草稿，说明 `citation_check.json` 中是否存在 missing refs 或 unsupported paragraphs。
- 不把论文正文、长 excerpt、evidence packet 或草稿正文写入反馈、日志或 memory。
