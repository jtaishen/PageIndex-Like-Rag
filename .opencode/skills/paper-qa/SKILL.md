---
name: paper-qa
description: 在候选论文中执行树搜索、证据抽取与可追溯问答。
---

使用步骤：

1. 先调用 `kb_search_docs` 获取候选文档。
2. 若问题涉及创新点、引用关系、局限性或综述准备，先调用 `kb_extract_doc_insights`，再读取 `kb_get_innovations` / `kb_get_citation_map`。
3. 对普通内容问答，调用 `kb_search_tree` 定位章节、段落或图表节点。
4. 调用 `kb_get_evidence` 获取 evidence packet。
5. 对跨论文比较，调用 `kb_compare` 并读取比较矩阵中的 evidence。
6. 对综述草稿，调用 `kb_generate_review`、`kb_draft_review`、`kb_check_review_citations`，必要时再 `kb_assemble_review`。
7. 仅基于 evidence packet、论文理解工件或任务工件中的 evidence 字段回答。
8. 若证据不足，明确说明不足，不要补写没有来源的结论。

输出要求：

- 简要回答用户问题。
- 列出关键证据来源，包括文档标题、章节路径和节点 ID。
- 对创新点、实验结论、局限性等高风险结论标注不确定性。
- 若使用 `innovation.json` 或 `citation_map.json`，说明工件状态是 `extracted` 还是 `partial`。
- 若使用综述草稿，说明 `citation_check.json` 中是否存在 missing refs 或 unsupported paragraphs。
