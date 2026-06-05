---
name: paper-qa
description: 在候选论文中执行树搜索、证据抽取与可追溯问答。
---

使用步骤：

1. 先调用 `kb_search_docs` 获取候选文档。
2. 对每个候选文档调用 `kb_search_tree`，定位章节、段落或图表节点。
3. 调用 `kb_get_evidence` 获取 evidence packet。
4. 仅基于 evidence packet 回答。
5. 若证据不足，明确说明不足，不要补写没有来源的结论。

输出要求：

- 简要回答用户问题。
- 列出关键证据来源，包括文档标题、章节路径和节点 ID。
- 对创新点、实验结论、局限性等高风险结论标注不确定性。

