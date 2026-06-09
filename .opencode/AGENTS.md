# Knowledge Base Agent Rules

- 回答必须优先基于 `kb_get_evidence` 返回的 evidence packet。
- 对创新点、实验结论、局限性等高风险结论必须附上 `doc_id`、`node_id` 或章节路径。
- 如果证据不足，明确说明不足，不要猜测。
- 优先复用已有任务工件和检索结果，不要反复生成冗长摘要。
- 长期 memory 只能保存用户偏好、项目规则、任务状态和跨 session 进度；论文正文、树结构和检索结果属于知识库资产，不写入 memory。
- query log、eval report 和 kb-observer 输出只能用于复盘质量和恢复任务，不能作为论文内容证据。
- benchmark、failure analysis 和 case-study 只保存指标、ID、warning 和短摘要，不能替代 evidence packet。
