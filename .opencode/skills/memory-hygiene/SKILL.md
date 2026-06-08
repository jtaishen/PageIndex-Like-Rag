---
name: memory-hygiene
description: 判断哪些内容应写入长期记忆，哪些只应作为任务工件或知识库资产保存。
---

长期 memory 应写入：

- 用户明确要求记住的信息。
- 稳定、可复用的用户偏好。
- 项目规则、默认目录、引用格式、关注维度。
- 跨 session 仍然有价值的任务状态和 next actions。

长期 memory 不应写入：

- 一次性问答结论。
- 临时检索命中结果。
- 大段论文原文。
- 可从知识库重新检索得到的内容。
- 低置信度猜测。

写入前检查：

1. 是否跨任务可复用。
2. 是否稳定。
3. 是否重要。
4. 是否和已有 memory 重复或冲突。
5. 是否应当作为 `.kb_state` 任务工件，而不是 memory。

工具要求：

- 写入长期记忆必须调用 `memory_put_gated` 或 CLI `memory-put`，不要绕过写入门控。
- 保存任务进度优先调用 `memory_remember_task`，让系统从 `.kb_state/<task_id>/` 压缩生成任务记忆。
- 恢复任务优先调用 `memory_resume_task`，再根据返回的 suggested commands 继续操作。
- 如果内容包含 `node_id`、`page_range`、`excerpt`、`evidence`、论文正文、章节草稿正文，应拒绝写入 memory，保留在知识库工件或任务工件中。
