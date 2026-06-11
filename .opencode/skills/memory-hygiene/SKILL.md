---
name: memory-hygiene
description: 长期 memory 写入与任务恢复 workflow；用于判断哪些内容可记忆、哪些必须留在知识库或任务工件中。
---

# Memory Hygiene Workflow

## 适用场景

用户要求记住偏好、保存项目规则、恢复任务、压缩任务进度或清理 memory 时使用。

## 必调工具顺序

1. `memory_put_gated`：写入用户偏好、项目规则、默认目录、引用格式或稳定任务状态。
2. `memory_resume_task`：恢复最近任务和建议下一步动作。
3. `memory_compile_context`：为当前 query、intent、task_id 和 skill_scope 编译短上下文包。
4. `memory_remember_task`：从 `.kb_state/<task_id>/` 压缩写入任务进度。
5. `memory_compact`：合并重复或过期的任务记忆。

## 可选工具

- `memory_get`：只读取与当前任务相关的少量 memory。
- `kb_get_task_artifact`：恢复任务时读取真实任务工件，不依赖聊天历史。
- `kb_eval_memory`：复盘 memory 污染、重复和恢复可用性。

## 可写入长期 memory

- 用户明确要求记住的信息。
- 稳定、可复用的用户偏好。
- 项目规则、默认目录、引用格式、关注维度。
- 跨 session 仍有价值的任务状态和 next actions。

## 不可写入长期 memory

- 论文正文、长 excerpt、完整 evidence packet。
- 临时检索结果、一次性问答结论。
- 综述草稿正文、完整 prompt、模型原文。
- 可从知识库或任务工件重新读取的内容。

## 输出要求

- 说明 memory 写入结果：accepted、rejected 或 merged。
- 恢复任务时返回 task_id、当前状态、compiled_context 摘要、缺口和建议命令。

## 禁止事项

- 不绕过 `memory_put_gated` 直接写长期 memory。
- 不把知识库资产当成用户长期记忆。
