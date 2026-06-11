---
name: task-resume
description: 任务恢复 workflow；用于从 .kb_state 和长期 memory 恢复综述、比较或质量复盘任务。
---

# Task Resume Workflow

## 适用场景

用户要求继续上次任务、查看当前任务进度、保存任务状态或跨 session 恢复工作时使用。

## 必调工具顺序

1. `memory_resume_task`：读取 current task、最近任务记忆和 suggested commands。
2. `memory_compile_context`：基于 task_id、intent 和 skill_scope 编译 artifact-first 短上下文包。
3. `kb_get_task_artifact`：读取当前任务的 manifest、next_actions、review_report、citation_check 或 comparison matrix。
4. `memory_remember_task`：用户要求保存当前任务进度时调用。

## 可选工具

- `memory_get`：读取与当前任务相关的少量项目偏好。
- `memory_compact`：重复任务记忆过多时调用。
- `kb_eval_review`：恢复综述任务后需要检查引用覆盖时调用。

## 停止条件

- 如果没有 current task，先说明无法恢复，并建议用户提供 task_id 或重新生成任务。
- 如果任务工件缺失，停止继续生成，先说明缺失文件和建议命令。

## 输出要求

- 输出 task_id、任务类型、当前状态、compiled_context 摘要、缺口、next actions 和建议命令。
- 只引用任务工件路径和短状态，不粘贴长草稿或 evidence。

## 禁止事项

- 不用聊天历史替代 `.kb_state` 工件。
- 不把论文正文、长 evidence 或草稿正文写入长期 memory。
