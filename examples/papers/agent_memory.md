# Agent Memory for Tool-Use Systems

This note summarizes a small research direction about memory compaction for tool-use agents.

## Problem

Long-running agents accumulate conversation history, tool results, plans, and user preferences. If all of this is stored as long-term memory, later sessions can suffer from memory pollution and context overload.

## Method

The proposed system separates external knowledge from agent memory. Papers, document trees, evidence nodes, and claims stay in the knowledge base. User preferences, project rules, current task state, and next actions can be stored as memory.

The system uses a write gate before saving memory. The gate checks stability, reusability, importance, and confidence. Temporary retrieval results and one-off answers should not be written to long-term memory.

## Evaluation

Useful metrics include memory precision, memory pollution rate, stale memory rate, resume success rate, evidence precision, and node recall.

