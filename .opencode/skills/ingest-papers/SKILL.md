---
name: ingest-papers
description: 论文目录入库与解析质量检查 workflow；用于同步本地论文并判断解析、结构树、图表和后续证据链是否可用。
---

# Ingest Papers Workflow

## 适用场景

用户要求导入论文目录、更新知识库、检查 PDF 解析质量或准备后续问答/比较/综述时使用。

## 必调工具顺序

1. `kb_sync`：同步目录或文件；PDF 可传 `pdf_parser="pypdf"|"docling"|"grobid"|"auto"`。
2. `kb_get_doc_card`：确认标题、摘要、description 和文档路由摘要。
3. `kb_get_parse_quality`：检查 `quality_level`、章节数、表格数、warning。
4. `kb_get_parse_report`：查看 parser chain、fallback、外部解析器失败原因。
5. `kb_get_layout_blocks`：确认版面块和 page-only 风险。

## 可选工具

- `kb_get_figures` / `kb_get_tables` / `kb_get_table_content` / `kb_get_table_summaries`：涉及图表、实验指标或表格证据时调用。
- `kb_build_semantic_index`：用户明确要使用真实或 hash embedding 检索时调用。
- `kb_extract_doc_insights` / `kb_extract_facts` / `kb_extract_evidence_units` / `kb_extract_claim_frames`：用户要后续做单篇理解、比较或综述时调用。

## 停止条件

- 同步失败文件存在时，先报告失败文件和原因，不继续假装可检索。
- 出现 `page_only_tree`、`weak_layout_blocks`、`low_section_count` 或 `scanned_pdf_or_empty_text` 时，明确说明解析限制。

## 输出要求

- 汇总入库数量、失败数量、主要 parser、质量等级和关键 warning。
- 给出下一步建议：重新用 Docling/GROBID、构建 embedding、抽取 facts 或开始问答。

## 禁止事项

- 不把解析失败解释成论文没有相关内容。
- 不自动运行真实 baseline、DeepSeek 长流程或外部解析器重跑，除非用户明确要求。
