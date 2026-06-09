---
name: ingest-papers
description: 将本地论文或文章目录同步到 PageIndex-like 知识库。
---

使用步骤：

1. 确认用户给出的目录或文件路径。
2. 调用 `kb_sync` 扫描并增量索引目录；PDF 可按需要传入 `pdf_parser` 为 `auto`、`pypdf`、`docling` 或 `grobid`。
3. 如果返回 failed 数量大于 0，提醒用户查看失败文件，常见原因是 PDF 解析依赖未安装。
4. 同步后调用 `kb_get_parse_quality`、`kb_get_parse_report` 和 `kb_get_layout_blocks` 查看解析质量、解析链、fallback 原因和版面块数量。
5. 如果 PDF 出现 `page_only_tree`、`weak_layout_blocks`、`low_section_count` 或章节/图表识别弱，优先建议重新运行 `kb_sync(..., force=True, pdf_parser="docling")`；Docling 不可用时再回退 `pypdf` 并明确说明质量限制。
6. 如果后续要做问答、比较或综述，调用 `kb_build_semantic_index` 构建默认 hash embedding。
7. 最后可调用 `kb_search_docs` 做一次简单验证；需要复现旧检索行为时传入 `search_mode="fts"`。

注意：

- 支持 Markdown、TXT、DOCX、HTML；PDF 默认可用 `pypdf` 兜底，也可选用 Docling 或配置 `GROBID_URL` 做增强。
- v0.14 会生成 `layout_blocks.json`、`figures.json`、`tables.json` 和 `reference_sections.json`；需要检查图表结构时优先读取 `kb_get_figures` 和 `kb_get_tables`。
- 扫描版 PDF 暂不做 OCR；看到 `scanned_pdf_or_empty_text` 时直接说明当前解析能力边界。
- 默认语义索引使用离线 hash provider，不会下载模型；sentence-transformers 只有在显式配置时使用。
- 不要把解析失败解释成论文内容不存在。
