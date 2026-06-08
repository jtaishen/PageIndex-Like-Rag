---
name: ingest-papers
description: 将本地论文或文章目录同步到 PageIndex-like 知识库。
---

使用步骤：

1. 确认用户给出的目录或文件路径。
2. 调用 `kb_sync` 扫描并增量索引目录；PDF 可按需要传入 `pdf_parser` 为 `auto`、`pypdf`、`docling` 或 `grobid`。
3. 如果返回 failed 数量大于 0，提醒用户查看失败文件，常见原因是 PDF 解析依赖未安装。
4. 同步后调用 `kb_get_parse_quality` 和 `kb_get_parse_report` 查看解析质量、解析链和 fallback 原因。
5. 最后可调用 `kb_search_docs` 做一次简单验证。

注意：

- 支持 Markdown、TXT、DOCX、HTML；PDF 默认可用 `pypdf` 兜底，也可选用 Docling 或配置 `GROBID_URL` 做增强。
- 不要把解析失败解释成论文内容不存在。
