# 文件说明

## 根目录文件

### `README.md`
项目主页。说明项目目标、运行方式、方法流程。

### `FILE_GUIDE.md`
当前这份文件。快速说明每个脚本和目录的作用。

### `PROJECT_RELEASE_NOTES.md`
从教学脚本版整理成项目版时做了哪些修改，都写在这里。

### `requirements.txt`
核心依赖列表。

### `.gitignore`
Git 忽略规则。用于排除数据图片、向量文件、索引文件、缓存和本地环境目录。

## 核心脚本

### `prepare_cifar10_dataset.py`
下载并整理 CIFAR-10 数据，抽取固定数量样本，生成 `data/captions.csv` 和 `data/images/` 下的图片。

### `encode_data.py`
加载 OpenCLIP，对图片和文本分别做编码，并将结果保存到 `artifacts/embeddings.npz` 和 `artifacts/metadata.json`。

### `search_text_to_image.py`
根据输入文本执行文本到图像检索。支持 `--use-expansion` 开启查询扩展。

### `search_image_to_text.py`
根据输入图片执行图像到文本检索。

### `build_ann_index.py`
尝试构建 FAISS 索引。成功时会生成图像索引和文本索引；失败时主流程仍然可以使用暴力搜索。

### `evaluate_recall.py`
计算 Recall@K 指标，并将结果保存到 `artifacts/recall_report.json`。

### `query_expansion_demo.py`
对比原始查询和扩展查询在文本到图像检索中的差异。

### `run_pipeline.py`
一键运行主流程：准备数据、提取向量、构建索引、计算 Recall@K。

### `app.py`
Gradio 页面，支持双向图文检索演示。

### `mm_utils.py`
公共工具模块，包含路径常量、模型加载、图文编码、暴力检索、FAISS 索引和查询扩展等函数。

## 文档目录

### `docs/project_report.md`
项目报告模板。可直接作为你简历项目的配套说明文档。

## 数据目录

### `data/images/`
CIFAR-10 图片保存目录。默认不建议上传到 GitHub。

### `data/captions.csv`
图文对元数据。默认不建议上传到 GitHub。

## 结果目录

### `artifacts/embeddings.npz`
图像向量与文本向量的压缩文件。

### `artifacts/metadata.json`
与向量文件对齐的元数据。

### `artifacts/image_faiss.index`
图像向量的 FAISS 索引。

### `artifacts/text_faiss.index`
文本向量的 FAISS 索引。

### `artifacts/recall_report.json`
Recall@K 的评测结果。

### `artifacts/ann_build_note.txt`
FAISS 索引构建状态说明。
