# 项目版整理说明

这份目录由原始的第三周教学脚本整理而来，目标是把“练习脚本集合”改成“可上传 GitHub 的项目版仓库”。

## 这次整理做了什么

### 1. 去掉了编号式脚本名

原始文件名：

- `00_prepare_cifar10_dataset.py`
- `01_encode_data.py`
- `02_text_to_image.py`
- `03_image_to_text.py`
- `04_build_ann_index.py`
- `05_evaluate_recall.py`
- `06_query_expansion_demo.py`

项目版统一改成：

- `prepare_cifar10_dataset.py`
- `encode_data.py`
- `search_text_to_image.py`
- `search_image_to_text.py`
- `build_ann_index.py`
- `evaluate_recall.py`
- `query_expansion_demo.py`

这样做的目的是减少“教程感”，增强“项目感”。

### 2. 删除了大段提醒类中文注释

原始版本里存在很多类似“目标”“运行方式”“请先运行上一步”的提醒类注释。项目版删除了这类教学提示，只保留简短 docstring 和必要的设计说明。

### 3. 增加了统一入口脚本

新增 `run_pipeline.py`，可以按顺序执行：

- 数据准备
- 向量提取
- 索引构建
- Recall@K 评测

### 4. 增加了项目文档

新增：

- `FILE_GUIDE.md`
- `PROJECT_RELEASE_NOTES.md`
- `docs/project_report.md`

### 5. 增加了更适合 GitHub 的文件

新增：

- `requirements.txt`
- `.gitignore`

用于控制依赖安装和仓库清洁度。

## 为什么不建议保留原来的提醒注释

原来的提醒类注释适合学习阶段，不适合放在简历项目仓库里。项目版代码里建议只保留：

- 文件级 docstring
- 函数级 docstring
- 非显然设计选择的简短说明

而不保留“像上课讲义一样”的提示语。

## 如果你已经跑过旧版脚本

可以直接把旧版目录里已经生成的这些文件复制到项目版目录中，避免重复生成：

- `data/images/`
- `data/captions.csv`
- `artifacts/embeddings.npz`
- `artifacts/metadata.json`
- `artifacts/image_faiss.index`
- `artifacts/text_faiss.index`
