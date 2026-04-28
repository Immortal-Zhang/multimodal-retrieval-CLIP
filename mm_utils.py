"""多模态图文检索项目的公共工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CAPTIONS_FILE = DATA_DIR / "captions.csv"
EMBEDDINGS_FILE = ARTIFACTS_DIR / "embeddings.npz"
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"
IMAGE_FAISS_FILE = ARTIFACTS_DIR / "image_faiss.index"
TEXT_FAISS_FILE = ARTIFACTS_DIR / "text_faiss.index"

for path in (DATA_DIR, IMAGES_DIR, ARTIFACTS_DIR):
    path.mkdir(parents=True, exist_ok=True)


CLASS_SYNONYMS = {
    "飞机": ["飞机", "飞行器", "airplane", "plane"],
    "汽车": ["汽车", "小汽车", "car", "automobile"],
    "鸟": ["鸟", "小鸟", "bird"],
    "猫": ["猫", "猫咪", "cat", "kitty"],
    "鹿": ["鹿", "deer"],
    "狗": ["狗", "小狗", "dog", "puppy"],
    "青蛙": ["青蛙", "frog"],
    "马": ["马", "horse"],
    "船": ["船", "轮船", "ship", "boat"],
    "卡车": ["卡车", "truck"],
}


def get_device() -> str:
    """返回当前可用的推理设备。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_metadata() -> pd.DataFrame:
    """读取图文对元数据。"""
    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError("没有找到 data/captions.csv，请先运行 prepare_cifar10_dataset.py")
    return pd.read_csv(CAPTIONS_FILE)


def load_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> tuple[Any, Any, Any, str]:
    """加载 OpenCLIP 模型、预处理函数、分词器和设备信息。"""
    import open_clip

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """批量编码文本，并返回归一化后的向量。"""
    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def encode_images(
    image_paths: list[str],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """批量编码图像，并返回归一化后的向量。"""
    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        image_tensors = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensors.append(preprocess(image))
        image_tensor = torch.stack(image_tensors).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(image_features.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def encode_single_image(image_path: str, model: Any, preprocess: Any, device: str) -> np.ndarray:
    """编码单张图像，返回形状为 [1, D] 的向量。"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    image_feature = model.encode_image(image_tensor)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    return image_feature.cpu().numpy()


def save_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    save_path: Path | None = None,
) -> Path:
    """保存图像向量、文本向量和对应元数据。"""
    target = save_path or EMBEDDINGS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )
    metadata.to_json(METADATA_FILE, force_ascii=False, orient="records", indent=2)
    return target


def load_embeddings() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """读取向量文件和元数据。"""
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("请先运行 encode_data.py 生成向量文件。")
    payload = np.load(EMBEDDINGS_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return payload["image_embeddings"], payload["text_embeddings"], metadata


def topk_from_matrix(
    query_embedding: np.ndarray,
    database_embeddings: np.ndarray,
    topk: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """基于点积相似度做暴力检索。"""
    scores = np.asarray(query_embedding @ database_embeddings.T).reshape(-1)
    k = min(topk, scores.shape[0])
    indices = np.argsort(scores)[::-1][:k]
    return indices, scores[indices]


def try_build_faiss_index(embeddings: np.ndarray, save_path: Path) -> bool:
    """尝试构建 FAISS 内积索引；若 faiss 不可用则返回 False。"""
    try:
        import faiss
    except Exception:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    faiss.write_index(index, str(save_path))
    return True


def try_search_faiss(
    query_embedding: np.ndarray,
    index_path: Path,
    topk: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """尝试使用 FAISS 搜索；若索引或依赖不存在则返回 None。"""
    try:
        import faiss
    except Exception:
        return None

    if not index_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    scores, indices = index.search(query_embedding.astype("float32"), topk)
    return indices[0], scores[0]


def expand_query_variants(query: str) -> list[str]:
    """基于类别同义词表扩展短查询。"""
    variants = [query]
    for synonyms in CLASS_SYNONYMS.values():
        if any(token in query for token in synonyms):
            variants.extend(synonyms)
    variants = [item.strip() for item in variants if item.strip()]
    return list(dict.fromkeys(variants))
