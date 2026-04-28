"""对双向检索效果做 Recall@K 评测。"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mm_utils import ARTIFACTS_DIR, load_embeddings, topk_from_matrix


def recall_at_k_text_to_image(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    metadata: list[dict],
    k: int,
) -> float:
    """文本到图像采用同类别命中作为正确标准。"""
    hits = 0
    total = len(metadata)
    labels = np.array([item["label_id"] for item in metadata])

    for index, item in enumerate(metadata):
        query_embedding = text_embeddings[index : index + 1]
        retrieved_indices, _scores = topk_from_matrix(query_embedding, image_embeddings, topk=k)
        retrieved_labels = labels[retrieved_indices]
        if item["label_id"] in retrieved_labels:
            hits += 1

    return hits / total if total else 0.0


def recall_at_k_image_to_text_exact(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    k: int,
) -> float:
    """图像到文本沿用一一对齐的 exact match 评测。"""
    hits = 0
    total = image_embeddings.shape[0]

    for index in range(total):
        query_embedding = image_embeddings[index : index + 1]
        retrieved_indices, _scores = topk_from_matrix(query_embedding, text_embeddings, topk=k)
        if index in retrieved_indices:
            hits += 1

    return hits / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测双向检索 Recall@K。")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="需要评测的 K 值列表。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_embeddings, text_embeddings, metadata = load_embeddings()

    report: dict[str, dict[str, float]] = {
        "text_to_image": {},
        "image_to_text_exact": {},
    }

    for k in args.k_values:
        t2i = recall_at_k_text_to_image(image_embeddings, text_embeddings, metadata, k=k)
        i2t = recall_at_k_image_to_text_exact(image_embeddings, text_embeddings, k=k)
        report["text_to_image"][f"R@{k}"] = t2i
        report["image_to_text_exact"][f"R@{k}"] = i2t
        print(f"Text -> Image 的 Recall@{k} = {t2i:.4f}")
        print(f"Image -> Text 的 Exact Recall@{k} = {i2t:.4f}")

    report_path = ARTIFACTS_DIR / "recall_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("评测报告已保存到：", report_path)


if __name__ == "__main__":
    main()
