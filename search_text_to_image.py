"""执行文本到图像检索。"""

from __future__ import annotations

import argparse

import numpy as np

from mm_utils import (
    IMAGE_FAISS_FILE,
    encode_texts,
    expand_query_variants,
    load_embeddings,
    load_model,
    topk_from_matrix,
    try_search_faiss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据文本查询检索相似图片。")
    parser.add_argument("--query", type=str, required=True, help="输入查询文本。")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数量。")
    parser.add_argument("--use-expansion", action="store_true", help="是否启用查询扩展。")
    return parser.parse_args()


def build_query_embedding(query: str, use_expansion: bool) -> np.ndarray:
    model, _preprocess, tokenizer, device = load_model()
    if not use_expansion:
        return encode_texts([query], model, tokenizer, device)

    variants = expand_query_variants(query)
    print("扩展后的查询：", variants)
    expanded_embeddings = encode_texts(variants, model, tokenizer, device)
    query_embedding = np.mean(expanded_embeddings, axis=0, keepdims=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    return query_embedding


def main() -> None:
    args = parse_args()
    image_embeddings, _text_embeddings, metadata = load_embeddings()
    query_embedding = build_query_embedding(args.query, use_expansion=args.use_expansion)

    faiss_result = try_search_faiss(query_embedding, IMAGE_FAISS_FILE, topk=args.topk)
    if faiss_result is None:
        indices, scores = topk_from_matrix(query_embedding, image_embeddings, topk=args.topk)
    else:
        indices, scores = faiss_result

    print(f"查询：{args.query}")
    print("-" * 80)
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(index)]
        print(
            f"[{rank}] score={float(score):.4f} | "
            f"类别={item['class_name_cn']} | 文本={item['caption']} | 路径={item['image_path']}"
        )


if __name__ == "__main__":
    main()
