"""比较有无查询扩展时的文本到图像检索结果。"""

from __future__ import annotations

import argparse

import numpy as np

from mm_utils import (
    encode_texts,
    expand_query_variants,
    load_embeddings,
    load_model,
    topk_from_matrix,
)


def print_results(title: str, indices, scores, metadata) -> None:
    print(f"\n{title}")
    print("-" * 80)
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(index)]
        print(
            f"[{rank}] score={float(score):.4f} | "
            f"类别={item['class_name_cn']} | 文本={item['caption']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较查询扩展前后的检索结果。")
    parser.add_argument("--query", type=str, required=True, help="输入查询文本。")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数量。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_embeddings, _text_embeddings, metadata = load_embeddings()
    model, _preprocess, tokenizer, device = load_model()

    plain_embedding = encode_texts([args.query], model, tokenizer, device)
    plain_indices, plain_scores = topk_from_matrix(plain_embedding, image_embeddings, topk=args.topk)
    print_results("原始查询结果", plain_indices, plain_scores, metadata)

    variants = expand_query_variants(args.query)
    expanded_embeddings = encode_texts(variants, model, tokenizer, device)
    query_embedding = np.mean(expanded_embeddings, axis=0, keepdims=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    expanded_indices, expanded_scores = topk_from_matrix(query_embedding, image_embeddings, topk=args.topk)
    print("扩展后的查询列表：", variants)
    print_results("扩展后查询结果", expanded_indices, expanded_scores, metadata)


if __name__ == "__main__":
    main()
