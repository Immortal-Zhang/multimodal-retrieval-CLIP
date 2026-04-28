"""执行图像到文本检索。"""

from __future__ import annotations

import argparse
from pathlib import Path

from mm_utils import (
    TEXT_FAISS_FILE,
    encode_single_image,
    load_embeddings,
    load_model,
    topk_from_matrix,
    try_search_faiss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据图片检索相似文本。")
    parser.add_argument("--image-path", type=str, required=True, help="输入图片路径。")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数量。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    _image_embeddings, text_embeddings, metadata = load_embeddings()
    model, preprocess, _tokenizer, device = load_model()
    query_embedding = encode_single_image(str(image_path), model, preprocess, device)

    faiss_result = try_search_faiss(query_embedding, TEXT_FAISS_FILE, topk=args.topk)
    if faiss_result is None:
        indices, scores = topk_from_matrix(query_embedding, text_embeddings, topk=args.topk)
    else:
        indices, scores = faiss_result

    print(f"输入图片：{image_path}")
    print("-" * 80)
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(index)]
        print(
            f"[{rank}] score={float(score):.4f} | "
            f"文本={item['caption']} | 类别={item['class_name_cn']}"
        )


if __name__ == "__main__":
    main()
