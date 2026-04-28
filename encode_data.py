"""提取图像和文本的 OpenCLIP 向量，并保存到 artifacts 目录。"""

from __future__ import annotations

import argparse

from mm_utils import (
    encode_images,
    encode_texts,
    load_metadata,
    load_model,
    save_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="编码图像和文本数据。")
    parser.add_argument("--batch-size", type=int, default=32, help="编码批大小。")
    parser.add_argument("--model-name", type=str, default="ViT-B-32", help="OpenCLIP 模型名。")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP 预训练权重名。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata()
    model, preprocess, tokenizer, device = load_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    image_paths = metadata["image_path"].tolist()
    captions = metadata["caption"].tolist()

    print("开始编码图像...")
    image_embeddings = encode_images(
        image_paths,
        model,
        preprocess,
        device,
        batch_size=args.batch_size,
    )

    print("开始编码文本...")
    text_embeddings = encode_texts(
        captions,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
    )

    output_path = save_embeddings(image_embeddings, text_embeddings, metadata)
    print("向量文件已保存到：", output_path)


if __name__ == "__main__":
    main()
