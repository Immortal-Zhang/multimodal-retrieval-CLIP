"""Gradio 双向图文检索演示界面。"""

from __future__ import annotations

import gradio as gr
import numpy as np
import torch
from PIL import Image

from mm_utils import (
    encode_single_image,
    encode_texts,
    expand_query_variants,
    load_embeddings,
    load_model,
    topk_from_matrix,
)


def text_to_image(query: str, topk: int, use_expansion: bool):
    image_embeddings, _text_embeddings, metadata = load_embeddings()
    model, _preprocess, tokenizer, device = load_model()

    if use_expansion:
        variants = expand_query_variants(query)
        text_features = encode_texts(variants, model, tokenizer, device)
        query_embedding = np.mean(text_features, axis=0, keepdims=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    else:
        query_embedding = encode_texts([query], model, tokenizer, device)

    indices, scores = topk_from_matrix(query_embedding, image_embeddings, topk=topk)

    gallery_items = []
    info_lines = []
    for index, score in zip(indices, scores):
        item = metadata[int(index)]
        gallery_items.append((item["image_path"], item["caption"]))
        info_lines.append(
            f"- 分数={float(score):.4f} | 类别={item['class_name_cn']} | 路径={item['image_path']}"
        )

    return gallery_items, "\n".join(info_lines)


@torch.no_grad()
def image_to_text(image, topk: int):
    if image is None:
        return "请上传图片。"

    _image_embeddings, text_embeddings, metadata = load_embeddings()
    model, preprocess, _tokenizer, device = load_model()

    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    query_embedding = encode_single_image_from_pil(pil_image, model, preprocess, device)
    indices, scores = topk_from_matrix(query_embedding, text_embeddings, topk=topk)

    lines = []
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        item = metadata[int(index)]
        lines.append(
            f"[{rank}] 分数={float(score):.4f} | 文本={item['caption']} | 类别={item['class_name_cn']}"
        )
    return "\n".join(lines)


@torch.no_grad()
def encode_single_image_from_pil(pil_image: Image.Image, model, preprocess, device) -> np.ndarray:
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    image_feature = model.encode_image(image_tensor)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    return image_feature.cpu().numpy()


def main() -> None:
    with gr.Blocks(title="双向图文检索演示") as demo:
        gr.Markdown("# 双向图文检索演示")
        gr.Markdown("输入文本可以检索图片，上传图片可以检索相似文本描述。")

        with gr.Tab("文本到图像"):
            query = gr.Textbox(label="输入文本", value="一只猫 cat")
            topk_t2i = gr.Slider(1, 10, value=5, step=1, label="Top K")
            use_expansion = gr.Checkbox(value=False, label="启用查询扩展")
            search_text_button = gr.Button("开始检索")
            gallery = gr.Gallery(label="检索到的图片")
            details = gr.Textbox(label="检索详情", lines=10)
            search_text_button.click(text_to_image, [query, topk_t2i, use_expansion], [gallery, details])

        with gr.Tab("图像到文本"):
            input_image = gr.Image(type="pil", label="上传图片")
            topk_i2t = gr.Slider(1, 10, value=5, step=1, label="Top K")
            search_image_button = gr.Button("开始检索")
            output_text = gr.Textbox(label="检索到的文本", lines=10)
            search_image_button.click(image_to_text, [input_image, topk_i2t], [output_text])

    demo.launch()


if __name__ == "__main__":
    main()
