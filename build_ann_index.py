"""尝试构建 FAISS 索引，用于加速近似最近邻搜索。"""

from __future__ import annotations

from pathlib import Path

from mm_utils import (
    ARTIFACTS_DIR,
    IMAGE_FAISS_FILE,
    TEXT_FAISS_FILE,
    load_embeddings,
    try_build_faiss_index,
)


def main() -> None:
    image_embeddings, text_embeddings, _metadata = load_embeddings()

    image_ok = try_build_faiss_index(image_embeddings, IMAGE_FAISS_FILE)
    text_ok = try_build_faiss_index(text_embeddings, TEXT_FAISS_FILE)

    note_path = ARTIFACTS_DIR / "ann_build_note.txt"
    with open(note_path, "w", encoding="utf-8") as file:
        if image_ok and text_ok:
            file.write("FAISS 索引构建成功。\n")
        else:
            file.write(
                "当前环境没有成功构建 FAISS 索引。"
                "这不是致命问题，因为 search_text_to_image.py 和 search_image_to_text.py "
                "仍然可以使用暴力相似度搜索完成实验。\n"
            )

    print(note_path.read_text(encoding="utf-8"))
    print("输出目录：", Path(ARTIFACTS_DIR).resolve())


if __name__ == "__main__":
    main()
