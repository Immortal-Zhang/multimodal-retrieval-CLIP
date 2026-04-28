"""下载并整理 CIFAR-10 样本，生成图文对元数据。"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd
from torchvision.datasets import CIFAR10

from mm_utils import DATA_DIR, IMAGES_DIR


CLASS_MAP = {
    0: ("airplane", "飞机"),
    1: ("automobile", "汽车"),
    2: ("bird", "鸟"),
    3: ("cat", "猫"),
    4: ("deer", "鹿"),
    5: ("dog", "狗"),
    6: ("frog", "青蛙"),
    7: ("horse", "马"),
    8: ("ship", "船"),
    9: ("truck", "卡车"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CIFAR-10 图文对数据集。")
    parser.add_argument("--samples-per-class", type=int, default=20, help="每个类别抽取的样本数量。")
    parser.add_argument("--image-size", type=int, default=128, help="保存图片的边长。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR10(root=str(DATA_DIR / "cifar_cache"), train=True, download=True)

    counters: defaultdict[int, int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for image, label in dataset:
        if counters[label] >= args.samples_per_class:
            continue

        class_name_en, class_name_cn = CLASS_MAP[label]
        save_name = f"{class_name_en}_{counters[label]:04d}.png"
        save_path = IMAGES_DIR / save_name

        image = image.resize((args.image_size, args.image_size))
        image.save(save_path)

        rows.append(
            {
                "image_path": str(save_path),
                "class_name_en": class_name_en,
                "class_name_cn": class_name_cn,
                "caption": f"一张{class_name_cn}的图片 {class_name_en}",
                "label_id": label,
            }
        )
        counters[label] += 1

        enough = len(counters) == len(CLASS_MAP) and all(
            count >= args.samples_per_class for count in counters.values()
        )
        if enough:
            break

    dataframe = pd.DataFrame(rows)
    output_path = DATA_DIR / "captions.csv"
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("图文对数据已生成：", output_path)
    print("总样本数：", len(dataframe))
    print(dataframe.head())


if __name__ == "__main__":
    main()
