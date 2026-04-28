"""按顺序执行第三周项目的主流程。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行多模态检索项目主流程。")
    parser.add_argument("--skip-prepare-dataset", action="store_true", help="跳过数据准备步骤。")
    parser.add_argument("--skip-build-ann", action="store_true", help="跳过 FAISS 索引构建步骤。")
    parser.add_argument("--samples-per-class", type=int, default=20, help="每个类别抽取的样本数量。")
    parser.add_argument("--image-size", type=int, default=128, help="保存图片的边长。")
    parser.add_argument("--batch-size", type=int, default=32, help="编码批大小。")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True, cwd=BASE_DIR)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable

    if not args.skip_prepare_dataset:
        run_step(
            [
                python_executable,
                "prepare_cifar10_dataset.py",
                "--samples-per-class",
                str(args.samples_per_class),
                "--image-size",
                str(args.image_size),
            ]
        )

    run_step(
        [
            python_executable,
            "encode_data.py",
            "--batch-size",
            str(args.batch_size),
        ]
    )

    if not args.skip_build_ann:
        run_step([python_executable, "build_ann_index.py"])

    run_step([python_executable, "evaluate_recall.py"])


if __name__ == "__main__":
    main()
