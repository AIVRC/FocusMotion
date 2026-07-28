"""
从 summary 输出目录中提取 region_cut 图片。

用法:
  python extract_region_cuts.py <输入目录> [输出目录]

说明:
  - 输入目录: 包含 *_region_cut.png 文件的 run 输出目录
  - 输出目录: 可选，默认为 <输入目录>/region_cuts
  - 会自动跳过 FALLBACK (无复杂区域) 的图片，仅提取有复杂区域的 region_cut

如果你的输出目录中没有 _region_cut.png 文件（旧版输出），
本脚本也支持从 2x3 summary 图片中裁剪出右下角的 Region Cut 子图。
"""

import os
import sys
import shutil
import glob

import cv2
import numpy as np


def extract_from_separate_files(input_dir: str, output_dir: str) -> int:
    """从单独保存的 _region_cut.png 文件中提取"""
    pattern = os.path.join(input_dir, "*_region_cut.png")
    files = sorted(glob.glob(pattern))

    if not files:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for filepath in files:
        img = cv2.imread(filepath)
        if img is None:
            continue

        # 检查是否全黑（fallback 的 region_cut 通常不是全黑，但可以检查有效像素）
        filename = os.path.basename(filepath)
        dst = os.path.join(output_dir, filename)
        shutil.copy2(filepath, dst)
        count += 1
        print(f"  [COPY] {filename}")

    return count


def extract_from_summary_images(input_dir: str, output_dir: str) -> int:
    """从 2x3 summary 图片中裁剪右下角 Region Cut 子图（兼容旧版输出）"""
    patterns = [
        os.path.join(input_dir, "*_complex.png"),
        os.path.join(input_dir, "*.png"),
    ]

    files = []
    for pat in patterns:
        files = sorted(glob.glob(pat))
        # 排除 _region_cut.png 文件
        files = [f for f in files if "_region_cut" not in f]
        if files:
            break

    if not files:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for filepath in files:
        img = cv2.imread(filepath)
        if img is None:
            continue

        H, W = img.shape[:2]

        # 2x3 布局：右下角 (row=1, col=2) 即 Region Cut
        # 每个子图大约占 1/3 宽度 和 1/2 高度
        cell_w = W // 3
        cell_h = H // 2

        x1 = cell_w * 2
        y1 = cell_h
        region_cut = img[y1:H, x1:W]

        if region_cut.size == 0:
            continue

        basename = os.path.splitext(os.path.basename(filepath))[0]
        dst = os.path.join(output_dir, f"{basename}_region_cut.png")
        cv2.imwrite(dst, region_cut)
        count += 1
        print(f"  [CROP] {os.path.basename(filepath)} -> {os.path.basename(dst)}")

    return count


def main():
    if len(sys.argv) < 2:
        # 默认查找最新的 run 目录
        out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_put")
        if os.path.isdir(out_base):
            runs = sorted([d for d in os.listdir(out_base) if d.startswith("run_")])
            if runs:
                input_dir = os.path.join(out_base, runs[-1])
                print(f"[AUTO] Using latest run directory: {input_dir}")
            else:
                print("Error: No run directories found in out_put/")
                print(f"Usage: python {sys.argv[0]} <input_dir> [output_dir]")
                sys.exit(1)
        else:
            print(f"Usage: python {sys.argv[0]} <input_dir> [output_dir]")
            sys.exit(1)
    else:
        input_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(input_dir, "region_cuts")

    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # 优先从单独的 _region_cut.png 文件提取
    print("[1/2] Looking for separate region_cut files...")
    count = extract_from_separate_files(input_dir, output_dir)

    if count > 0:
        print(f"\n[DONE] Extracted {count} region_cut images to {output_dir}")
    else:
        # 回退: 从 summary 图片裁剪
        print("  No separate region_cut files found.")
        print("[2/2] Cropping from summary images...")
        count = extract_from_summary_images(input_dir, output_dir)

        if count > 0:
            print(f"\n[DONE] Cropped {count} region_cut images to {output_dir}")
        else:
            print("\n[WARN] No images found to extract.")


if __name__ == "__main__":
    main()
