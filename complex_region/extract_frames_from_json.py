#!/usr/bin/env python3
"""
从 JSON 元数据文件中提取视频帧并保存为图片。

功能：
  - 读取 JSON 元数据（包含 video_path, idx, img_path 字段）
  - 根据 video_path 和 idx 从视频文件中提取指定帧
  - 保存的图片尺寸与原视频帧完全一致（不做任何缩放）
  - 支持多个 JSON 文件批量处理
  - 支持自定义输出目录

使用方法：
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json --output ./extracted_frames
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json ../meta3/fd_2000_meta.json --output ./extracted_frames
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json  # 默认输出到 ./extracted_frames
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 配置
# ============================================================

# 视频搜索的备用目录列表（当 JSON 中的相对路径找不到视频时依次尝试）
FALLBACK_VIDEO_DIRS = [
    "../meta2/Tik_meta2.json",
    # "/home/yanghaotian/server_data/yanghaotian/digital_virtual/data/train",
]

DEFAULT_OUTPUT_DIR = "./extracted_frames"


# ============================================================
# 核心函数
# ============================================================

def resolve_video_path(video_rel_path: str, json_dir: str) -> Optional[str]:
    """
    解析视频文件的实际路径。

    优先使用 JSON 文件所在目录拼接的相对路径，
    如果不存在则依次在备用目录中查找同名文件。

    Args:
        video_rel_path: JSON 中记录的视频相对路径
        json_dir: JSON 文件所在目录的绝对路径

    Returns:
        视频文件的绝对路径，找不到则返回 None
    """
    # 1. 相对于 JSON 所在目录的路径
    vp = os.path.normpath(os.path.join(json_dir, video_rel_path))
    if os.path.exists(vp):
        return vp

    # 2. 在备用目录中查找
    video_basename = os.path.basename(video_rel_path)
    for fallback_dir in FALLBACK_VIDEO_DIRS:
        alt = os.path.join(fallback_dir, video_basename)
        if os.path.exists(alt):
            return alt

    return None


def read_frame(video_path: str, idx: int) -> Optional[np.ndarray]:
    """
    从视频文件中读取指定帧。

    Args:
        video_path: 视频文件路径
        idx: 帧索引（从 0 开始）

    Returns:
        BGR 格式的帧图像 (numpy array)，失败时返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_info(video_path: str) -> Tuple[int, int, int]:
    """
    获取视频基本信息。

    Args:
        video_path: 视频文件路径

    Returns:
        (width, height, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (w, h, total)


def load_json_metadata(json_path: str) -> Optional[List[dict]]:
    """
    加载 JSON 元数据文件。

    Args:
        json_path: JSON 文件路径

    Returns:
        元数据列表，失败时返回 None
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON 文件不存在: {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        return items
    except Exception as e:
        print(f"[ERROR] 加载 JSON 失败 {json_path}: {e}")
        return None


def extract_frames_from_json(
    json_path: str,
    output_dir: str,
    use_img_path_as_name: bool = True,
) -> Tuple[int, int, int]:
    """
    从单个 JSON 元数据文件中提取所有帧并保存。

    每个 JSON item 需包含以下字段：
      - video_path: 视频文件的相对路径
      - idx: 帧索引
      - img_path: 输出图片的参考路径（用于生成文件名）

    保存的图片尺寸与原视频帧完全一致。

    Args:
        json_path: JSON 元数据文件路径
        output_dir: 输出目录
        use_img_path_as_name: 是否使用 img_path 中的文件名作为保存名

    Returns:
        (成功数, 跳过数, 失败数)
    """
    json_path = os.path.abspath(json_path)
    json_dir = os.path.dirname(json_path)
    json_name = os.path.basename(json_path)

    items = load_json_metadata(json_path)
    if items is None:
        return (0, 0, 0)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[DATASET] {json_name} ({len(items)} items)")
    print(f"[OUTPUT]  {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, item in enumerate(items):
        video_rel_path = item.get('video_path')
        idx = item.get('idx')
        img_rel_path = item.get('img_path')

        # 验证必要字段
        if not video_rel_path or idx is None:
            print(f"  [{i+1}/{len(items)}] [SKIP] 缺少 video_path 或 idx 字段")
            skip_count += 1
            continue

        # 解析视频路径
        video_path = resolve_video_path(video_rel_path, json_dir)
        if video_path is None:
            print(f"  [{i+1}/{len(items)}] [SKIP] 视频文件未找到: {video_rel_path}")
            skip_count += 1
            continue

        # 读取帧
        frame = read_frame(video_path, idx)
        if frame is None:
            print(f"  [{i+1}/{len(items)}] [FAIL] 读取帧失败: {os.path.basename(video_path)} frame={idx}")
            fail_count += 1
            continue

        # 获取帧尺寸（即原视频尺寸）
        h, w = frame.shape[:2]

        # 确定输出文件名
        if use_img_path_as_name and img_rel_path:
            # 使用 JSON 中 img_path 的文件名
            out_filename = os.path.basename(img_rel_path)
            # 确保扩展名为 .png 以保证无损
            name_no_ext = os.path.splitext(out_filename)[0]
            out_filename = name_no_ext + ".png"
        else:
            # 使用视频名 + 帧号 作为文件名
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_filename = f"{video_name}_frame{idx:06d}.png"

        save_path = os.path.join(output_dir, out_filename)

        # 直接保存，不做任何缩放，保持原始尺寸
        ok = cv2.imwrite(save_path, frame)
        if ok:
            success_count += 1
            print(f"  [{i+1}/{len(items)}] [OK] {out_filename}  ({w}x{h})")
        else:
            fail_count += 1
            print(f"  [{i+1}/{len(items)}] [FAIL] 保存失败: {save_path}")

    return (success_count, skip_count, fail_count)


def main():
    parser = argparse.ArgumentParser(
        description="从 JSON 元数据文件中提取视频帧并保存为原始尺寸的图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取单个 JSON 文件中的所有帧
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json

  # 指定输出目录
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json --output ./my_frames

  # 批量提取多个 JSON 文件
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json ../meta3/fd_2000_meta.json

  # 使用视频名+帧号作为文件名（而非 JSON 中的 img_path）
  python extract_frames_from_json.py --json ../meta3/fd_1000_meta.json --auto-name
        """,
    )

    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="JSON 元数据文件路径（支持多个）",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="使用「视频名_frameXXXXXX.png」作为文件名，而非 JSON 中的 img_path",
    )

    args = parser.parse_args()

    total_success = 0
    total_skip = 0
    total_fail = 0

    for json_path in args.json:
        s, sk, f = extract_frames_from_json(
            json_path=json_path,
            output_dir=args.output,
            use_img_path_as_name=not args.auto_name,
        )
        total_success += s
        total_skip += sk
        total_fail += f

    print(f"\n{'='*60}")
    print(f"[DONE] 全部完成")
    print(f"  成功: {total_success}")
    print(f"  跳过: {total_skip}")
    print(f"  失败: {total_fail}")
    print(f"  输出目录: {os.path.abspath(args.output)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
