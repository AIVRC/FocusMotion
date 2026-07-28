import argparse
from pathlib import Path

import pyiqa
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 你可以直接在这里填写两个文件夹路径（不传命令行参数时会使用这里的默认值）
# DEFAULT_PRED_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/output/image-20260508/0708-pose_guider-1199450/res"
DEFAULT_PRED_DIR = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/image-20260509/0807-pose_guider-70800/res"
DEFAULT_GT_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_test10/frame300"


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def find_gt_match(pred_path: Path, gt_images):
    pred_name = pred_path.name.lower()
    pred_stem = pred_path.stem.lower()

    # 1) 先尝试完全同名
    for gt_path in gt_images:
        if pred_name == gt_path.name.lower():
            return gt_path

    # 2) 再尝试 pred 文件名包含 gt 文件名或 gt stem
    candidates = []
    for gt_path in gt_images:
        gt_name = gt_path.name.lower()
        gt_stem = gt_path.stem.lower()
        if gt_name in pred_name or gt_stem in pred_stem or gt_stem in pred_name:
            # 用更长的关键字优先，减少误匹配
            match_len = max(len(gt_name), len(gt_stem))
            candidates.append((match_len, gt_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description="对两个文件夹按同名图片计算 SSIM/PSNR/LPIPS")
    parser.add_argument(
        "--pred_dir",
        default=DEFAULT_PRED_DIR,
        help="生成结果图片目录（默认使用代码里的 DEFAULT_PRED_DIR）",
    )
    parser.add_argument(
        "--gt_dir",
        default=DEFAULT_GT_DIR,
        help="GT 图片目录（默认使用代码里的 DEFAULT_GT_DIR）",
    )
    parser.add_argument(
        "--output_txt",
        default=None,
        help="输出结果 txt 路径，默认写到 pred_dir/metrics_ssim_psnr_lpips.txt",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：若 pred 中有文件在 gt 中不存在则直接报错退出",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    if str(pred_dir) == "/path/to/your/pred_dir" or str(gt_dir) == "/path/to/your/gt_dir":
        raise ValueError("请先在代码顶部修改 DEFAULT_PRED_DIR 和 DEFAULT_GT_DIR，或通过 --pred_dir/--gt_dir 传入路径")
    if not pred_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"目录不存在: pred_dir={pred_dir}, gt_dir={gt_dir}")

    output_txt = Path(args.output_txt) if args.output_txt else (pred_dir / "metrics_ssim_psnr_lpips.txt")
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ssim_metric = pyiqa.create_metric("ssim", device=device)
    psnr_metric = pyiqa.create_metric("psnr", device=device)
    lpips_metric = pyiqa.create_metric("lpips", device=device)

    pred_images = list_images(pred_dir)
    gt_images = list_images(gt_dir)

    results = []
    failed = []
    ssim_sum = 0.0
    psnr_sum = 0.0
    lpips_sum = 0.0
    count = 0

    for pred_path in pred_images:
        gt_path = find_gt_match(pred_path, gt_images)
        if gt_path is None:
            msg = f"{pred_path.name}\tmissing GT (exact or contains match)"
            if args.strict:
                raise FileNotFoundError(msg)
            failed.append(msg)
            continue

        try:
            ssim_val = float(ssim_metric(str(pred_path), str(gt_path)).item())
            psnr_val = float(psnr_metric(str(pred_path), str(gt_path)).item())
            lpips_val = float(lpips_metric(str(pred_path), str(gt_path)).item())

            ssim_sum += ssim_val
            psnr_sum += psnr_val
            lpips_sum += lpips_val
            count += 1

            line = (
                f"{pred_path.name}\t"
                f"SSIM={ssim_val:.6f}\t"
                f"PSNR={psnr_val:.4f}\t"
                f"LPIPS={lpips_val:.6f}"
            )
            results.append(line)
            print(f"OK {pred_path.name} | SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f}")
        except Exception as e:
            failed.append(f"{pred_path.name}\terror: {e}")

    avg_ssim = ssim_sum / count if count else 0.0
    avg_psnr = psnr_sum / count if count else 0.0
    avg_lpips = lpips_sum / count if count else 0.0

    with output_txt.open("w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
        f.write("\n=== SUMMARY ===\n")
        f.write(f"paired_samples: {count}\n")
        f.write(f"avg_ssim: {avg_ssim:.6f}\n")
        f.write(f"avg_psnr: {avg_psnr:.4f}\n")
        f.write(f"avg_lpips: {avg_lpips:.6f}\n")
        f.write(f"failed_samples: {len(failed)}\n")
        if failed:
            f.write("\n=== FAILED ===\n")
            for line in failed:
                f.write(line + "\n")

    print("\nDone.")
    print(f"paired_samples: {count}")
    print(f"avg_ssim: {avg_ssim:.6f}")
    print(f"avg_psnr: {avg_psnr:.4f}")
    print(f"avg_lpips: {avg_lpips:.6f}")
    print(f"failed_samples: {len(failed)}")
    print(f"saved: {output_txt}")


if __name__ == "__main__":
    main()
