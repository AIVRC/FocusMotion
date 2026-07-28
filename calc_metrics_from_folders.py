import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pyiqa
import torch
from PIL import Image
from torchvision.transforms import ToTensor


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def find_generated_match(gt_path: Path, gen_images: List[Path]) -> Path:
    gt_name = gt_path.name.lower()
    gt_stem = gt_path.stem.lower()

    candidates: List[Tuple[int, Path]] = []
    for gen_path in gen_images:
        gen_name = gen_path.name.lower()
        gen_stem = gen_path.stem.lower()
        if gt_name in gen_name or gt_stem in gen_name or gt_stem in gen_stem:
            # 使用更长关键字优先，减少误匹配
            match_score = max(len(gt_name), len(gt_stem))
            candidates.append((match_score, gen_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_pair_as_tensor(gt_path: Path, gen_path: Path, device: torch.device):
    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    if gt_img.size != gen_img.size:
        gen_img = gen_img.resize(gt_img.size, Image.BILINEAR)

    to_tensor = ToTensor()
    gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)
    gen_tensor = to_tensor(gen_img).unsqueeze(0).to(device)
    return gt_tensor, gen_tensor


def main():
    parser = argparse.ArgumentParser(
        description="计算两个文件夹图片的 SSIM / PSNR / LPIPS（生成图文件名包含 GT 文件名）"
    )
    parser.add_argument("--gt_dir", required=True, help="Ground Truth 图片目录")
    parser.add_argument("--gen_dir", required=True, help="生成图片目录")
    parser.add_argument(
        "--output",
        default=None,
        help="结果 txt 输出路径，默认写到 gen_dir/metrics_results.txt",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：遇到 GT 找不到匹配生成图时直接报错退出",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    gen_dir = Path(args.gen_dir)
    if not gt_dir.exists() or not gen_dir.exists():
        raise FileNotFoundError(f"目录不存在: gt_dir={gt_dir}, gen_dir={gen_dir}")

    output_path = Path(args.output) if args.output else (gen_dir / "metrics_results.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ssim_metric = pyiqa.create_metric("ssim", device=device)
    psnr_metric = pyiqa.create_metric("psnr", device=device)
    lpips_metric = pyiqa.create_metric("lpips", device=device)

    gt_images = collect_images(gt_dir)
    gen_images = collect_images(gen_dir)
    if not gt_images:
        raise RuntimeError(f"GT 目录里没有图片: {gt_dir}")
    if not gen_images:
        raise RuntimeError(f"生成目录里没有图片: {gen_dir}")

    details: List[str] = []
    unmatched: List[str] = []
    ssim_sum = 0.0
    psnr_sum = 0.0
    lpips_sum = 0.0
    matched_count = 0

    for gt_path in gt_images:
        gen_path = find_generated_match(gt_path, gen_images)
        if gen_path is None:
            msg = f"{gt_path.name}\tno matched generated image"
            if args.strict:
                raise FileNotFoundError(msg)
            unmatched.append(msg)
            continue

        try:
            gt_tensor, gen_tensor = load_pair_as_tensor(gt_path, gen_path, device)
            ssim_val = float(ssim_metric(gen_tensor, gt_tensor).item())
            psnr_val = float(psnr_metric(gen_tensor, gt_tensor).item())
            lpips_val = float(lpips_metric(gen_tensor, gt_tensor).item())

            matched_count += 1
            ssim_sum += ssim_val
            psnr_sum += psnr_val
            lpips_sum += lpips_val

            details.append(
                f"GT={gt_path.name}\tGEN={gen_path.name}\t"
                f"SSIM={ssim_val:.6f}\tPSNR={psnr_val:.4f}\tLPIPS={lpips_val:.6f}"
            )
            print(
                f"OK GT={gt_path.name} -> GEN={gen_path.name} | "
                f"SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f}"
            )
        except Exception as e:
            unmatched.append(f"{gt_path.name}\terror: {e}")

    avg_ssim = ssim_sum / matched_count if matched_count else 0.0
    avg_psnr = psnr_sum / matched_count if matched_count else 0.0
    avg_lpips = lpips_sum / matched_count if matched_count else 0.0

    with output_path.open("w", encoding="utf-8") as f:
        for line in details:
            f.write(line + "\n")
        f.write("\n=== SUMMARY ===\n")
        f.write(f"gt_total: {len(gt_images)}\n")
        f.write(f"matched: {matched_count}\n")
        f.write(f"unmatched: {len(unmatched)}\n")
        f.write(f"avg_ssim: {avg_ssim:.6f}\n")
        f.write(f"avg_psnr: {avg_psnr:.4f}\n")
        f.write(f"avg_lpips: {avg_lpips:.6f}\n")
        if unmatched:
            f.write("\n=== UNMATCHED_OR_FAILED ===\n")
            for line in unmatched:
                f.write(line + "\n")

    print("\nFinished.")
    print(f"GT total: {len(gt_images)}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched/failed: {len(unmatched)}")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average LPIPS: {avg_lpips:.6f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
