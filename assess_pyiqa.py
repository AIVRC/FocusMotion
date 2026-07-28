import os
import torch
import pyiqa
from PIL import Image

# =========================
# Paths
# =========================
GT_IMG_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_test10/frame300"
OUTPUT_IMG_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_test10/frame300"
# OUTPUT_IMG_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/output/image-20260508/0708-pose_guider-1199450/res"
RESULT_TXT = os.path.join(OUTPUT_IMG_DIR, "iqa_results.txt")

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Load metrics
# =========================
ssim_metric = pyiqa.create_metric("ssim", device=device)
psnr_metric = pyiqa.create_metric("psnr", device=device)
lpips_metric = pyiqa.create_metric("lpips", device=device)

# FID is folder-based
fid_metric = pyiqa.create_metric("fid", device=device)

# =========================
# Stats
# =========================
results = []
failed = []

ssim_sum = 0.0
psnr_sum = 0.0
lpips_sum = 0.0
count = 0

# =========================
# Pairwise evaluation
# =========================
print("\n🔍 Computing SSIM / PSNR / LPIPS...\n")

for fname in sorted(os.listdir(OUTPUT_IMG_DIR)):

    if not fname.endswith(".jpg"):
        continue

    if not fname.startswith("res_"):
        continue

    try:
        # res_xxx_frame1_img_xxx_frame1_video_xxx_frame300_7_42.jpg

        if "_video_" not in fname:
            failed.append(f"{fname} | invalid name format")
            continue

        core = fname.replace("_7_42.jpg", "")
        gt_part = core.split("_video_")[-1]
        gt_name = f"{gt_part}.png"

        gen_path = os.path.join(OUTPUT_IMG_DIR, fname)
        gt_path = os.path.join(GT_IMG_DIR, gt_name)

        if not os.path.exists(gt_path):
            failed.append(f"{fname} | missing GT: {gt_name}")
            continue

        # -------------------------
        # Resize GT if mismatch
        # -------------------------
        gen_img = Image.open(gen_path)
        gt_img = Image.open(gt_path)

        if gen_img.size != gt_img.size:
            gt_img = gt_img.resize(gen_img.size, Image.BILINEAR)
            gt_img.save(gt_path)   # overwrite resized GT for consistent FID

        # -------------------------
        # Compute metrics (path input)
        # -------------------------
        ssim_val = ssim_metric(gen_path, gt_path).item()
        psnr_val = psnr_metric(gen_path, gt_path).item()
        lpips_val = lpips_metric(gen_path, gt_path).item()

        # -------------------------
        # Accumulate
        # -------------------------
        ssim_sum += ssim_val
        psnr_sum += psnr_val
        lpips_sum += lpips_val
        count += 1

        results.append(
            f"{fname}\tGT={gt_name}\t"
            f"SSIM={ssim_val:.6f}\t"
            f"PSNR={psnr_val:.4f}\t"
            f"LPIPS={lpips_val:.6f}"
        )

        print(f"✅ {fname} | SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} LPIPS={lpips_val:.4f}")

    except Exception as e:
        failed.append(f"{fname} | error: {str(e)}")

# =========================
# Compute averages
# =========================
if count > 0:
    avg_ssim = ssim_sum / count
    avg_psnr = psnr_sum / count
    avg_lpips = lpips_sum / count
else:
    avg_ssim = avg_psnr = avg_lpips = 0.0

# =========================
# Compute FID (folder)
# =========================
print("\n🚀 Computing FID...")

try:
    fid_score = fid_metric(OUTPUT_IMG_DIR, GT_IMG_DIR).item()
    print("FID:", fid_score)
except Exception as e:
    fid_score = None
    print("❌ FID failed:", e)

# =========================
# Write results
# =========================
with open(RESULT_TXT, "w") as f:

    for line in results:
        f.write(line + "\n")

    f.write("\n=== SUMMARY ===\n")
    f.write(f"Total Samples: {count}\n")
    f.write(f"Average SSIM: {avg_ssim:.6f}\n")
    f.write(f"Average PSNR: {avg_psnr:.4f}\n")
    f.write(f"Average LPIPS: {avg_lpips:.6f}\n")

    if fid_score is not None:
        f.write(f"FID: {fid_score:.4f}\n")
    else:
        f.write("FID: FAILED\n")

    if failed:
        f.write("\n=== FAILED CASES ===\n")
        for line in failed:
            f.write(line + "\n")

# =========================
# Final summary
# =========================
print("\n🎉 Evaluation finished")
print("Saved to:", RESULT_TXT)
print("Successful:", count)
print("Failed:", len(failed))
