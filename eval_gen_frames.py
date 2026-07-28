import os
from PIL import Image
from eval_methods import ImageEvalue
import torch
import lpips
import torchvision.transforms as transforms


# =========================
# Paths
# =========================
POSE_IMG_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_test10/frame300"
OUTPUT_IMG_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/output/image-20260508/0708-pose_guider-1199450/res"
RESULT_TXT = os.path.join(OUTPUT_IMG_DIR, "ssim_psnr_results.txt")

# =========================
# Init
# =========================
evaluator = ImageEvalue()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),                # [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))  # -> [-1,1]
])

results = []
failed = []

ssim_sum = 0.0
psnr_sum = 0.0
lpips_sum = 0.0
count = 0


# =========================
# Process generated images
# =========================
for fname in sorted(os.listdir(OUTPUT_IMG_DIR)):

    if not fname.lower().endswith(".jpg"):
        continue

    if not fname.startswith("res_"):
        continue

    try:
        # Example:
        # res_xxx_frame1_video_xxx_frame300_7_42.jpg

        if "_video_" not in fname:
            failed.append(f"{fname} | invalid filename format")
            continue

        core = fname.replace("_7_42.jpg", "")
        pose_part = core.split("_video_")[-1]
        pose_img_name = f"{pose_part}.png"

        pose_img_path = os.path.join(POSE_IMG_DIR, pose_img_name)
        out_img_path = os.path.join(OUTPUT_IMG_DIR, fname)

        if not os.path.exists(pose_img_path):
            failed.append(f"{fname} | missing pose image: {pose_img_name}")
            continue

        # -------------------------
        # Load grayscale (SSIM/PSNR)
        # -------------------------
        img1 = Image.open(out_img_path).convert("L")   # generated output
        img2 = Image.open(pose_img_path).convert("L") # pose frame300

        # -------------------------
        # Load RGB (LPIPS)
        # -------------------------
        img1_rgb = Image.open(out_img_path).convert("RGB")
        img2_rgb = Image.open(pose_img_path).convert("RGB")

        # -------------------------
        # Resize pose if needed
        # -------------------------
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)
            img2_rgb = img2_rgb.resize(img1_rgb.size, Image.BILINEAR)

        # -------------------------
        # Metrics
        # -------------------------
        ssim = evaluator.SSIM(img1, img2)
        psnr = evaluator.PSNR(img1, img2)

        t1 = to_tensor(img1_rgb).unsqueeze(0).to(device)
        t2 = to_tensor(img2_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            lpips_val = lpips_fn(t1, t2).item()

        # -------------------------
        # Accumulate
        # -------------------------
        ssim_sum += ssim
        psnr_sum += psnr
        lpips_sum += lpips_val
        count += 1

        results.append(
            f"{fname}\tPOSE={pose_img_name}\t"
            f"SSIM={ssim:.6f}\tPSNR={psnr:.4f}\tLPIPS={lpips_val:.6f}"
        )

        print(f"✅ {fname} | SSIM={ssim:.4f} PSNR={psnr:.2f} LPIPS={lpips_val:.4f}")

    except Exception as e:
        failed.append(f"{fname} | error: {str(e)}")


# =========================
# Safe averages
# =========================
if count > 0:
    avg_ssim = ssim_sum / count
    avg_psnr = psnr_sum / count
    avg_lpips = lpips_sum / count
else:
    avg_ssim = 0.0
    avg_psnr = 0.0
    avg_lpips = 0.0


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

    if failed:
        f.write("\n=== FAILED CASES ===\n")
        for line in failed:
            f.write(line + "\n")


# =========================
# Console summary
# =========================
print("\n🎉 Evaluation complete")
print(f"Results written to: {RESULT_TXT}")
print(f"Successful evaluations: {count}")
print(f"Failed evaluations: {len(failed)}")
