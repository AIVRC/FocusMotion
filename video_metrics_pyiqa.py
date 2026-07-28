import cv2
import torch
import pyiqa
import numpy as np
from tqdm import tqdm

# =========================
# Paths
# =========================

GT_VIDEO = "/home/yanghaotian/server_data/yanghaotian/data/testing/91cC+1+C4SS.mp4"
GEN_VIDEO = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/video-20260128/1042-3.5-pose_guider-130200-motion_module-2517/results/91-NTYmAx2S_frame1_img_91-NTYmAx2S_frame1_video_91-NTYmAx2S_3.5_20_0.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load metrics
# =========================

print("Loading pyiqa models...")

ssim_metric = pyiqa.create_metric("ssim", device=DEVICE)
psnr_metric = pyiqa.create_metric("psnr", device=DEVICE)
lpips_metric = pyiqa.create_metric("lpips", device=DEVICE)

# =========================
# Open videos
# =========================

cap_gt = cv2.VideoCapture(GT_VIDEO)
cap_gen = cv2.VideoCapture(GEN_VIDEO)

gt_len = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
gen_len = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))

num_frames = min(gt_len, gen_len)

print(f"GT frames: {gt_len}")
print(f"GEN frames: {gen_len}")
print(f"Using frames: {num_frames}")

# =========================
# Score storage
# =========================

ssim_scores = []
psnr_scores = []
lpips_scores = []

# =========================
# Frame loop
# =========================

for idx in tqdm(range(num_frames)):

    ret_gt, frame_gt = cap_gt.read()
    ret_gen, frame_gen = cap_gen.read()

    if not ret_gt or not ret_gen:
        break

    # BGR → RGB
    frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
    frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)

    # Resize GEN to GT size if mismatch
    if frame_gt.shape != frame_gen.shape:
        frame_gen = cv2.resize(frame_gen, (frame_gt.shape[1], frame_gt.shape[0]))

    # To tensor: [1,3,H,W], range [0,1]
    gt_tensor = torch.from_numpy(frame_gt).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    gen_tensor = torch.from_numpy(frame_gen).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    gt_tensor = gt_tensor.to(DEVICE)
    gen_tensor = gen_tensor.to(DEVICE)

    # =========================
    # Compute metrics
    # =========================

    with torch.no_grad():

        ssim_val = ssim_metric(gen_tensor, gt_tensor).item()
        psnr_val = psnr_metric(gen_tensor, gt_tensor).item()
        lpips_val = lpips_metric(gen_tensor, gt_tensor).item()

    ssim_scores.append(ssim_val)
    psnr_scores.append(psnr_val)
    lpips_scores.append(lpips_val)

# =========================
# Cleanup
# =========================

cap_gt.release()
cap_gen.release()

# =========================
# Results
# =========================

print("\n========== Video Metrics ==========")

print(f"Average SSIM : {np.mean(ssim_scores):.4f}")
print(f"Average PSNR : {np.mean(psnr_scores):.2f}")
print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")

print("===================================")
