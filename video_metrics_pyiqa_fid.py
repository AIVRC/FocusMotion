import cv2
import torch
import pyiqa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# =========================
# Config
# =========================
GT_DIR = "/home/yanghaotian/server_data/yanghaotian/data/testing"   # folder with ground truth videos
GEN_DIR = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/video-20260128/1042-3.5-pose_guider-130200-motion_module-2517/results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FID_STRIDE = 5               # sample every frame; increase to 2-5 for long videos

# Temporary folders for FID
FID_TEMP_DIR = Path("./fid_temp")
FID_GT_DIR = FID_TEMP_DIR / "gt"
FID_GEN_DIR = FID_TEMP_DIR / "gen"
FID_GT_DIR.mkdir(parents=True, exist_ok=True)
FID_GEN_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load IQA metrics
# =========================
print("Loading pyiqa models...")
ssim_metric = pyiqa.create_metric("ssim", device=DEVICE)
psnr_metric = pyiqa.create_metric("psnr", device=DEVICE)
lpips_metric = pyiqa.create_metric("lpips", device=DEVICE)
fid_metric = pyiqa.create_metric("fid", device=DEVICE)

# =========================
# Gather video paths
# =========================
gt_videos = sorted(Path(GT_DIR).glob("*.mp4"))
gen_videos = sorted(Path(GEN_DIR).glob("*.mp4"))
assert len(gt_videos) == len(gen_videos), "GT and GEN video counts must match!"

# =========================
# Accumulators
# =========================
all_ssim, all_psnr, all_lpips = [], [], []
global_idx = 0

# =========================
# Video loop
# =========================
for gt_path, gen_path in tqdm(zip(gt_videos, gen_videos), total=len(gt_videos), desc="Processing videos"):

    cap_gt = cv2.VideoCapture(str(gt_path))
    cap_gen = cv2.VideoCapture(str(gen_path))

    gt_len = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    gen_len = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(gt_len, gen_len)

    for frame_idx in range(num_frames):
        ret_gt, frame_gt = cap_gt.read()
        ret_gen, frame_gen = cap_gen.read()
        if not ret_gt or not ret_gen:
            break

        # BGR → RGB
        frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
        frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)

        # Resize GEN to GT size if mismatch (to avoid frame mismatch errors)
        if frame_gt.shape != frame_gen.shape:
            frame_gen = cv2.resize(frame_gen, (frame_gt.shape[1], frame_gt.shape[0]))

        # -------------------------
        # Per-frame metrics
        # -------------------------
        gt_tensor = torch.from_numpy(frame_gt).float().permute(2,0,1).unsqueeze(0)/255.0
        gen_tensor = torch.from_numpy(frame_gen).float().permute(2,0,1).unsqueeze(0)/255.0
        gt_tensor = gt_tensor.to(DEVICE)
        gen_tensor = gen_tensor.to(DEVICE)

        with torch.no_grad():
            ssim_val = ssim_metric(gen_tensor, gt_tensor).item()
            psnr_val = psnr_metric(gen_tensor, gt_tensor).item()
            lpips_val = lpips_metric(gen_tensor, gt_tensor).item()

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_lpips.append(lpips_val)

        # -------------------------
        # FID frames (stride sampling)
        # -------------------------
        if frame_idx % FID_STRIDE == 0:
            # save raw frames without resizing
            cv2.imwrite(str(FID_GT_DIR / f"{global_idx:07d}.png"), cv2.cvtColor(frame_gt, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(FID_GEN_DIR / f"{global_idx:07d}.png"), cv2.cvtColor(frame_gen, cv2.COLOR_RGB2BGR))
            global_idx += 1

    cap_gt.release()
    cap_gen.release()

# =========================
# Print per-frame IQA results
# =========================
print("\n========== Frame Metrics ==========")
print(f"Average SSIM : {np.mean(all_ssim):.4f}")
print(f"Average PSNR : {np.mean(all_psnr):.2f}")
print(f"Average LPIPS: {np.mean(all_lpips):.4f}")
print("===================================")

# =========================
# Compute FID
# =========================
print("Computing FID...")
fid_score = fid_metric(str(FID_GEN_DIR), str(FID_GT_DIR)).item()
print(f"FID score: {fid_score:.4f}")

# =========================
# Cleanup
# =========================
shutil.rmtree(FID_TEMP_DIR)
