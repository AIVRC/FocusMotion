import os

# =========================
# Paths
# =========================
REF_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/frame1"
POSE_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/frame300"
COMPLEX_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/ubc_comp"

OUTPUT_TXT = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/pairs.txt"

# Supported image extensions
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# =========================
# Collect files
# =========================
ref_imgs = sorted([
    f for f in os.listdir(REF_DIR)
    if f.lower().endswith(IMG_EXTS)
])

pose_imgs = sorted([
    f for f in os.listdir(POSE_DIR)
    if f.lower().endswith(IMG_EXTS)
])

comp_imgs = sorted([
    f for f in os.listdir(COMPLEX_DIR)
    if f.lower().endswith(IMG_EXTS)
])

print("Reference images:", len(ref_imgs))
print("Pose images:", len(pose_imgs))

# =========================
# Safety check
# =========================
if len(ref_imgs) != len(pose_imgs):
    raise ValueError("❌ Number of ref and pose images does NOT match!")

# =========================
# Write pairs.txt
# =========================
with open(OUTPUT_TXT, "w") as f:

    for ref_name, pose_name, comp_name in zip(ref_imgs, pose_imgs, comp_imgs): 

        ref_path = os.path.join(REF_DIR, ref_name)
        pose_path = os.path.join(POSE_DIR, pose_name)
        comp_path = os.path.join(COMPLEX_DIR, comp_name)

        f.write(f"{ref_path},{pose_path},{comp_path}\n")

print("\n🎉 Done!")
print("Pairs file saved to:")
print(OUTPUT_TXT)
print("Total pairs:", len(ref_imgs))
