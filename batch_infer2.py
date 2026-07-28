import os
import subprocess
import yaml

# =========================
# Paths
# =========================

PROJECT_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1"

POSE_ALIGN_SCRIPT = os.path.join(PROJECT_DIR, "pose_align.py")
TEST_SCRIPT = os.path.join(PROJECT_DIR, "test_stage_2.py")

CONFIG_PATH = os.path.join(PROJECT_DIR, "configs/test_stage_2_copy.yaml")

PAIRS_TXT = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/pairs.txt"

POSE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "assets/poses/align")

# =========================
# Load pairs
# =========================

pairs = []

with open(PAIRS_TXT, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        ref, video = line.split(",")
        pairs.append((ref.strip(), video.strip()))

print("Total pairs loaded:", len(pairs))

if len(pairs) == 0:
    raise RuntimeError("No valid pairs found in pairs.txt")

# =========================
# Step 1: Run pose_align
# =========================

print("\n🚀 Running pose alignment...\n")

for idx, (ref_img, video_path) in enumerate(pairs, start=1):

    print(f"[{idx}/{len(pairs)}]")
    print("Reference:", ref_img)
    print("Video    :", video_path)

    cmd = [
        "python",
        POSE_ALIGN_SCRIPT,
        "--imgfn_refer", ref_img,
        "--vidfn", video_path
    ]

    subprocess.run(cmd, check=True)

print("\n✅ All pose alignment finished")

# =========================
# Step 2: Update YAML test cases
# =========================

print("\n✏ Updating test_stage_2.yaml ...")

with open(CONFIG_PATH, "r") as f:
    config_data = yaml.safe_load(f)

new_test_cases = {}

missing_pose = 0

for ref_img, video_path in pairs:

    ref_name = os.path.splitext(os.path.basename(ref_img))[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    pose_video_name = f"img_{ref_name}_video_{video_name}.mp4"
    pose_video_path = os.path.join(POSE_OUTPUT_DIR, pose_video_name)

    if not os.path.exists(pose_video_path):
        print("⚠ Missing pose video:", pose_video_path)
        missing_pose += 1
        continue

    new_test_cases[ref_img] = [pose_video_path]

print("Valid test cases:", len(new_test_cases))
print("Missing pose videos:", missing_pose)

if len(new_test_cases) == 0:
    raise RuntimeError("No valid pose videos found — aborting YAML update")

# Replace ONLY test cases section
config_data["test_cases"] = new_test_cases

# Write updated config
with open(CONFIG_PATH, "w") as f:
    yaml.dump(config_data, f, sort_keys=False)

print("✅ YAML updated successfully")

# =========================
# Step 3: Run inference
# =========================

print("\n🚀 Running test_stage_2.py ...\n")

cmd = [
    "python",
    TEST_SCRIPT,
    "--config",
    "./configs/test_stage_2_copy.yaml"
]

subprocess.run(cmd, check=True)

print("\n🎉 Batch pipeline finished successfully")
