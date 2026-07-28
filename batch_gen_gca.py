import os
import subprocess
import yaml

# =========================
# Paths (adjust if needed)
# =========================
BASE_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1"
CONFIG_PATH = os.path.join(BASE_DIR, "configs/test_stage_1_ubc.yaml")
POSE_ALIGN_SCRIPT = "pose_align_image.py"
POSE_OUTPUT_DIR = os.path.join(BASE_DIR, "assets/poses/align")
PAIRS_FILE = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/pairs.txt"

# =========================
# Load existing config
# =========================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if not isinstance(config, dict):
    raise RuntimeError("Config file is not a YAML mapping")

# =========================
# Replace test_cases with empty list
# =========================
config["test_cases"] = []

# =========================
# Process pairs
# =========================
with open(PAIRS_FILE, "r") as f:
    for line in f:
        ref_img, pose_img, complex_region = line.strip().split(",")

        ref_name = os.path.splitext(os.path.basename(ref_img))[0]
        pose_name = os.path.splitext(os.path.basename(pose_img))[0]

        # 1️⃣ Run pose alignment
        subprocess.run(
            [
                "/home/yanghaotian/server_data/yanghaotian/miniconda3/envs/pose_fixed/bin/python",
                POSE_ALIGN_SCRIPT,
                "--imgfn_refer", ref_img,
                "--vidfn", pose_img
            ],
            cwd=BASE_DIR,
            check=True
        )

        # 2️⃣ Pose structure output path
        pose_struct_path = os.path.join(
            POSE_OUTPUT_DIR,
            f"img_{ref_name}_video_{pose_name}.png"
        )

        # 3️⃣ Add to test_cases as dict
        config["test_cases"].append({
            "ref": ref_img,
            "pose": pose_struct_path,
            "complex": complex_region
        })

# =========================
# Save updated config
# =========================
with open(CONFIG_PATH, "w") as f:
    yaml.dump(
        config,
        f,
        sort_keys=False,
        default_flow_style=False
    )

print("✅ test_cases section replaced successfully.")

# =========================
# Run inference
# =========================
subprocess.run(
    ["/home/yanghaotian/server_data/yanghaotian/miniconda3/envs/pcdms/bin/python", 
     "test_stage_1_comp.py", 
     "--config", "./configs/test_stage_1_ubc.yaml"],
    cwd=BASE_DIR,
    check=True
)