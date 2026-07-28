import os
import subprocess
import yaml

# =========================
# Paths (adjust if needed)
# =========================
BASE_DIR = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1"
CONFIG_PATH = os.path.join(BASE_DIR, "configs/test_stage_1_comp.yaml")
POSE_ALIGN_SCRIPT = "pose_align_image.py"
POSE_OUTPUT_DIR = os.path.join(BASE_DIR, "assets/poses/align")
PAIRS_FILE = "/home/yanghaotian/server_data/yanghaotian/data/ubc_test10/pairs.txt"
DEFAULT_COMPLEX_IMAGE = ""

# =========================
# Load existing config
# =========================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if not isinstance(config, dict):
    raise RuntimeError("Config file is not a YAML mapping")

# =========================
# Explicitly replace test_cases
# =========================
if "test_cases" in config:
    del config["test_cases"]

config["test_cases"] = []


def parse_pairs_line(line: str):
    parts = [x.strip() for x in line.strip().split(",") if x.strip()]
    if len(parts) == 2:
        ref_img, pose_img = parts
        complex_img = DEFAULT_COMPLEX_IMAGE
    elif len(parts) >= 3:
        ref_img, pose_img, complex_img = parts[:3]
    else:
        raise ValueError(f"Invalid line in pairs file: {line!r}")
    return ref_img, pose_img, complex_img

# =========================
# Process pairs
# =========================
with open(PAIRS_FILE, "r") as f:
    for line in f:
        if not line.strip():
            continue

        ref_img, pose_img, complex_img = parse_pairs_line(line)

        ref_name = os.path.splitext(os.path.basename(ref_img))[0]
        pose_name = os.path.splitext(os.path.basename(pose_img))[0]

        # 1️⃣ Run pose alignment
        subprocess.run(
            [
                "python",
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

        # 3️⃣ Add to test_cases (format expected by test_stage_1_comp.py)
        config["test_cases"].append(
            {
                "ref": ref_img,
                "pose": pose_struct_path,
                "complex": complex_img,
            }
        )

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
    ["python3", "test_stage_1_comp.py", "--config", "./configs/test_stage_1_comp.yaml"],
    cwd=BASE_DIR,
    check=True
)
