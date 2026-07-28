import json
import os
import random
from decord import VideoReader
from decord import cpu

# ------------------------
# Config
# ------------------------
input_json = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/meta/ubc_train_meta.json"          # input json path
output_json = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/meta_comp3/ubc_train_meta.json"  # output json path
img_output_dir = "../../data/fd_complex_3"   # where images will be saved

os.makedirs(img_output_dir, exist_ok=True)

# ------------------------
# Load input json
# ------------------------
with open(input_json, "r") as f:
    vid_meta = json.load(f)

new_vid_meta = []

# ------------------------
# Process each video entry
# ------------------------
for item in vid_meta:
    video_path = item["video_path"]
    kps_path = item["kps_path"]

    # 1. read video
    video_reader = VideoReader(video_path, ctx=cpu(0))

    # 2. get video length
    video_length = len(video_reader)

    # 3. random reference frame index
    ref_img_idx = random.randint(0, video_length - 1)

    # 4. extract video name (e.g. 00331 from ../../data/TikTok/00331.mp4)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 5. generate image path
    img_path = os.path.join(
        img_output_dir,
        f"{video_name}_complex.png"
    )

    # 6. create new entry
    new_item = {
        "video_path": video_path,
        "kps_path": kps_path,
        "idx": ref_img_idx,
        "img_path": img_path,
        "label": 0
    }

    new_vid_meta.append(new_item)

# ------------------------
# Save output json
# ------------------------
with open(output_json, "w") as f:
    json.dump(new_vid_meta, f, indent=2)

print(f"Saved new json to {output_json}")
