import json
import os
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

# ------------------------
# Config
# ------------------------
input_json = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/meta2/Tik_meta2.json"

# ------------------------
# Load json
# ------------------------
with open(input_json, "r") as f:
    vid_meta = json.load(f)

# ------------------------
# Process videos
# ------------------------
for item in tqdm(vid_meta, desc="Extracting frames"):
    video_path = item["video_path"]
    idx = item["idx"]
    img_path = item["img_path"]

    # make sure output directory exists
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    try:
        # read video
        vr = VideoReader(video_path, ctx=cpu(0))
        video_length = len(vr)

        # safety check
        if idx < 0 or idx >= video_length:
            print(f"[WARN] idx {idx} out of range for {video_path}")
            continue

        # extract frame
        frame = vr[idx]                # decord NDArray
        frame_np = frame.asnumpy()     # (H, W, 3), uint8

        # save image
        img = Image.fromarray(frame_np)
        img.save(img_path)

    except Exception as e:
        print(f"[ERROR] Failed processing {video_path}: {e}")

print("Done!")
