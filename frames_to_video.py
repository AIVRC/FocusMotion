import os
import cv2
from tqdm import tqdm

# Input dataset root
input_root = "/home/yanghaotian/.cache/kagglehub/datasets/yasaminjafarian/tiktokdataset/versions/3/TikTok_dataset/TikTok_dataset"

# Output video directory
output_root = "/home/yanghaotian/server_data/yanghaotian/data/TikTok"

os.makedirs(output_root, exist_ok=True)

# Video parameters
FPS = 30              # change if needed
VIDEO_EXT = ".mp4"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # good compatibility


def numeric_sort_key(filename):
    """
    Sort frames like 0001.png, 0002.png ...
    """
    name = os.path.splitext(filename)[0]
    return int(name)


# List all sequence folders (00001, 00002 ...)
sequences = sorted([
    d for d in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, d))
])

print(f"Found {len(sequences)} sequences")

for seq in tqdm(sequences):

    img_dir = os.path.join(input_root, seq, "images")

    if not os.path.exists(img_dir):
        print(f"Skip {seq}: no images folder")
        continue

    frames = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".png")],
        key=numeric_sort_key
    )

    if len(frames) == 0:
        print(f"Skip {seq}: empty folder")
        continue

    # Read first frame to get size
    first_frame_path = os.path.join(img_dir, frames[0])
    frame = cv2.imread(first_frame_path)

    if frame is None:
        print(f"Skip {seq}: cannot read first frame")
        continue

    height, width, _ = frame.shape

    # Output video path
    out_path = os.path.join(output_root, f"{seq}{VIDEO_EXT}")

    writer = cv2.VideoWriter(out_path, FOURCC, FPS, (width, height))

    for fname in frames:
        fpath = os.path.join(img_dir, fname)
        img = cv2.imread(fpath)

        if img is None:
            print(f"Warning: skipped frame {fpath}")
            continue

        writer.write(img)

    writer.release()

print("All videos generated successfully ✅")
