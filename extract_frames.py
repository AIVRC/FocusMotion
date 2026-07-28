import os
import cv2

# =========================
# Paths
# =========================
VIDEO_DIR = "/home/yanghaotian/server_data/yanghaotian/digital_virtual/data/test"
OUTPUT_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ubc_frames"

FRAME1_DIR = os.path.join(OUTPUT_DIR, "frame1")
FRAME300_DIR = os.path.join(OUTPUT_DIR, "frame300")

PAIRS_TXT = os.path.join(OUTPUT_DIR, "pairs.txt")

FRAME_1_INDEX = 1        # frame 1
FRAME_300_INDEX = 300    # frame 300

# =========================
# Create directories
# =========================
os.makedirs(FRAME1_DIR, exist_ok=True)
os.makedirs(FRAME300_DIR, exist_ok=True)

pairs = []

# =========================
# Process videos
# =========================
for video_name in sorted(os.listdir(VIDEO_DIR)):

    if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    video_base = os.path.splitext(video_name)[0]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️ Failed to open {video_name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= FRAME_300_INDEX:
        print(f"⚠️ {video_name} has only {total_frames} frames, skipped")
        cap.release()
        continue

    saved = {}

    # -------------------------
    # Extract target frames
    # -------------------------
    targets = [
        (FRAME_1_INDEX, "frame1", FRAME1_DIR),
        (FRAME_300_INDEX, "frame300", FRAME300_DIR),
    ]

    for frame_idx, tag, out_dir in targets:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Failed to read frame {frame_idx} from {video_name}")
            break

        out_path = os.path.join(
            out_dir,
            f"{video_base}_{tag}.png"
        )

        cv2.imwrite(out_path, frame)
        saved[tag] = out_path

    cap.release()

    # -------------------------
    # Save pairs
    # -------------------------
    if "frame1" in saved and "frame300" in saved:

        pairs.append(f"{saved['frame1']},{saved['frame300']}")
        print(f"✅ Processed {video_name}")

# =========================
# Write pairs.txt
# =========================
with open(PAIRS_TXT, "w") as f:
    for line in pairs:
        f.write(line + "\n")

print("\n🎉 Done!")
print(f"Frame1 saved to: {FRAME1_DIR}")
print(f"Frame300 saved to: {FRAME300_DIR}")
print(f"Pairs file written to: {PAIRS_TXT}")
print(f"Total valid video pairs: {len(pairs)}")
