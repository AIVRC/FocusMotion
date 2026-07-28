import os
import subprocess
from collections import defaultdict

VIDEO_DIR = "/home/yanghaotian/server_data/yanghaotian/digital_virtual/data/first_data_1015/videos"   # CHANGE THIS

groups = defaultdict(list)
skipped = []

# ---------------------------------
# Extract group key from filename
# ---------------------------------

def extract_group_key(fname):

    name = fname.replace(".mp4", "")

    # Pattern B: xxxx_xxxx_segment_XX
    if "_segment_" in name:
        return name.split("_segment_")[0]

    # Pattern A: xxxx-xxxx-nnn-nnn
    parts = name.split("-")
    if len(parts) >= 4:
        return parts[0] + "-" + parts[1]
    if len(parts) == 3:
        return parts[0]

    return None


# ---------------------------------
# Scan directory
# ---------------------------------

for fname in os.listdir(VIDEO_DIR):

    if not fname.endswith(".mp4"):
        continue

    key = extract_group_key(fname)

    if key is None:
        skipped.append(fname)
        continue

    groups[key].append(fname)


# ---------------------------------
# ffprobe duration
# ---------------------------------

def get_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


# ---------------------------------
# Summary
# ---------------------------------

total_groups = len(groups)
total_videos = sum(len(v) for v in groups.values())

print("\n========== SUMMARY ==========")
print(f"Total groups: {total_groups}")
print(f"Total videos: {total_videos}")
print(f"Average videos per group: {total_videos / total_groups:.2f}")

if skipped:
    print(f"\nSkipped files: {len(skipped)}")
    for f in skipped[:5]:
        print("  ", f)
    if len(skipped) > 5:
        print("  ...")


# ---------------------------------
# Per-group analysis
# ---------------------------------

print("\n========== PER GROUP ==========")

for group, files in sorted(groups.items()):

    max_file = None
    max_dur = 0

    for f in files:
        full_path = os.path.join(VIDEO_DIR, f)
        dur = get_duration(full_path)

        if dur > max_dur:
            max_dur = dur
            max_file = f

    print(f"\nGroup: {group}")
    print(f"Video count: {len(files)}")
    print(f"Longest video: {max_file}")
    print(f"Duration: {max_dur:.2f} seconds")