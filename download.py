import os
import pandas as pd
import subprocess
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

CSV_PATH = "/home/yanghaotian/.cache/kagglehub/datasets/rounakbanik/ted-talks/versions/3/ted_main.csv"          # your csv file
URL_COLUMN = "url"            # column name
OUTPUT_DIR = "/home/yanghaotian/server_data/yanghaotian/data/ted/videos"     # download folder
MAX_HEIGHT = None              # set None for best quality

# ==========================
# Create output folder
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Load CSV
# ==========================

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

if URL_COLUMN not in df.columns:
    raise ValueError(f"Column '{URL_COLUMN}' not found in CSV")

urls = df[URL_COLUMN].dropna().astype(str)

print(f"Found {len(urls)} URLs")

# ==========================
# Download Loop
# ==========================

for url in tqdm(urls, desc="Downloading"):

    url = url.strip().replace('"', '').replace('\n', '')

    if not url.startswith("http"):
        print(f"Skipping invalid url: {url}")
        continue

    # Build yt-dlp command
    if MAX_HEIGHT:
        format_selector = f"bv*[height<={MAX_HEIGHT}]+ba/b"
    else:
        format_selector = "bv*+ba/b"

    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "--continue",
        "--no-overwrites",
        "-o", f"{OUTPUT_DIR}/%(title)s.%(ext)s",
        url
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed: {url}")
        continue

print("Download finished ✅")
