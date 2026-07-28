#!/bin/bash

SRC_DIR="/home/yanghaotian/.cache/kagglehub/datasets/yasaminjafarian/tiktokdataset/versions/3/TikTok_dataset/TikTok_dataset"
DST_DIR="/home/yanghaotian/server_data/yanghaotian/data/applied_dataset/pose"

mkdir -p "$DST_DIR"

for i in $(seq 101 2 199); do
    # Convert to 5-digit format (101 -> 00101)
    dir=$(printf "%05d" $i)

    img="$SRC_DIR/$dir/images/0001.png"

    if [ -f "$img" ]; then
        cp "$img" "$DST_DIR/${dir}_0001.png"
        echo "Copied: $dir"
    else
        echo "Missing: $dir"
    fi
done