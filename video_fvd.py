from __future__ import absolute_import, division, print_function
import sys
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

# ----------------------------
# Add Google Research path
# ----------------------------
sys.path.append("/home/yanghaotian/google-research")
from frechet_video_distance import frechet_video_distance as fvd

tf.disable_v2_behavior()

# ----------------------------
# Paths
# ----------------------------
REAL_DIR = "/home/yanghaotian/server_data/yanghaotian/data/testing"
GEN_DIR = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/video-20260128/1042-3.5-pose_guider-130200-motion_module-2517/results"

# ----------------------------
# Parameters
# ----------------------------
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)
BATCH_SIZE = 16   # REQUIRED by FVD

# ----------------------------
# Uniform frame sampler
# ----------------------------
def load_video(path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    cur = 0
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cur == indices[idx]:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            idx += 1
            if idx >= len(indices):
                break

        cur += 1

    cap.release()

    # pad if too short
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames).astype(np.float32)

# ----------------------------
# Repeat single video to batch
# ----------------------------
def make_batch(video, batch_size=BATCH_SIZE):
    return np.repeat(video[np.newaxis], batch_size, axis=0)

# ----------------------------
# Main
# ----------------------------
def main(argv):
    del argv

    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.endswith(".mp4")])
    gen_files = sorted([f for f in os.listdir(GEN_DIR) if f.endswith(".mp4")])

    assert len(real_files) == len(gen_files), "Video count mismatch!"

    print("Total pairs:", len(real_files))

    scores = []

    with tf.Graph().as_default():

        real_ph = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, 224, 224, 3])
        gen_ph  = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FRAMES, 224, 224, 3])

        fvd_value = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(gen_ph, FRAME_SIZE)),
            fvd.create_id3_embedding(fvd.preprocess(real_ph, FRAME_SIZE))
        )

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for i, (rname, gname) in enumerate(zip(real_files, gen_files)):

                print(f"[{i+1}/{len(real_files)}] Processing:", rname)

                real_video = load_video(os.path.join(REAL_DIR, rname))
                gen_video  = load_video(os.path.join(GEN_DIR, gname))

                real_batch = make_batch(real_video)
                gen_batch  = make_batch(gen_video)

                score = sess.run(fvd_value, feed_dict={
                    real_ph: real_batch,
                    gen_ph: gen_batch
                })

                print("   FVD:", round(float(score), 2))

                scores.append((rname, float(score)))

    # ----------------------------
    # Save results
    # ----------------------------
    with open("per_video_fvd.txt", "w") as f:
        for name, val in scores:
            f.write(f"{name}\t{val:.2f}\n")

    print("\n===== SUMMARY =====")
    values = [s[1] for s in scores]
    print("Mean FVD:", np.mean(values))
    print("Median FVD:", np.median(values))
    print("Min FVD:", np.min(values))
    print("Max FVD:", np.max(values))


if __name__ == "__main__":
    tf.app.run(main)
