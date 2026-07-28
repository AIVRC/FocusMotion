from __future__ import absolute_import, division, print_function
import sys
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

# ---------------------------
# Add path to frechet_video_distance
# ---------------------------
sys.path.append("/home/yanghaotian/google-research")
from frechet_video_distance import frechet_video_distance as fvd

tf.disable_v2_behavior()

# ---------------------------
# Settings
# ---------------------------
REAL_DIR = "/home/yanghaotian/server_data/yanghaotian/data/testing"
GEN_DIR = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/output/video-20260128/1042-3.5-pose_guider-130200-motion_module-2517/results"

NUM_FRAMES = 16          # I3D recommended
FRAME_SIZE = (224, 224)  # I3D input

# ---------------------------
# Helper function: load video as tensor
# ---------------------------
def load_video(path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # If video too short, repeat last frame
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    # Convert to float32 tensor: [T, H, W, C]
    video = np.stack(frames).astype(np.float32)
    return video
def load_video_uniform_cap(path,
                           num_frames=16,
                           max_frames=300,
                           size=FRAME_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    frames = np.array(frames, dtype=np.float32)

    total = len(frames)

    if total == 0:
        raise ValueError(f"Empty video: {path}")

    # Cap long videos
    if total > max_frames:
        frames = frames[:max_frames]
        total = max_frames

    # Pad short videos
    if total < num_frames:
        pad = num_frames - total
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], pad, axis=0)],
            axis=0
        )
        return frames

    # Uniform sampling
    indices = np.linspace(0, total - 1, num_frames).astype(np.int32)
    sampled = frames[indices]

    return sampled


# ---------------------------
# Load all videos from folder
# ---------------------------
def load_videos_from_folder(folder):
    video_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                          if f.endswith(('.mp4', '.avi', '.mov'))])
    # videos = [load_video(f) for f in video_files]
    videos = [load_video_uniform_cap(f) for f in video_files]

    return np.stack(videos)  # shape: [N, T, H, W, C]

def pad_to_multiple(videos, multiple=16):
    n = videos.shape[0]
    if n % multiple == 0:
        return videos
    pad = multiple - (n % multiple)
    extra = np.repeat(videos[-1:], pad, axis=0)
    return np.concatenate([videos, extra], axis=0)

# ---------------------------
# Main
# ---------------------------
def main(argv):
    del argv

    print("Loading real videos...")
    real_videos = load_videos_from_folder(REAL_DIR)
    print("Loading generated videos...")
    gen_videos = load_videos_from_folder(GEN_DIR)


    real_videos = pad_to_multiple(real_videos)
    gen_videos = pad_to_multiple(gen_videos)


    print(f"Real videos: {real_videos.shape}, Generated videos: {gen_videos.shape}")

    with tf.Graph().as_default():
        real_tensor = tf.convert_to_tensor(real_videos)
        gen_tensor = tf.convert_to_tensor(gen_videos)

        # Preprocess -> I3D embeddings -> FVD
        fvd_value = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(gen_tensor, FRAME_SIZE)),
            fvd.create_id3_embedding(fvd.preprocess(real_tensor, FRAME_SIZE))
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            score = sess.run(fvd_value)
            print("FVD score: %.2f" % score)


if __name__ == "__main__":
    tf.app.run(main)
