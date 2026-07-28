import cv2
from musepose.utils.util import get_fps, read_frames, save_videos_grid

video_path = "/home/yanghaotian/server_data/yanghaotian/test/MusePose/assets/poses/align/img_91-3003CN5S_frame1_video_91-3003CN5S.mp4"

cap = cv2.VideoCapture(video_path)
print(len(read_frames(video_path)))

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

print("Total frames:", frame_count)