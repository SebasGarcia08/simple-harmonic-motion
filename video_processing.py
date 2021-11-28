import cv2
import numpy as np


def read_video_array(video_fname: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_fname)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video: np.ndarray = np.empty((frame_count, frame_h, frame_w, 3), np.dtype("uint8"))

    ret: bool = True
    for i in range(frame_count):
        ret, img = cap.read()
        video[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not ret:
            break

    return video, fps
