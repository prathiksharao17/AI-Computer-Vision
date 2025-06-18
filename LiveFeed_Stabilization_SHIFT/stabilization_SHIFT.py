import cv2
import numpy as np
import math
from collections import deque

SMOOTHING_RADIUS = 10
CROP_BORDER = 20

cap = cv2.VideoCapture(0)  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	

transforms = deque(maxlen=SMOOTHING_RADIUS)
trajectory = np.zeros((3,), dtype=np.float32)

ret, prev = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
last_H = np.eye(3, dtype=np.float32)

while True:
    ret, curr = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        continue

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None:
        continue

    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]

    if len(valid_prev) < 10:
        H = last_H.copy()
    else:
        H, _ = cv2.findHomography(valid_prev, valid_curr, cv2.RANSAC, 5.0)
        if H is None:
            H = last_H.copy()
        else:
            last_H = H.copy()

    dx = H[0, 2]
    dy = H[1, 2]
    da = math.atan2(H[1, 0], H[0, 0])

    trajectory += np.array([dx, dy, da])
    transforms.append(trajectory.copy())

    if len(transforms) < SMOOTHING_RADIUS:
        smooth = trajectory.copy()
    else:
        smooth = np.mean(transforms, axis=0)

    diff = smooth - trajectory
    dx += diff[0]
    dy += diff[1]
    da += diff[2]

    cos_a = math.cos(da)
    sin_a = math.sin(da)
    new_H = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy]
    ], dtype=np.float32)

    stabilized = cv2.warpAffine(prev, new_H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    cropped = stabilized[CROP_BORDER:height - CROP_BORDER, CROP_BORDER:width - CROP_BORDER]
    cropped = cv2.resize(cropped, (width, height))

    stacked = cv2.hconcat([prev, cropped])
    cv2.imshow("Original vs Stabilized (SHIFT)", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev = curr.copy()
    prev_gray = curr_gray.copy()

cap.release()
cv2.destroyAllWindows()

