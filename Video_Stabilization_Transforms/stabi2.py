import cv2
import numpy as np
import math
from tqdm import tqdm

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360

VIDEO_PATH = '/home/prathiksha/Downloads/4763823-uhd_4096_2160_24fps.mp4'
OUTPUT_PATH = 'stabilized_output_resized.mp4'
SIDE_BY_SIDE_PATH = 'side_by_side_output.mp4'

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    return np.convolve(curve, np.ones(window_size) / window_size, mode='same')

def smooth_trajectory(trajectory):
    trajectory = np.array(trajectory)
    smoothed = np.copy(trajectory)
    for i in range(3):  # x, y, angle
        smoothed[:, i] = moving_average(trajectory[:, i], SMOOTHING_RADIUS)
    return smoothed

cap = cv2.VideoCapture(VIDEO_PATH)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (RESIZE_WIDTH, RESIZE_HEIGHT))
side_out = cv2.VideoWriter(SIDE_BY_SIDE_PATH, fourcc, fps, (RESIZE_WIDTH * 2, RESIZE_HEIGHT))

_, prev = cap.read()
prev = cv2.resize(prev, (RESIZE_WIDTH, RESIZE_HEIGHT))
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

transforms = []
trajectory = []

for _ in tqdm(range(n_frames - 1), desc="Calculating transforms"):
    success, curr = cap.read()
    if not success:
        break

    curr = cv2.resize(curr, (RESIZE_WIDTH, RESIZE_HEIGHT))
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        transforms.append([0, 0, 0])
        trajectory.append(trajectory[-1] if trajectory else [0, 0, 0])
        prev_gray = curr_gray
        continue

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]

    if len(valid_prev) < 10:
        m = np.eye(2, 3)
    else:
        m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
        if m is None:
            m = np.eye(2, 3)

    dx = m[0, 2]
    dy = m[1, 2]
    da = math.atan2(m[1, 0], m[0, 0])

    transforms.append([dx, dy, da])

    if trajectory:
        x = trajectory[-1][0] + dx
        y = trajectory[-1][1] + dy
        a = trajectory[-1][2] + da
    else:
        x, y, a = dx, dy, da

    trajectory.append([x, y, a])
    prev_gray = curr_gray

trajectory = np.array(trajectory)
smoothed_trajectory = smooth_trajectory(trajectory)

new_transforms = []
for i in range(len(transforms)):
    diff = smoothed_trajectory[i] - trajectory[i]
    dx = transforms[i][0] + diff[0]
    dy = transforms[i][1] + diff[1]
    da = transforms[i][2] + diff[2]
    new_transforms.append([dx, dy, da])

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in tqdm(range(n_frames - 1), desc="Stabilizing video"):
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    dx, dy, da = new_transforms[i]
    m = np.array([
        [math.cos(da), -math.sin(da), dx],
        [math.sin(da),  math.cos(da), dy]
    ])

    stabilized = cv2.warpAffine(frame, m, (RESIZE_WIDTH, RESIZE_HEIGHT))
    stabilized = stabilized[HORIZONTAL_BORDER_CROP:RESIZE_HEIGHT - HORIZONTAL_BORDER_CROP,
                            HORIZONTAL_BORDER_CROP:RESIZE_WIDTH - HORIZONTAL_BORDER_CROP]
    stabilized = cv2.resize(stabilized, (RESIZE_WIDTH, RESIZE_HEIGHT))

    out.write(stabilized)

    side_by_side = np.hstack((frame, stabilized))
    side_out.write(side_by_side)

    cv2.imshow("Before and After", side_by_side)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
side_out.release()
cv2.destroyAllWindows()

