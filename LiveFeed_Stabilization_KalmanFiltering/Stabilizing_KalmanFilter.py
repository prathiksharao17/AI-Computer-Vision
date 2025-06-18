import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

Q = np.array([1e-4, 1e-4, 1e-5])
R = np.array([0.25, 0.25, 0.1])
X = np.zeros(3, dtype=np.float32)
P = np.ones(3, dtype=np.float32)
trajectory = np.zeros(3, dtype=np.float32)

alpha = 0.95
ZOOM = 1.03  # slight zoom-in to remove edge artifacts

ret, prev = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera")

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
last_m = np.eye(2, 3, dtype=np.float32)

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

    status = status.flatten()
    prev_pts = prev_pts[status == 1]
    curr_pts = curr_pts[status == 1]

    if len(prev_pts) < 15:
        m = last_m.copy()
    else:
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            m = last_m.copy()
        else:
            last_m = m.copy()

    dx = m[0, 2]
    dy = m[1, 2]
    da = math.atan2(m[1, 0], m[0, 0])
    trajectory += np.array([dx, dy, da])

    X_ = X.copy()
    P_ = P + Q
    K = P_ / (P_ + R)
    z = trajectory.copy()
    X = X_ + K * (z - X_)
    P = (1 - K) * P_

    X = alpha * X + (1 - alpha) * trajectory
    diff = X - trajectory
    dx += diff[0]
    dy += diff[1]
    da += diff[2]

    cos_a = math.cos(da)
    sin_a = math.sin(da)
    m_smooth = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy]
    ], dtype=np.float32)

    stabilized = cv2.warpAffine(prev, m_smooth, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

    center_x, center_y = width // 2, height // 2
    zoom_mat = cv2.getRotationMatrix2D((center_x, center_y), 0, ZOOM)
    stabilized = cv2.warpAffine(stabilized, zoom_mat, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

    stacked = cv2.hconcat([prev, stabilized])
    cv2.imshow("Original vs Stabilized", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray.copy()
    prev = curr.copy()

cap.release()
cv2.destroyAllWindows()

