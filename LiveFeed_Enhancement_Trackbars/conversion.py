import cv2
import numpy as np

def apply_sharpening(img, level=0):
    if level == 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1 + level * 0.2, blurred, -level * 0.2, 0)
    return sharpened

def apply_contrast(img, clip_limit=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(clip_limit, 1.0), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_brightness(img, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.1)
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_saturation(img, saturation_scale=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_denoising(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Sharpness", "Controls", 0, 5, nothing)
cv2.createTrackbar("Contrast", "Controls", 20, 100, nothing)
cv2.createTrackbar("Gamma", "Controls", 10, 50, nothing)
cv2.createTrackbar("Saturation", "Controls", 10, 30, nothing)
cv2.createTrackbar("Denoise", "Controls", 0, 1, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    
    sharp_level = cv2.getTrackbarPos("Sharpness", "Controls")
    clahe_clip = cv2.getTrackbarPos("Contrast", "Controls") / 10.0
    gamma_value = cv2.getTrackbarPos("Gamma", "Controls") / 10.0
    saturation_scale = cv2.getTrackbarPos("Saturation", "Controls") / 10.0
    denoise_flag = cv2.getTrackbarPos("Denoise", "Controls")

    raw = frame.copy()

    final = frame.copy()
    final = apply_sharpening(final, sharp_level)
    final = apply_contrast(final, clahe_clip)
    final = apply_brightness(final, gamma_value)
    final = apply_saturation(final, saturation_scale)
    if denoise_flag:
        final = apply_denoising(final)
        q
    combined = np.hstack((raw, final))
    cv2.imshow("Camera Feed: Raw | Final", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

