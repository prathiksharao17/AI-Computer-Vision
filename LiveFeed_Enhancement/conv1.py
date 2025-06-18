import cv2
import numpy as np

def apply_gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / max(gamma, 0.1)
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_brightness(img, brightness_factor=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_saturation(img, saturation_scale=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_sharpening(img, level=1):
    if level <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1 + level * 0.3, blurred, -level * 0.3, 0)
    return sharpened

def apply_clahe(img, clip_limit=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def adaptive_gamma(mean_intensity):
    if mean_intensity < 50:
        return 2.0
    elif mean_intensity < 90:
        return 1.5
    elif mean_intensity > 180:
        return 0.8
    else:
        return 1.0

def smooth(prev, new, weight=0.8):
    return weight * prev + (1 - weight) * new

cap = cv2.VideoCapture(0)
smoothed_gamma = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    raw = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    gamma = adaptive_gamma(mean_intensity)
    smoothed_gamma = smooth(smoothed_gamma, gamma)

    clip_limit = np.interp(mean_intensity, [30, 200], [2.5, 1.0])
    brightness_factor = np.interp(mean_intensity, [30, 200], [1.5, 1.0])
    saturation_scale = np.interp(mean_intensity, [30, 200], [1.6, 1.0])
    sharpen_level = int(np.interp(mean_intensity, [30, 200], [2, 1]))

    processed = apply_clahe(frame, clip_limit=clip_limit)
    processed = apply_gamma_correction(processed, gamma=smoothed_gamma)
    processed = apply_brightness(processed, brightness_factor=brightness_factor)
    processed = apply_saturation(processed, saturation_scale=saturation_scale)
    processed = apply_sharpening(processed, level=sharpen_level)

    combined = np.hstack((raw, processed))

    info_lines = [
        f"Mean Brightness: {mean_intensity:.1f}",
        f"Gamma: {smoothed_gamma:.2f}",
        f"CLAHE Contrast (clipLimit): {clip_limit:.2f}",
        f"Brightness Factor: {brightness_factor:.2f}",
        f"Saturation Scale: {saturation_scale:.2f}",
        f"Sharpness Level: {sharpen_level}"
    ]

    y0, dy = 22, 24
    for i, line in enumerate(info_lines):
        y = y0 + i * dy
        cv2.putText(combined, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Camera Feed: Raw | Enhanced", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

