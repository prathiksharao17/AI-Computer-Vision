import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("/home/prathiksha/Downloads/best.pt")

# Load your image
image_path = "/home/prathiksha/Downloads/traffic1.png"  
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError("Image not found at the specified path.")

# Initialize variables
drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = 0, 0, 0, 0
roi_selected = False
display_frame = frame.copy()

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, rx, ry, rw, rh, drawing, roi_selected, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx, ry = min(ix, x), min(iy, y)
            rw, rh = abs(x - ix), abs(y - iy)
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = min(ix, x), min(iy, y)
        rw, rh = abs(x - ix), abs(y - iy)
        roi_selected = True
        process_roi()

def process_roi():
    global display_frame
    roi = frame[ry:ry+rh, rx:rx+rw]
    results = model(roi)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw on the full image with ROI offset
            cv2.rectangle(display_frame, (rx + x1, ry + y1), (rx + x2, ry + y2), (0, 0, 255), 2)
            cv2.putText(display_frame, label, (rx + x1, ry + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

while True:
    cv2.imshow("Image", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):  # Clear ROI
        roi_selected = False
        display_frame = frame.copy()

cv2.destroyAllWindows()

