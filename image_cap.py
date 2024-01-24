from capture_interface import SecurityCapture
import cv2
from ultralytics import YOLO
import os
import sys

os.makedirs('images/', exist_ok=True)
os.makedirs('frames/', exist_ok=True)

vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@192.168.1.8/", 10)
model = YOLO('yolov8m.pt')

frame_count = 0

while True:
  frame_count += 1
  ret, frame = vcap.read()

  resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
  resized = resized[32*4:, :] # crop

  results = model.predict(resized, device='cpu', imgsz=(resized.shape[0], resized.shape[1]))
  result = results[0]

  all_classes = []
  for index, box in enumerate(result.boxes.xyxy.round().long()):
    class_name = result.names[int(result.boxes.cls[index])]
    if class_name in ['car', 'traffic light', 'bus', 'potted plant', 'tv']: continue
    all_classes.append(class_name)
    y_s, x_s, y_e, x_e = box.tolist()
    cropped_frame = resized[x_s:x_e,y_s:y_e]
    cv2.imwrite(f"images/{class_name}_{frame_count}_{index}.jpg", cropped_frame)
  
  if len(all_classes) > 0:
    classes_str = "_".join(all_classes)
    cv2.imwrite(f"frames/{classes_str}-{frame_count}.jpg", frame)
