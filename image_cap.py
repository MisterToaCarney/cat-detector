from capture_interface import SecurityCapture
import cv2
from ultralytics import YOLO
import os

os.makedirs('images/', exist_ok=True)

vcap = SecurityCapture('rtsp://admin:Drumkit1@192.168.1.8/', 10)
model = YOLO('yolov8m.pt')

frame_count = 0

while True:
  frame_count += 1
  ret, frame = vcap.read()

  resized = cv2.resize(frame, [640, 480])

  results = model.predict(resized, device='cpu')
  result = results[0]

  for index, box in enumerate(result.boxes.xyxy.round().long()):
    class_name = result.names[int(result.boxes.cls[index])]
    if class_name in ['car', 'traffic light', 'bus', 'potted plant']: continue
    y_s, x_s, y_e, x_e = box.tolist()
    cropped_frame = resized[x_s:x_e,y_s:y_e]
    cv2.imwrite(f"images/{class_name}_{frame_count}_{index}.jpg", cropped_frame)
