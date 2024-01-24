from capture_interface import SecurityCapture
import cv2
from ultralytics import YOLO
import sys

def end():
  vcap.release()
  cv2.destroyAllWindows()

vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@192.168.1.8/", 30)
model = YOLO('yolov8m.pt')
frame_count = 0

while True:
  frame_count += 1
  ret, frame = vcap.read()

  if not ret:
    print("Cant get frame. Exiting...")
    break

  resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
  resized = resized[32*4:, :]
  results = model.predict(resized, device='cpu', imgsz=(resized.shape[0], resized.shape[1]))
  result = results[0]

  for index, box in enumerate(result.boxes.xyxy.round().long()):
    class_name = result.names[int(result.boxes.cls[index])]
    cv2.rectangle(resized, box[0:2].numpy(), box[2:4].numpy(), (0,255,0))
    cv2.putText(resized, class_name, box[0:2].numpy(), cv2.QT_FONT_NORMAL, 0.5, (0,255,0), 1, cv2.LINE_AA)

  cv2.imshow('frame', resized)
  if cv2.waitKey(1) == ord('q'): end()

