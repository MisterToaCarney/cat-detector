import cv2
from ultralytics import YOLO

vcap = cv2.VideoCapture('rtsp://admin:Drumkit1@192.168.1.8/')

model = YOLO('yolov8l.pt')

def end():
  vcap.release()
  cv2.destroyAllWindows()
  exit()

if not vcap.isOpened(): 
  print("Cannot open camera")
  exit()

frame_count = 0

while True:
  ret = vcap.grab()
  if not ret:
    print("Cant get frame. Exiting...")
    end()

  frame_count += 1

  if (frame_count % 15 == 0): 
    ret, frame = vcap.retrieve()
  else:
    continue

  if not ret:
    print("Cant get frame. Exiting...")
    end()

  resized = cv2.resize(frame, [640, 480])

  results = model.predict(resized, device='cpu')
  result = results[0]

  for index, box in enumerate(result.boxes.xyxy.round().long()):
    cv2.rectangle(resized, box[0:2].numpy(), box[2:4].numpy(), (0,255,0))
    class_name = result.names[int(result.boxes.cls[index])]
    cv2.putText(resized, class_name, box[0:2].numpy(), cv2.QT_FONT_NORMAL, 0.5, (0,255,0), 1, cv2.LINE_AA)

  cv2.imshow('frame', resized)
  if cv2.waitKey(1) == ord('q'): end()

