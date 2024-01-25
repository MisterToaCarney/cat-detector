from capture_interface import SecurityCapture
import cv2
import sys
import collections
import numpy as np
import os

os.makedirs('motion/', exist_ok=True)

frame_offset = 3
q = collections.deque(maxlen=frame_offset + 1)
vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@192.168.1.8/", 1)
# vcap = cv2.VideoCapture(0)

mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = cv2.resize(mask, [1280, 960])
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

def end():
  vcap.release()
  cv2.destroyAllWindows()
  exit()

frame_count = 0

while True:
  frame_count += 1
  ret, frame_fs = vcap.read()
  frame_orig = cv2.resize(frame_fs, [1280, 960])
  frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
  q.append(frame)
  if len(q) <= frame_offset: continue

  old_frame = q.popleft()
  diff = cv2.absdiff(frame, old_frame)
  diff = cv2.bitwise_and(mask, diff)
  diff = cv2.GaussianBlur(diff, (31, 31), 0)
  ret, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
  contours, hr = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  f_x = frame_fs.shape[1] / frame_orig.shape[1]
  f_y = frame_fs.shape[0] / frame_orig.shape[0]

  out = frame_orig.copy()
  for index, contour in enumerate(contours):
    if cv2.contourArea(contour) < 0: continue
    x,y = contour.mean(axis=(0,1), dtype=np.int64)
    box_size = 61
    start_x, end_x = np.clip([x-box_size, x+box_size], 0, frame_orig.shape[1]-1)
    start_y, end_y = np.clip([y-box_size, y+box_size], 0, frame_orig.shape[0]-1)
    fs_start_x, fs_end_x = int(start_x * f_x), int(end_x * f_x)
    fs_start_y, fs_end_y = int(start_y * f_y), int(end_y * f_y)

    cropped = frame_fs[fs_start_y:fs_end_y, fs_start_x:fs_end_x]
    cropped = cv2.resize(cropped, [224,224])
    cv2.imwrite(f"motion/{frame_count}_{index}.jpg", cropped)
    cv2.rectangle(out, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
    cv2.imshow('cropped', cropped)

  cv2.imshow('frame', out)
  if cv2.waitKey(20) == ord('q'): end()