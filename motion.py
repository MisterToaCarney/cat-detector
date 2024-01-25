from capture_interface import SecurityCapture
import cv2
import sys
import collections
import numpy as np

frame_offset = 6
q = collections.deque(maxlen=frame_offset + 1)
vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@192.168.1.8/", 1)
# vcap = cv2.VideoCapture(0)

mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

def end():
  vcap.release()
  cv2.destroyAllWindows()
  exit()

frame_count = 0
  
while True:
  ret, frame_orig = vcap.read()
  frame_orig = cv2.resize(frame_orig, [1280,960])
  # frame_orig = frame_orig[50:]
  frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
  frame = cv2.GaussianBlur(frame, (15,15), 0)
  q.append(frame)

  if len(q) <= frame_offset: continue

  old_frame = q.popleft()
  old_frame = cv2.bitwise_not(old_frame)

  frame = frame.astype(np.float64) / 255
  old_frame = old_frame.astype(np.float64) / 255
  diff = (frame*0.5) + (old_frame*0.5)

  # ret, thres1 = cv2.threshold(diff, 0.51, 1, cv2.THRESH_BINARY)
  ret, thres = cv2.threshold(diff, 0.49, 1, cv2.THRESH_BINARY_INV)
  # thres = thres1 + thres2
  thres = (thres*255).astype(np.uint8)
  thres = cv2.multiply(thres, mask)
  thres = cv2.GaussianBlur(thres, (31,31), 0)
  

  contours, hr = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  out = frame_orig.copy()

  largest_area = 0
  largest_box = None
  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    area = w*h
    if area > 0:
      out = cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
      if area > largest_area:
        largest_area = area
        largest_box = (x,y,w,h)
    
    
  
  if largest_box:
    x,y,w,h = largest_box
    cropped = frame_orig[y:y+h, x:x+h]
    cropped = cv2.resize(cropped, [224, 224])
    cv2.imshow('cropped', cropped)

  cv2.imshow('difference', thres)
  cv2.imshow('cont', out)
  if cv2.waitKey(1) == ord('q'): end()


