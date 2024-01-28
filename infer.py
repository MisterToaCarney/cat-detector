from capture_interface import SecurityCapture
import cv2
import sys
import collections
from queue import Queue
import numpy as np
import threading
import os
import time

frame_offset = 1
frame_count = 0
frame_q = collections.deque(maxlen=frame_offset + 1)
job_q = Queue()

vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@192.168.1.8/", 3)
# vcap = cv2.VideoCapture(0)

event_count_lock = threading.Lock()
event_count = 0

terminating = threading.Event()

os.makedirs('detections/', exist_ok=True)

mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = cv2.resize(mask, [1280, 960])
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

def end():
  vcap.release()
  cv2.destroyAllWindows()
  terminating.set()
  event_count_thread.join()
  inference_thread.join()
  exit()

def label_path_func(path): # this is required to load the learner
  return path.parent.name

def do_inference(queue: Queue):
  global event_count
  from fastai.vision.learner import load_learner
  learn = load_learner('models/catsec-v3.pkl')
  print("AI is loaded")
  while True:
    if terminating.is_set(): return

    jobs = []
    while not queue.empty(): jobs.append(queue.get())
    
    if len(jobs) == 0: continue

    dl = learn.dls.test_dl(jobs)
    with learn.no_bar(): preds, _, dec_preds = learn.get_preds(dl=dl, with_decoded=True, inner=False)
    predictions = np.array([learn.dls.vocab[int(pred)] for pred in dec_preds])

    with event_count_lock: event_count += np.sum(predictions == 'cat')

    for index, pred in enumerate(predictions):
      if pred != 'cat': continue
      cv2.imwrite(f"detections/detected_{int(time.time()*1000)}.jpg", jobs[index])

def measure_event_rate():
  global event_count
  while True:
    if terminating.is_set(): return
    time.sleep(1)
    with event_count_lock:
      print("Counts per second", event_count)
      event_count = 0


inference_thread = threading.Thread(group=None, target=do_inference, args=(job_q,))
event_count_thread = threading.Thread(group=None, target=measure_event_rate)
inference_thread.start()
event_count_thread.start()

try:
  while True:
    frame_count += 1
    ret, frame_fs = vcap.read()
    frame_orig = cv2.resize(frame_fs, [1280, 960])
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    frame_q.append(frame)
    if len(frame_q) <= frame_offset: continue

    old_frame = frame_q.popleft()
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
      
      # cv2.rectangle(out, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
      # cv2.imshow('cropped', cropped)
      job_q.put_nowait(cropped)

    # cv2.imshow('frame', out)
    # if cv2.waitKey(1) == ord('q'): end()

except KeyboardInterrupt:
  end()
