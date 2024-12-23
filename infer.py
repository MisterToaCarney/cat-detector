from capture_interface import SecurityCapture
import cv2
import sys
import collections
from queue import Queue
import numpy as np
import threading
import os
import time
import notifications
import random

vcap = SecurityCapture(f"rtsp://{sys.argv[1]}@127.0.0.1:8554/front", 3)
# vcap = cv2.VideoCapture(0)


frame_offset = 1
frame_count = 0
frame_q = collections.deque(maxlen=frame_offset + 1)
job_q = Queue()
event_count_lock = threading.Lock()
latest_image_lock = threading.Lock()
terminating = threading.Event()
GUI = os.environ.get("GUI", "0") == '1'

event_count = 0
latest_detection_image = None

os.makedirs('detections/', exist_ok=True)
os.makedirs('detections/cat/', exist_ok=True)
os.makedirs('detections/not', exist_ok=True)

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
  global event_count, latest_detection_image
  from fastai.vision.learner import load_learner
  learn = load_learner('models/catsec-vit-v6.pkl')
  print("AI is loaded")
  while True:
    if terminating.is_set(): return
    jobs = []
    while not queue.empty(): jobs.append(queue.get())
    if len(jobs) == 0: continue

    # Randomly drop some frames to avoid overloading CPU
    if len(jobs) > 150:
      random.shuffle(jobs)
      jobs = jobs[:150]

    dl = learn.dls.test_dl(jobs)
    with learn.no_bar(): preds, _, dec_preds = learn.get_preds(dl=dl, with_decoded=True, inner=False)
    preds = np.array(preds)
    predictions = np.array([learn.dls.vocab[int(pred)] for pred in dec_preds])
    detections = np.sum(preds > 0.98, axis=0)
    print(detections)

    for index, pred in enumerate(predictions):
      confidence_int = int(np.rint(preds[index, list(learn.dls.vocab).index(pred)] * 100))
      if pred == 'cat':
        cv2.imwrite(f"detections/cat/{confidence_int}_{int(time.time()*1000)}_cat.jpg", jobs[index])
      elif pred == 'not':
        cv2.imwrite(f"detections/not/{confidence_int}_{int(time.time()*1000)}_not.jpg", jobs[index])

    num_cat_detections = detections[list(learn.dls.vocab).index('cat')]
    
    if num_cat_detections > 0:
      most_confident_job_index = preds.argmax(axis=0)[list(learn.dls.vocab).index('cat')]
      with event_count_lock: event_count += num_cat_detections
      with latest_image_lock: latest_detection_image = jobs[most_confident_job_index]

def notification_handler():
  global event_count, latest_detection_image
  while True:
    if terminating.is_set(): return
    time.sleep(1)
    with event_count_lock:
      if event_count > 1:
        with latest_image_lock:
          encoded_image = cv2.imencode(".jpg", latest_detection_image)[1].tobytes()
          notifications.send_message_with_attachment("Cat detected!", encoded_image)
      event_count = 0

inference_thread = threading.Thread(group=None, target=do_inference, args=(job_q,))
event_count_thread = threading.Thread(group=None, target=notification_handler)
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
      
      if GUI:
        cv2.rectangle(out, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
        cv2.imshow('cropped', cropped)
      job_q.put_nowait(cropped)

    if GUI:
      cv2.imshow('frame', out)
      if cv2.waitKey(1) == ord('q'): end()

except KeyboardInterrupt:
  end()
