import cv2

class SecurityCapture(cv2.VideoCapture):
  def __init__(self, url: str, frame_interval = 1):
    self.frame_interval = frame_interval
    self.url = url
    self.frame_number = 0
    super().__init__(url)

  def read(self):
    while True:
      self.frame_number += 1
      ret1 = super().grab()
      if (self.frame_number % self.frame_interval == 0):
        ret2, frame = super().retrieve()
        return ret1 and ret2, frame
