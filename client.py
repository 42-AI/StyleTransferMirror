import time
import cv2
from threading import Thread
import torch

FPS = 16
BATCH_PER_SEC = 2

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frames = []
        self.to_process = []
        self.time_slide = 0

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        a = time.time_ns()
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                self.frames.append(self.frame)
                if not len(self.frames) - (FPS // BATCH_PER_SEC):
                    self.to_process.append(self.frames)
                    self.frames = []
                self.time_slide += (time.time_ns() - a) * 1e-9
                if self.time_slide < 1/FPS:
                    time.sleep(1/FPS - self.time_slide)
                    self.time_slide = 0
                else:
                    self.time_slide -= 1/FPS
                a = time.time_ns()

    def stop(self):
        self.stopped = True

class VideoProcess:
    """
    Class that continuously process batch of images with a dedicated thread.
    """

    def __init__(self):
        self.stopped = False
        self.to_process = []
        self.processed = []


    def start(self):    
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
            if len(self.to_process):
                # Do process here
                self.processed.append(self.to_process[0])
                self.to_process = self.to_process[1:]

    def stop(self):
        self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


video_getter = VideoGet(0).start()
video_shower = VideoShow(video_getter.frame).start()
video_processer = VideoProcess().start()

time_slide = 0

while True:
    if video_getter.stopped or video_shower.stopped:
        video_shower.stop()
        video_getter.stop()
        video_processer.stop()
        break
    if (len(video_getter.to_process)):
        for tp in video_getter.to_process:
            video_processer.to_process.append(tp)
        video_getter.to_process = []
    if (len(video_processer.processed)):
        a = time.time_ns()
        _frames = video_processer.processed[0]
        video_processer.processed = video_processer.processed[1:]
        for frame in _frames:
            if video_shower.stopped:
                break
            video_shower.frame = frame
            time_slide += (time.time_ns() - a) * 1e-9
            if time_slide < 1/FPS:
                time.sleep(1/FPS - time_slide)
                time_slide = 0
            else:
                time_slide -= 1/FPS
            a = time.time_ns()