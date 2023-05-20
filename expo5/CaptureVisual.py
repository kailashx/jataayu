# RTSP info -- change these 4 values according to your RTSP URL
import multiprocessing
import time
import traceback
from multiprocessing import Queue

import cv2
import numpy as np
import torch
from torchvision import transforms
from threading import Thread


class CamStream:
    """'
    Ref: https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """

    def __init__(self, src):
        # initialize the video camera stream and read the first frame
        # from the stream
        try:
            self.stream = cv2.VideoCapture(src)
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_q = Queue(maxsize=50)
        except Exception as e:
            print("ERROR @ CamStream:", e)
            traceback.print_exc()
            exit(0)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        multiprocessing.Process(target=self.update).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            try:
                (self.grabbed, self.frame) = self.stream.read()
                # if full clear frame_q so next frame can put
                if self.frame_q.full():
                    self.frame_q.queue.clear()
                self.frame_q.put(self.frame)
            except Exception as e:
                print("ERROR @ CamStream:", e)
                traceback.print_exc()

    def read(self):
        # return the frame most recently read
        return self.frame_q.get()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.frame_q.close()
        self.stream.release()


class Capture:
    def __init__(self, streams=None):
        # RTSP info -- change these 4 values according to your RTSP URL
        self.streams = streams
        self.frames = []

        self.stopped = False
        self.frame_q = Queue(maxsize=50)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.edge_k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def transform_f(self, frame):
        # converting image to gray scale
        # frame = np.hstack((cv2.resize(frame1, (960, 540)), cv2.resize(frame2, (960, 540))))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # highlighting edge
        frame = cv2.filter2D(frame, -1, self.edge_k)

        return frame

    def start(self):
        # start the thread to read frames from the video stream
        multiprocessing.Process(target=self.update).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread

            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            try:
                if self.frame_q.full():
                    self.frame_q.queue.clear()
                frames = []
                # capture frames from all cameras
                for stream in self.streams:
                    frames.append(
                        stream.read()
                    )
                self.frames = torch.stack(frames)
                self.frame_q.put(self.frames)
            except Exception as e:
                print("ERROR @ Capture:", e)
                traceback.print_exc()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def read_frame(self):
        wait_timer = len(self.frames)
        while wait_timer == 0:
            wait_timer = len(self.frames)
        return self.frame_q.get()
