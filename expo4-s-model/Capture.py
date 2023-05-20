# RTSP info -- change these 4 values according to your RTSP URL
import time
import traceback

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
        self.stream = cv2.VideoCapture(src)
        try:
            (self.grabbed, self.frame) = self.stream.read()
        except Exception as e:
            print("ERROR @ CamStream:", e)
            traceback.print_exc()
            exit(0)
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
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

            except Exception as e:
                print("ERROR @ CamStream:", e)
                traceback.print_exc()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()



class Capture:
    def __init__(self, streams=None):
        # RTSP info -- change these 4 values according to your RTSP URL
        self.streams = streams
        self.frames = []

        self.stopped = False

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.edge_k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def transform_f(self, frame):
        # converting image to gray scale
        # frame = np.hstack((cv2.resize(frame1, (960, 540)), cv2.resize(frame2, (960, 540))))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # highlighting edge
        frame = cv2.filter2D(frame, -1, self.edge_k)

        return self.transform(frame)

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread

            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            try:
                frames = []
                for stream in self.streams:
                    frames.append(
                        self.transform_f(
                            cv2.resize(stream.read(), (120, 120))
                        )
                    )
                self.frames = torch.stack(frames)
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
        return self.frames
