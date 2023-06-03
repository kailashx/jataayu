import traceback
from multiprocessing import Queue

import stomp
import torch
import argparse
import cv2
import detect_utils
from PIL import Image

from expo5.CaptureVisual import CamStream, Capture
from model import get_model

image_file = f''
threshold = 0.2

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
'''
# read the image
image = Image.open(image_file)
image = image.resize((image.height // 2, image.width // 2))
# detect outputs
boxes, classes, labels = detect_utils.predict(image, model, device, threshold)
print(boxes, classes, labels)
# draw bounding boxes
image = detect_utils.draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
cv2.waitKey(0)
'''
run_signal = True


class Listener(stomp.ConnectionListener):
    # Override the methods on_error and on_message provides by the
    # parent class
    def on_error(self, frame):
        print('received an error "%s"' % frame)

        # Print out the message received

    def on_message(self, frame):
        global run_signal
        print('received a message "%s"' % frame)
        run_signal = False

    def on_send(self, frame):
        print('sent a message "%s"' % frame)


def send_email():
    pass


def report():
    print("Reporting...")
    send_email()


# Main function
if __name__ == "__main__":

    _username = 'admin'
    _password = 'Qf13SGz:XHj97P8N'
    _endpoint = 'mode=real&idc=1&ids=1'
    _ip = '192.168.1.12:554'

    _username1 = 'admin'
    _password1 = 'Qf12SGz:XHj97P8N'
    _endpoint1 = 'mode=real&idc=1&ids=1'
    _ip1 = '192.168.1.11:554'

    _cam1 = f'rtsp://{_username1}:{_password1}@{_ip1}/{_endpoint1}'
    _cam2 = f'rtsp://{_username}:{_password}@{_ip}/{_endpoint}'

    stream1 = CamStream(_cam1)
    stream2 = CamStream(_cam2)
    cap = Capture([stream1, stream2])

    frame_processed = 0

    hosts = [('localhost', 61613)]
    conn = stomp.Connection(host_and_ports=hosts)
    conn.set_listener('', Listener())
    conn.connect('admin', 'admin', wait=True)
    conn.subscribe(destination='/queue/shutdown', id='1', ack='auto')


    report_c = 0
    try:

        cap.start()

        while run_signal:
            frame = cap.read_frame()
            # detect_change(footage)
            # Quit when 'x' is pressed
            boxes, classes, labels = detect_utils.predict(frame, model, device, threshold)
            # draw bounding boxes
            image = detect_utils.draw_boxes(boxes, classes, labels, frame)
            frame_processed = frame_processed + 1
            print(f"Frame_processed = {frame_processed}, Hash = {hash(frame)}")

    except Exception as e:
        print("ERROR @ Main:", e)
        traceback.print_exc()

    # Release and close stream
    print('Shutting down...')
    cap.stop()
    stream1.stop()
    stream2.stop()

