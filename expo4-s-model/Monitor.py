import cv2
import numpy as np
import torch
from torch import nn

from Analyser import Analyser
from torchvision import transforms

# RTSP info -- change these 4 values according to your RTSP URL
username = 'admin'
password = ''
endpoint = 'mode=real&idc=1&ids=1'
ip = '192.168.1.12:554'

username1 = 'admin'
password1 = 'Qf12SGz:XHj97P8N'
endpoint1 = 'mode=real&idc=1&ids=1'
ip1 = '192.168.1.11:554'

# Stream
stream1 = cv2.VideoCapture(f'rtsp://{username1}:{password1}@{ip1}/{endpoint1}')
stream2 = cv2.VideoCapture(f'rtsp://{username}:{password}@{ip}/{endpoint}')
model = Analyser()
'''
#define identity mat3(0, 0, 0, 0, 1, 0, 0, 0, 0)
#define edge0 mat3([1, 0, -1], [0, 0, 0], [-1, 0, 1])
#define edge1 mat3([0, 1, 0], [1, -4, 1], [0, 1, 0])
#define edge2 mat3([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])
#define sharpen mat3(0, -1, 0, -1, 5, -1, 0, -1, 0)
#define box_blur mat3([1, 1, 1], [1, 1, 1], [1, 1, 1]) * 0.1111
#define gaussian_blur mat3([1, 2, 1], [2, 4, 2], [1, 2, 1]) * 0.0625
'''
edge_k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

transform = transforms.Compose([
    transforms.ToTensor()
])

def pipeline(frame):
    frame = np.hstack(frame)
    # converting image to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # highlighting edge
    frame = cv2.filter2D(frame, -1, edge_k)

    return transform(frame)


flag = 0
ref_frame = 0


def detect_change(frame):
    global ref_frame
    diff = np.subtract(frame, ref_frame)
    print(np.sum(diff))
    if np.sum(diff) > 47477097:
        report()


def report():
    print("Reporting...")


# Main function
if __name__ == "__main__":
    model.eval()
    criterion = nn.L1Loss(reduction='sum')
    try:
        while True:
            # Read the input live stream and resize the footage into (176*144)
            f = (cv2.resize(stream2.read()[1], (120, 120)), cv2.resize(stream1.read()[1], (120, 120)))

            frame = pipeline(f)

            #cv2.imshow("Home Security Camera", frame)

            # detect_change(footage)
            # Quit when 'x' is pressed
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
    except Exception as e:
        print("ERROR:", e)

    # Release and close stream
    stream1.release()
    stream2.release()
    cv2.destroyAllWindows()
