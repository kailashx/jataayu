import time
import traceback

import cv2
import numpy as np
import torch
import csv
from torch import nn

from Analyser import Analyser
from Capture import Capture, CamStream

'''
#define identity mat3(0, 0, 0, 0, 1, 0, 0, 0, 0)
#define edge0 mat3([1, 0, -1], [0, 0, 0], [-1, 0, 1])
#define edge1 mat3([0, 1, 0], [1, -4, 1], [0, 1, 0])
#define edge2 mat3([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])
#define sharpen mat3(0, -1, 0, -1, 5, -1, 0, -1, 0)
#define box_blur mat3([1, 1, 1], [1, 1, 1], [1, 1, 1]) * 0.1111
#define gaussian_blur mat3([1, 2, 1], [2, 4, 2], [1, 2, 1]) * 0.0625
'''


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
    fields = ['Frame', 'Loss']

    # name of csv file
    time.time()
    model_learning_rate = str(time.time())+".csv"
    PATH = r'expo4-s-model/jatayu_model'
    model = Analyser()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    # anomaly_optimizer = torch.optim.Adam(anomaly_net.parameters(), lr=1e-03)
    criterion = nn.L1Loss()

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    file = None
    model.train()
    try:
        file = open(model_learning_rate, 'w')
        # creating a csv dict writer object
        writer = csv.writer(file)
        writer.writerow(fields)

        cap.start()
        while frame_processed <= 100:
            frame = cap.read_frame()
            # detect_change(footage)
            # Quit when 'x' is pressed
            optimizer.zero_grad()
            output = model(frame)
            loss = criterion(output, frame)
            loss.backward()
            optimizer.step()
            frame_processed = frame_processed + 1
            print(f"Frame_processed = {frame_processed}, Loss = {loss}")
            writer.writerow([str(frame_processed), str(loss.item())])

    except Exception as e:
        print("ERROR @ Main:", e)
        traceback.print_exc()

    print('Saving Model')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)
    print(f'Model Saved as {PATH}')
    # Release and close stream
    cap.stop()
    stream1.stop()
    stream2.stop()
    file.close()
    cv2.destroyAllWindows()
