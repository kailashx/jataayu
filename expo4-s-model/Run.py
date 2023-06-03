import time
import traceback
import stomp

import cv2
import torch
from torch import nn

from Analyser import Analyser
from Capture import Capture, CamStream
from Reporting import send_email

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


def report():
    print("Reporting...")
    send_email()


# Main function
if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()  # start @ the beginning
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

    # PATH = r'expo4-s-model/jatayu_model'
    PATH = r'./jatayu_model'
    model = Analyser()
    # anomaly_optimizer = torch.optim.Adam(anomaly_net.parameters(), lr=1e-03)
    criterion = nn.L1Loss()

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    report_c = 0
    try:

        stream1.start()
        stream2.start()
        cap.start()

        while run_signal:
            frame = cap.read_frame()
            # detect_change(footage)
            # Quit when 'x' is pressed
            output = model(frame)
            loss = criterion(output, frame)
            if (loss > 0.16) & (report_c % 3 == 0):
                report()
                report_c = report_c + 1
            frame_processed = frame_processed + 1
            print(f"Frame_processed = {frame_processed}, Loss = {loss}")

    except Exception as e:
        print("ERROR @ Main:", e)
        traceback.print_exc()

    # Release and close stream
    print('Shutting down...')
    stream1.stop()
    stream2.stop()
    cap.stop()

