import cv2

# RTSP info -- change these 4 values according to your RTSP URL
username = 'admin'
password = 'Qf13SGz:XHj97P8N'
endpoint = 'mode=real&idc=1&ids=1'
ip = '192.168.1.12:554'

username1 = 'admin'
password1 = 'Qf12SGz:XHj97P8N'
endpoint1 = 'mode=real&idc=1&ids=1'
ip1 = '192.168.1.11:554'

stream1 = cv2.VideoCapture(f'rtsp://{username1}:{password1}@{ip1}/{endpoint1}')
stream2 = cv2.VideoCapture(f'rtsp://{username}:{password}@{ip}/{endpoint}')

