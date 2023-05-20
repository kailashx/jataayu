import cv2
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms

from expo3.StaticMonitor import StaticMonitor
from timeit import default_timer as timer

# RTSP info -- change these 4 values according to your RTSP URL
username = 'admin'
password = ''
endpoint = 'mode=real&idc=1&ids=1'
ip = '192.168.1.12:554'

username1 = 'admin'
password1 = 'Qf12SGz:XHj97P8N'
endpoint1 = 'mode=real&idc=1&ids=1'
ip1 = '192.168.1.11:554'

PATH = './jatayu_model'
# Stream
stream = cv2.VideoCapture(f'rtsp://{username1}:{password1}@{ip1}/{endpoint1}')

model = StaticMonitor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
'''
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']
'''
model.train()

transform = transforms.Compose([
    transforms.ToTensor()
])

def pipeline(frame):
    # 144p - 176*144 240p - 320×240 - 480×360 640×360
    frame = cv2.resize(frame, (640, 360))

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# Train on real data
labels = torch.from_numpy(np.array([[0.0, 1.0]]))
# zero the parameter gradients
optimizer.zero_grad()
frame_processed = 0
try:
    while True:
        # Read the input live stream
        ret, frame = stream.read()
        frame = pipeline(frame)
        # Show video frame
        cv2.imshow("Home Security Camera", frame)


        outputs = model(transform(frame))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        frame_processed = frame_processed + 1
        print(f"Frame_processed = {frame_processed}, Loss = {loss}, Output = {outputs}, Label = {labels}")

        # Quit when 'x' is pressed
        if cv2.waitKey(1) & 0xFF == ord('x'):
            print('Saving Model')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, PATH)
            print(f'Model Saved as {PATH}')
            break
except Exception as e:
    print("ERROR:", e)

# Main function
if __name__ == "__main__":
    # Release and close stream
    stream.release()
    cv2.destroyAllWindows()
