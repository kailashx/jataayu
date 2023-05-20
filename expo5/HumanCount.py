import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.eval()
x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
predictions = model(x)
print(predictions)