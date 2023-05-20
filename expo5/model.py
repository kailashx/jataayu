import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights


def get_model(device):
    # load the model
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # load the model onto the computation device
    model = model.eval().to(device)
    return model
