import torch
from PIL import Image


class YoloModel:

    def __init__(self, yolo_name: str, device: str = 'cpu'):
        self.model = torch.hub.load('ultralytics/yolov5', yolo_name).to(device)

    def __call__(self, img: Image):
        results = self.model(img).xyxy[0]
        results = results[results[:, 5] == 2]
        res = results[torch.argmax(results, dim=0)[4]]
        bbox = res[:4].detach().cpu().numpy()
        return img.crop(bbox)
