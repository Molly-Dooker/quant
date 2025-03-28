from ultralytics import YOLO

from torch.fx import symbolic_trace
import torch
import ipdb;
from PIL import Image




from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

from yolo import Yolov8





if __name__ == '__main__':
    yolo =  YOLO("yolov8s.pt") 
    model = yolo.model.model
    device = 'cuda:2'   
    yolo.to(device)
    for m in model: print(m.i, type(m))
    

    # y8 = Yolov8(model)
    # ipdb.set_trace()   
    # y8_trace = symbolic_trace(y8)

    im1 = Image.open("im1.jpg")
    im2 = Image.open("im2.jpg")
    im3 = Image.open("im3.jpg")
    im4 = Image.open("im4.jpg")
    ipdb.set_trace()
    results = yolo.predict(source=[im1,im2,im3,im4], save=True)
    for m in model: print(m.i, type(m))
    # ipdb.set_trace()
    




