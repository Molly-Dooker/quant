from ultralytics import YOLO
from datasets import load_dataset
import ipdb;
model = YOLO("yolov9c.pt")
model.eval()
model.to('cuda:1')
ds = load_dataset(path='rafaelpadilla/coco2017', cache_dir='/Data/Dataset/COCO', split='val')

img = ds[0]['image']

result = model(img,save=True)