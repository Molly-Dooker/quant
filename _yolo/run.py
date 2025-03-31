from ultralytics import YOLO
from torch.fx import symbolic_trace
import torch
import ipdb;
from PIL import Image
from yolo import Yolov8s
from datasets import load_dataset
from _dataloader import Processor, transform, custom_collate_fn



if __name__ == '__main__':
    device = 'cuda:2'  
    
    yolo =  YOLO("yolov8s.pt")
    yolo.fuse()
    yolo.eval()

    ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO',split='val')   
    processor = Processor()
    prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)


    batch_for_custom = next(iter(dataloader))['image']
    batch_for_origin = ds[:8]['image']

    y8 = Yolov8s(yolo.model.model).to(device)
    yolo.to(device)

    with torch.no_grad():
        batch_for_custom = batch_for_custom.to(device)
        result_custom = y8(batch_for_custom)



    result_origin = yolo.predict(source=batch_for_origin, save=True)    
    ipdb.set_trace()
    




