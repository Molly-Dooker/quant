from ultralytics.data.augment import LetterBox
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
import ipdb;
from tqdm import tqdm


class Processor:
    def __init__(self, new_shape=(640,640), stride=32):
        self.letterbox = LetterBox(
            new_shape = new_shape,
            auto      = False, # always 640,640
            stride    = stride,
        )        
    def __call__(self, im):
        im = np.stack(self.pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        return im
    def pre_transform(self,im):
        result = [self.letterbox(image=x) for x in im]
        return result

def transform(batch, processor):
    im = batch["image"]
    im = [np.array(image.convert('RGB')) for image in im]
    inputs = dict()
    inputs['images']  = processor(im)
    inputs["objects"] = batch["objects"]
    return inputs

if __name__ == '__main__':
    # rest
    ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO',split='val')
    
    
    processor = Processor()
    prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    ipdb.set_trace()