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
        im = self.pre_transform(im)
        im = np.stack(im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        # np.save(file=root+f'im2',arr=im.clone().detach().cpu().numpy())
        return im
    def pre_transform(self,im):
        result = [self.letterbox(image=x) for x in im]
        return result

def transform(data_batch, processor):
    # ipdb.set_trace()
    origin_shape=[]
    image_ = []
    data_batch['origin_image'] = data_batch['image']
    for im in data_batch['image']:
        im = np.array(im.convert('RGB'))[..., ::-1]
        image_.append(im)
        origin_shape.append(im.shape)
    image_ = processor(image_)
    data_batch["image"] = image_
    data_batch['origin_shape'] = origin_shape
    return data_batch

def custom_collate_fn(batch):
    # batch는 각 transform 결과를 담은 dict들의 리스트입니다.
    
    # image는 텐서이므로 torch.stack으로 배치화합니다.
    images = torch.stack([item['image'] for item in batch])
    
    # image_id는 리스트 형태로 묶습니다.
    image_ids = [item['image_id'] for item in batch]
    origin_shape = [item['origin_shape'] for item in batch]
    origin_image = [item['origin_image'] for item in batch]
    # objects 역시 리스트로 묶습니다.
    objects = [item['objects'] for item in batch]
    
    return {
        'image': images,
        'image_id': image_ids,
        'objects': objects,
        'origin_shape': origin_shape,
        'origin_image':origin_image
    }

# if __name__ == '__main__':
    # rest
    # ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO',split='val')   
    # processor = Processor()
    # prepared_ds = ds.with_transform(lambda batch: transform(batch, processor))
    # dataloader = torch.utils.data.DataLoader(prepared_ds, batch_size=512, shuffle=True, collate_fn=custom_collate_fn)