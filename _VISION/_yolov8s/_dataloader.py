from ultralytics.data.augment import LetterBox
import numpy as np
import torch


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
        return im
    def pre_transform(self,im):
        result = [self.letterbox(image=x) for x in im]
        return result

def transform(img, targets, processor):
    img_p = np.array(img.convert('RGB'))[..., ::-1]
    img_p = processor([img_p])
    size = img.size[::-1]
    return (img_p,img,size), targets


def custom_collate_fn(batch):
    image = []
    # image_id = []
    objects = []
    origin_shape = []
    origin_image = []
    for i,item in enumerate(batch):

        image.append(item[0][0].squeeze(0))
        origin_image.append(item[0][1])
        origin_shape.append(item[0][2]  + (3,))
        # image_id.append(item[1][0]['image_id'])


        # objects
        objects_ = {
            'id':[],
            'area':[],
            'bbox':[],
            'label':[],
            'iscrowd':[]
            }

        for t in item[1]:
            objects_['id'].append(t['id'])
            objects_['area'].append(t['area'])  
            objects_['bbox'].append(t['bbox']) 
            objects_['label'].append(t['category_id']) 
            objects_['iscrowd'].append(t['iscrowd']) 
        objects.append(objects_)

    return {
        'image': torch.stack(image),
        # 'image_id': image_id,
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