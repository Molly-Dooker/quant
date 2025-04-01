root = '/home/lsm0729/Repo/Bos_vision/_optimum/datasets/coco/images/val2017/'
annFile = '/home/lsm0729/Repo/Bos_vision/_optimum/datasets/coco/annotations/instances_val2017.json'
import ipdb;

import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

# 데이터 전처리를 위한 transform 정의 (예: ToTensor 변환)
transform = transforms.Compose([
    transforms.ToTensor()
])

# COCO 데이터셋의 경로 설정
# root: 이미지 파일들이 있는 디렉토리
# annFile: annotation 파일의 경로 (예: instances_train2017.json)
coco_dataset = CocoDetection(
    root=root,
    annFile=annFile,
    transform=transform
)

ipdb.set_trace()