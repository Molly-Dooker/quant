import torch
from ultralytics import YOLO
import ipdb
from datasets import load_dataset

ds = load_dataset(path='rafaelpadilla/coco2017',cache_dir='/Data/Dataset/COCO')

