import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class VOCDataset(Dataset):
    
    # VOC数据集中包含的类别
    CLASSES = (
        '__background__',  # 背景类别
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    )
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 构建图像和标注路径
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.split_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        
        # 读取图像列表
        split_file = os.path.join(self.split_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
            
        # 创建类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        image = self._load_image(image_id)
        target = self._load_target(image_id, idx)

        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target
    
    def _load_image(self, image_id):
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        return image
    
    def _load_target(self, image_id, idx):  
        annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 提取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 提取所有对象
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_to_idx:
                labels.append(self.class_to_idx[class_name])
            else:
                continue 
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)
            
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.as_tensor([idx]),  # 现在 idx 已正确定义
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8)
        }
        
        return target
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))