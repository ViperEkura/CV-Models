import os
import json
import torch
import xml.etree.ElementTree as ET

from PIL import Image
from typing import Tuple, Dict

from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset


def collate_fn_pad(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    boxes  = [item[2] for item in batch]

    images = torch.stack(images, dim=0)

    max_boxes = max(len(box) for box in boxes)

    padded_boxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    padded_labels = torch.full((len(batch), max_boxes), 0, dtype=torch.long)

    for i, (box, lab) in enumerate(zip(boxes, labels)):
        padded_boxes[i, :len(box)] = box
        padded_labels[i, :len(lab)] = lab

    return images, padded_labels, padded_boxes


class COCODataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        anno_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        mode: str = 'train'
    ):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.image_size = image_size
        self.mode = mode
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if mode == 'train':
            self.image_subdir = 'train2017'
            self.anno_file = 'instances_train2017.json'
        elif mode == 'val':
            self.image_subdir = 'val2017'
            self.anno_file = 'instances_val2017.json'
        else:
            raise ValueError("Unsupported annotation file name.")
        
        with open(os.path.join(self.anno_dir, self.anno_file), 'r') as f:
            self.annotations = json.load(f)
        
        self.image_ids = [img['id'] for img in self.annotations['images']]


    def __len__(self) -> int:
        return len(self.image_ids)


    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_id = self.image_ids[idx]
        
        image_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)
        
        annos = [anno for anno in 
                 self.annotations['annotations'] if anno['image_id'] == image_id]
        boxes = []
        labels = []
        
        for anno in annos:
            x_min, y_min, w, h = anno['bbox']
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(anno['category_id'])
        
        if not annos:
            boxes = torch.zeros((1, 4), dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        h, w = self.image_size
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        
        image = self.transform(image)
        
        return image, labels, boxes
    
    
class VOCDataset(Dataset):
    
    def __init__(
        self, 
        root_dir, 
        split='train', 
        default_image_size=(224, 224)
    ):
        self.root_dir = root_dir
        self.split = split
        self.default_image_size = default_image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(self.default_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.split_dir = os.path.join(root_dir, 'ImageSets', 'Main')

        # load image names
        split_file = os.path.join(self.split_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # load classes
        classes = set()
        for image_id in self.image_ids:
            annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)

        self.classes = ('__background__',) + tuple(sorted(classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = self._load_image(image_id)
        target = self._load_target(image_id)
        image = self.transform(image)

        return image, target

    def _load_image(self, image_id):
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        return image

    def _load_target(self, image_id):
        annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        # 获取图像原始尺寸
        size = root.find('size')
        original_width = int(size.find('width').text)
        original_height = int(size.find('height').text)

        # 计算缩放比例
        if self.default_image_size is not None:
            new_width, new_height = self.default_image_size
            scale_x = new_width / original_width
            scale_y = new_height / original_height
        else:
            scale_x = scale_y = 1.0

        background_idx = self.class_to_idx['__background__']
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            labels.append(self.class_to_idx.get(class_name, background_idx))
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) * scale_x
            ymin = float(bbox.find('ymin').text) * scale_y
            xmax = float(bbox.find('xmax').text) * scale_x
            ymax = float(bbox.find('ymax').text) * scale_y 
            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int32),
        }

        return target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))