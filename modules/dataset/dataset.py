import os
import glob
import json
import torch
import xml.etree.ElementTree as ET

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Callable, List, Tuple, Dict


class DetectionDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str='train', 
        transform: Callable[..., Tensor] = None,
        default_image_size: Tuple[int, int]=(224, 224),
        device="cuda"
    ):
        self.root_dir = root_dir
        self.split = split
        self.default_image_size = default_image_size
        self.device = device
        self.class_to_id_lut: Dict[str, int] = {}
        self.idx_to_class_lut: Dict[int, str] = {}
        self.class_counts: Dict[str, int] = {}
        self.image_paths: List[str] = []
        self.annotations: List[Dict[str, Any]] = []
        
        if not transform:
            self.transform = transforms.Compose([
                transforms.Resize(self.default_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transform
        
    def get_class_counts(self):
        """获取各类别的统计数量"""
        return self.class_counts
    
    def class_to_idx(self, class_name: str):
        return self.class_to_id_lut[class_name]
    
    def idx_to_class(self, idx):
        return self.idx_to_class_lut[idx]

    def load_image(self, idx: int) -> Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).to(self.device)
        
        return  image_tensor
    
    def load_annotation(self, idx: int) -> Dict[str, Any]:
        annotation = self.annotations[idx]
        for key, value in annotation.items():
            if isinstance(value, Tensor):
                annotation[key] = value.to(self.device)
                
        return annotation
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        image = self.load_image(idx)
        annotation = self.load_annotation(idx)
        labels = annotation['labels']
        boxes = annotation['boxes']
        
        return image, labels, boxes

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        boxes  = [item[2] for item in batch]
        
        images = torch.stack(images, dim=0)
        
        return images, labels, boxes


class VOCDataset(DetectionDataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str='train', 
        transform: transforms.Compose = None,
        default_image_size: Tuple[int, int]=(224, 224),
        device="cuda"
    ):
        super().__init__(root_dir, split, transform, default_image_size, device)
        
        image_dir = os.path.join(root_dir, 'JPEGImages')
        anno_dir = os.path.join(root_dir, 'Annotations')
        split_dir = os.path.join(root_dir, 'ImageSets', 'Main')

        # Load image names
        split_file = os.path.join(split_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        # Load classes
        classes = set()
        for image_id in self.image_ids:
            annotation_path = os.path.join(anno_dir, f'{image_id}.xml')
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)

        self.classes = ('__background__',) + tuple(sorted(classes))
        self.class_to_id_lut = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class_lut = {idx: cls for idx, cls in enumerate(self.classes)}   
        self.class_counts = [0] * len(self.classes)
        background_idx = self.class_to_id_lut['__background__']
        
        # load images and annotations
        for image_id in self.image_ids:
            image_path = os.path.join(image_dir, f'{image_id}.jpg')
            self.image_paths.append(image_path)
            
            annotation_path = os.path.join(anno_dir, f'{image_id}.xml')
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            size = root.find('size')
            original_width = int(size.find('width').text)
            original_height = int(size.find('height').text)
            
            boxes = []
            labels = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                labels.append(self.class_to_id_lut.get(class_name, background_idx))
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) / original_width
                ymin = float(bbox.find('ymin').text) / original_height
                xmax = float(bbox.find('xmax').text) / original_width
                ymax = float(bbox.find('ymax').text) / original_height
                
                # xywh format
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
                boxes.append([x_center, y_center, w, h])
                
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)

                annotation = { 'boxes': boxes_tensor, 'labels': labels_tensor}
                self.annotations.append(annotation)


class COCODataset(DetectionDataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str='train', 
        transform: transforms.Compose = None,
        default_image_size: Tuple[int, int]=(224, 224),
        device="cuda"
    ):
        super().__init__(root_dir, split, transform, default_image_size, device)


        image_dir_candidates = glob.glob(os.path.join(root_dir, f'{split}*'))
        if not image_dir_candidates:
            raise ValueError(f"No image directory found for split {split}")
        image_dir = image_dir_candidates[0]


        anno_dir = os.path.join(root_dir, 'annotations')
        anno_file_candidates = glob.glob(os.path.join(anno_dir, f'instances_{split}*.json'))
        if not anno_file_candidates:
            raise ValueError(f"No annotation file found for split {split}")
        annotation_path = anno_file_candidates[0]

        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        self.classes = ('__background__',) + tuple(cat['name'] for cat in categories)
        self.class_to_id_lut = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class_lut = {idx: cls for idx, cls in enumerate(self.classes)}
        self.class_counts = [0] * len(self.classes)

  
        self.image_ids = []
        self.image_paths = []
        for img_info in coco_data['images']:
            self.image_ids.append(img_info['id'])
            self.image_paths.append(os.path.join(image_dir, img_info['file_name']))

        self.annotations = []
        for img_id in self.image_ids:
            img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
            
            boxes = []
            labels = []
            for ann in img_anns:
                category_id = ann['category_id']
                labels.append(self.class_to_id_lut.get(
                    next(cat['name'] for cat in categories if cat['id'] == category_id), 
                    self.class_to_id_lut['__background__']
                ))
                
                xmin, ymin, w, h = ann['bbox']
                xmax = xmin + w
                ymax = ymin + h
                original_width = img_info['width']
                original_height = img_info['height']
                x_center = (xmin + xmax) / 2 / original_width
                y_center = (ymin + ymax) / 2 / original_height
                w_norm = w / original_width
                h_norm = h / original_height
                boxes.append([x_center, y_center, w_norm, h_norm])

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            self.annotations.append({'boxes': boxes_tensor, 'labels': labels_tensor})
            
            for cls_id in labels:
                self.class_counts[cls_id] += 1
