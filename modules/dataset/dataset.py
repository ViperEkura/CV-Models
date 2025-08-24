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
                class_idx = self.class_to_id_lut.get(class_name, background_idx)
                labels.append(class_idx)
                self.class_counts[class_idx] += 1
                
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

        # 查找图像目录
        image_dir_candidates = glob.glob(os.path.join(root_dir, f'{split}*'))
        if not image_dir_candidates:
            raise ValueError(f"No image directory found for split {split}")
        image_dir = image_dir_candidates[0]

        # 查找标注文件
        anno_dir = os.path.join(root_dir, 'annotations')
        anno_file_candidates = glob.glob(os.path.join(anno_dir, f'instances_{split}*.json'))
        if not anno_file_candidates:
            raise ValueError(f"No annotation file found for split {split}")
        annotation_path = anno_file_candidates[0]

        # 加载COCO数据
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)

        # 预处理：建立类别映射
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        self.classes = ('__background__',) + tuple(cat['name'] for cat in categories)
        self.class_to_id_lut = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class_lut = {idx: cls for idx, cls in enumerate(self.classes)}
        self.class_counts = [0] * len(self.classes)
        
        # 建立类别ID到名称的映射
        category_id_to_name = {cat['id']: cat['name'] for cat in categories}
        # 建立类别ID到索引的直接映射（跳过背景类）
        category_id_to_idx = {cat_id: self.class_to_id_lut[name] 
                            for cat_id, name in category_id_to_name.items()}

        # 预处理：建立图像索引
        image_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # 预处理：建立图像到标注的映射
        image_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_anns:
                image_id_to_anns[img_id] = []
            image_id_to_anns[img_id].append(ann)

        # 初始化图像ID和路径
        self.image_ids = []
        self.image_paths = []
        for img_info in coco_data['images']:
            self.image_ids.append(img_info['id'])
            self.image_paths.append(os.path.join(image_dir, img_info['file_name']))

        # 处理标注信息（优化后的循环）
        self.annotations = []
        for img_id in self.image_ids:
            # 直接通过映射获取，避免嵌套循环
            img_anns = image_id_to_anns.get(img_id, [])
            img_info = image_id_to_info.get(img_id)
            
            if not img_info:
                continue
                
            boxes = []
            labels = []
            original_width = img_info['width']
            original_height = img_info['height']
            
            for ann in img_anns:
                # 直接通过映射获取类别索引，避免嵌套查找
                category_id = ann['category_id']
                cls_idx = category_id_to_idx.get(category_id, 0)  # 0对应背景类
                labels.append(cls_idx)
                
                # 处理边界框坐标
                xmin, ymin, w, h = ann['bbox']
                xmax = xmin + w
                ymax = ymin + h
                
                # 归一化坐标
                x_center = (xmin + xmax) / 2 / original_width
                y_center = (ymin + ymax) / 2 / original_height
                w_norm = w / original_width
                h_norm = h / original_height
                
                boxes.append([x_center, y_center, w_norm, h_norm])

            # 转换为张量
            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
                labels_tensor = torch.tensor(labels, dtype=torch.int64, device=self.device)
            else:
                # 处理没有标注的情况
                boxes_tensor = torch.empty((0, 4), dtype=torch.float32, device=self.device)
                labels_tensor = torch.empty((0,), dtype=torch.int64, device=self.device)
                
            self.annotations.append({
                'boxes': boxes_tensor, 
                'labels': labels_tensor
            })
            
            # 更新类别计数
            for cls_id in labels:
                if cls_id < len(self.class_counts):
                    self.class_counts[cls_id] += 1
