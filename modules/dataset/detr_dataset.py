import os
import json
from typing import Tuple, Dict
from PIL import Image
import torch
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


class DETRDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        anno_dir: str,
        anno_file: str,
        image_size: Tuple[int, int] = (224, 224),
        mode: str = 'train'
    ):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.anno_file = anno_file
        self.image_size = image_size
        self.mode = mode
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with open(os.path.join(self.anno_dir, self.anno_file), 'r') as f:
            self.annotations = json.load(f)
        
        self.image_ids = [img['id'] for img in self.annotations['images']]
        if 'train' in self.anno_file:
            self.image_subdir = 'train2017'
        elif 'val' in self.anno_file:
            self.image_subdir = 'val2017'
        else:
            raise ValueError("Unsupported annotation file name.")


    def __len__(self) -> int:
        return len(self.image_ids)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        h, w = self.image_size
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        
        image = self.transform(image)
        
        return image, labels, boxes