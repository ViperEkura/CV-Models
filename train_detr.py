from modules.dataset.download_coco import download_coco
from modules.dataset.detr_dataset import DETRDataset, collate_fn_pad
from modules.model import DETR, HungarianMatcher
from modules.loss.detr_loss import SetCriterion
from modules.utils.detection import train_loop
from torch.utils.data import DataLoader
from torch import optim
import torch
import os


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_path = os.path.join(os.getcwd(), 'data', "coco")
    train_path, val_path, annotation_path = download_coco(download_path)
    
    train_dataset = DETRDataset(train_path, annotation_path, 'instances_train2017.json')
    val_dataset = DETRDataset(val_path, annotation_path, 'instances_val2017.json')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_pad)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_pad)
    
    model = DETR(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    matcher = HungarianMatcher(2, 1, 1)
    criterion = SetCriterion(
        num_classes=100, 
        matcher=matcher, 
        weight_dict={'loss_class': 2.0, 'loss_bbox': 1.0, 'loss_giou': 1.0}
    )
    
    train_loop(model, train_loader, test_loader, optimizer, criterion, 1, 1, device)