from modules.dataset import download_coco
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
    
    train_dataset = DETRDataset(train_path, annotation_path, mode='train')
    val_dataset = DETRDataset(val_path, annotation_path, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_pad)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_pad)
    
    model = DETR(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    matcher = HungarianMatcher(1, 5, 2)
    criterion = SetCriterion(num_classes=100, matcher=matcher)
    
    train_loop(model, train_loader, test_loader, optimizer, criterion, 1, 1, device)