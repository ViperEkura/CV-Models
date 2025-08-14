from modules.dataset.dataset import VOCDataset, collate_fn_pad
from modules.dataset.download import download_voc
from modules.model import DETR, HungarianMatcher
from modules.loss.detr_loss import SetCriterion
from modules.utils.detection import train_fn
from torch.utils.data import DataLoader
from torch import optim
import torch
import os


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    download_voc(os.path.join(os.getcwd(), 'data', 'voc'))
    dataset_path = os.path.join(os.getcwd(), 'data', 'voc', 'VOC2012')
    train_dataset = VOCDataset(root_dir=dataset_path, split='train')
    val_dataset = VOCDataset(root_dir=dataset_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_pad)
    test_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_pad)
    
    model = DETR(num_classes=100).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    matcher = HungarianMatcher(1, 5, 2)
    criterion = SetCriterion(num_classes=100, matcher=matcher, eos_coef=0.1)
    
    train_fn(model, train_loader, optimizer, criterion, 1, device, 4)
    torch.save(model.state_dict(), 'detr.pth')