from modules.dataset.download_coco import download_coco
from modules.dataset.detr_dataset import DETRDataset
from modules.model import DETR
from torch.utils.data import DataLoader
from torch import optim
import torch
import os


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_path = os.path.join(os.getcwd(), 'data', "coco")
    train_path, val_path, annotation_path = download_coco(download_path)
    
    dataset = DETRDataset(train_path, annotation_path, 'instances_train2017.json')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    model = DETR(num_classes=80).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    