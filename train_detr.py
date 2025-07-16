from torchvision.datasets import CocoDetection
from modules.model import DETR
from torch.utils.data import DataLoader
from torch import optim
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # download dataset
    dataset = CocoDetection(root='./data/coco/images', annFile='./data/coco/annotations.json')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    model = DETR(num_classes=80).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    