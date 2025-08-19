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
    train_dataset = VOCDataset(root_dir=dataset_path, split='trainval')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_pad)
 
    model = DETR(num_classes=100).to(device)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                       if not n.find("backbone") and p.requires_grad],
            'lr': 1e-4,
            'weight_decay': 1e-3
        },
        {
            'params': [p for n, p in model.named_parameters() 
                       if n.find("backbone") and p.requires_grad],
            'lr': 1e-5,
            'weight_decay': 1e-3
        }
    ]

    optimizer = optim.AdamW(param_groups)
    matcher = HungarianMatcher(1, 5, 2)
    criterion = SetCriterion(num_classes=100, matcher=matcher, eos_coef=0.01)
    
    avg_loss = []
    for i in range(1, 2):
        loss = train_fn(model, train_loader, optimizer, criterion, i, device, 4)
        avg_loss.append(loss)
    
    torch.save(model.state_dict(), 'detr.pth') 