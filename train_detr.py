from modules.dataset import COCODataset, download_coco
from modules.model import DETR, HungarianMatcher
from modules.loss import SetCriterion
from modules.utils.detection import train_loop
from modules.utils.plot import plot_loss
from torch.utils.data import DataLoader
from torch import optim
import torch
import os


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_coco(os.path.join(os.getcwd(), 'data', 'coco'))
    dataset_path = os.path.join(os.getcwd(), 'data', 'coco')
    train_dataset = COCODataset(root_dir=dataset_path, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=COCODataset.collate_fn)

    model = DETR(num_classes=20).to(device)
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                       if not n.find("backbone") and p.requires_grad],
            'lr': 1e-5,
            'weight_decay': 1e-4
        },
        {
            'params': [p for n, p in model.named_parameters() 
                       if n.find("backbone") and p.requires_grad],
            'lr': 1e-4,
            'weight_decay': 1e-4
        }
    ]

    optimizer = optim.AdamW(param_groups)
    matcher = HungarianMatcher(1, 5, 2)
    class_counts = train_dataset.get_class_counts()
    class_weight = 1 / (torch.tensor(class_counts, dtype=torch.float) + 1)
    eos_coef = 1 / (sum(class_counts) * 20)
    criterion = SetCriterion(num_classes=20, matcher=matcher, class_weight=class_weight, eos_coef=eos_coef)
    
    train_loss, _ = train_loop(model, train_loader, optimizer, criterion, 10, 4)
    torch.save(model.state_dict(), 'detr.pth')
    plot_loss(train_loss)
    
    