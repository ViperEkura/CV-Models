import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader



def train_fn(
    model: Module | Callable[..., Tensor],
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    criterion: Callable[..., Tensor] , 
    epoch: int,
    device: torch.device,
    print_every: int
) -> float:
    model.train()
    loss_list = []
    for batch_idx, (img, label, box) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        box = box.to(device)
        
        pred_class, pred_bbox = model(img)
        loss = criterion(pred_class, pred_bbox, label, box)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        
        if batch_idx % print_every == 0:
            epoch_info = f"{epoch} [{batch_idx}/{len(train_loader)}]"
            iteration_info = f'\nEpoch: {epoch_info} | Loss: {loss.item():.5f}'
            print(iteration_info)
    
    avg_loss = sum(loss_list) / len(loss_list)
    
    return avg_loss


def test_fn(
    model: Module | Callable[..., Tensor],
    test_loader: DataLoader,
    criterion: Callable[...,  Tensor],
    device: torch.device,
) -> float:
    model.eval()
    loss_list = []
    for _, (img, label, box) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)
        box = box.to(device)
        
        pred_class, pred_bbox = model(img)
        loss = criterion(pred_class, pred_bbox, label, box)
        loss_list.append(loss.item())
    
    avg_loss = sum(loss_list) / len(loss_list)
    print(f"Test | Average loss: {avg_loss :.4f}")
    
    return avg_loss


def train_loop(
    model: Module | Callable[..., Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable[[Tensor, Tensor],  Tensor],
    epochs: int,
    print_every: int,
    device: torch.device,
)-> Tuple[List[float], List[float]]:
    train_losses, test_losses = [], []
    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer, criterion, epoch, device, print_every)
        test_loss = test_fn(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
    return train_losses, test_losses