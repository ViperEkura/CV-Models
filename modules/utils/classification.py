import torch

from torch import Tensor
from typing import List, Callable, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_fn(
    model: Callable[[Tensor],  Tensor],
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    criterion: Callable[[Tensor, Tensor],  Tensor], 
    epoch: int,
    device: torch.device,
    print_every: int
) -> Tuple[float, float]:
    loss_list = []
    total = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % print_every == 0:
            batch_acc = 100. * correct / total
            epoch_info = f"{epoch} [{batch_idx}/{len(train_loader)}]"
            iteration_info = f'Epoch: {epoch_info} | Loss: {loss.item():.5f} | Acc: {batch_acc:.2f}%'
            print(iteration_info)
    
    train_acc = 100. * correct / total
    train_loss = sum(loss_list) / len(loss_list)
    return train_loss, train_acc


def test_fn(
    model: Callable[[Tensor],  Tensor],
    test_loader: DataLoader,
    criterion: Callable[[Tensor, Tensor],  Tensor],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    acc_info = f'{correct}/{total} ({test_acc:.2f}%)'
    iteration_info = f'Test | Average loss: {test_loss:.4f} | Accuracy: {acc_info}'
    print(iteration_info)
    
    return test_loss, test_acc


def train_loop(
    model: Callable[[Tensor],  Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable[[Tensor, Tensor],  Tensor],
    epochs: int,
    print_every: int,
    device: torch.device,
)-> Tuple[List[float], List[float], List[float], List[float]]:
    
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, epoch, device, print_every)
        test_loss, test_acc = test_fn(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    return train_losses, test_losses, train_accs, test_accs
