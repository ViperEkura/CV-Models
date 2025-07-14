import torch

from torch import Tensor
from typing import List, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_fn(
    model: Callable[[Tensor],  Tensor],
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    criterion: Callable[[Tensor, Tensor],  Tensor], 
    epoch: int,
    print_every: int
) -> List[int]:
    loss_list = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        
        if batch_idx % print_every == 0:
            epoch_info = f"{epoch} [{batch_idx}/{len(train_loader)}]"
            iteration_info = f'\nEpoch: {epoch_info} | Loss: {loss.item():.5f}'
            print(iteration_info)
    
    return loss_list

def test_fn(
    model: Callable[[Tensor],  Tensor],
    test_loader: DataLoader,
    criterion: Callable[[Tensor, Tensor],  Tensor],
    epoch: int,
):
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    acc_info = f'{correct}/{total} ({test_acc:.2f}%)'
    iteration_info = f'\nEpoch: {epoch} | Average loss: {test_loss:.4f} | Accuracy: {acc_info}'
    print(iteration_info)
    
    return test_loss, test_acc
