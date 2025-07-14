import torch.nn as nn

from torch import Tensor
from typing import List, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_fn(
    model: nn.Module,
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
            epoch_info = f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                            f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.5f}'
            print(epoch_info)

    return loss_list


def val_fn(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    criterion: nn.Module,
    print_every=10,
):
    pass