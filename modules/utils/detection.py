import torch

from torch import Tensor
from typing import Callable, Tuple, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader



def train_fn(
    model: Callable[[Tensor],  Tensor],
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    criterion: Callable[[Tensor, Tensor],  Tensor], 
    epoch: int,
    device: torch.device,
):
    pass


def test_fn(
    model: Callable[[Tensor],  Tensor],
    test_loader: DataLoader,
    criterion: Callable[[Tensor, Tensor],  Tensor],
    device: torch.device,
) -> Tuple[float, float]:
    pass


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
    pass
