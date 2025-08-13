import torch
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Module
from typing import Callable, Literal, Union, Tuple

def plot_detection(
    model: Union[Module, Callable[..., Tuple[Tensor, Tensor]]], 
    image: Tensor, 
    threshold=0.5, 
    device: Literal["cpu", "cuda"] = "cpu"
):
    model.to(device)
    pred_class, pred_bbox = model(image.to(device), inference=True, threshold=threshold)
    