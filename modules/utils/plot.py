import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torch import Tensor
from typing import List, Optional, Tuple

def plot_detection(
    image: Tensor,
    boxes: Tensor,
    labels: Tensor,
    scores: Tensor,
    class_names: List[str],
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 10),
    fontsize: int = 12,
    save_path: str = None,
    show: bool = True,
    save: bool = False,
    title: str = None,
):
    img_ori_h, img_ori_w = image.shape[1], image.shape[2]
    np_image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC

    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    inds = scores > threshold
    boxes = boxes[inds]
    labels = labels[inds]
    scores = scores[inds]

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(np_image)
    ax.axis('off')

    for box, label, score in zip(boxes, labels, scores):
        x, y, w, h = box
        x_min = (x - w / 2) * img_ori_w
        y_min = (y - h / 2) * img_ori_h

        rect = Rectangle(
            (x_min, y_min), 
            w * img_ori_w, 
            h * img_ori_h,
            fill=False, 
            edgecolor='red', 
            linewidth=2
        )
        ax.add_patch(rect)

        ax.text(
            x_min, 
            y_min, 
            f'{class_names[label]} {score:.2f}', 
            color='red', 
            fontsize=fontsize,
            verticalalignment='top'
        )
    if title:
        ax.set_title(title, fontsize=fontsize)

    if save and save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()




def plot_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    epochs: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (10, 6),
    fontsize: int = 12,
    save_path: str = None,
    show: bool = True,
    title: str = "Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    train_label: str = "Training Loss",
    val_label: str = "Validation Loss",
):
    """
    Plot training (and optionally validation) loss curve.

    Args:
        train_losses: List of training loss values for each epoch.
        val_losses: Optional list of validation loss values for each epoch.
        epochs: Optional list of epoch numbers. If None, uses range(1, len(train_losses)+1).
        figsize: Figure size (width, height).
        fontsize: Font size for labels and title.
        save_path: Path to save the plot image. If None, saving is skipped.
        show: Whether to display the plot.
        title: Title of the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        train_label: Legend label for training loss.
        val_label: Legend label for validation loss.
    """
    if epochs is None:
        epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=2, label=train_label)

    if val_losses is not None:
        if len(val_losses) != len(epochs):
            raise ValueError(f"Length of val_losses ({len(val_losses)}) must match length of epochs/train_losses ({len(epochs)}).")
        plt.plot(epochs, val_losses, marker='s', linestyle='--', linewidth=2, label=val_label)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300) 
    if show:
        plt.show()

    plt.close() 