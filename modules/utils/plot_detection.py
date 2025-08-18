import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torch import Tensor
from typing import List, Tuple

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
