from torch import Tensor


def postprocess_detection(
    pred_class: Tensor, 
    pred_bbox: Tensor, 
    threshold: float
):
    """
    Args:
        pred_class: shape [batch_size, num_queries, num_classes+1]
        pred_bbox:  shape [batch_size, num_queries, 4]
        threshold: threshold of classification score
    Returns:
        filtered_class: list
        filtered_bbox: list
    """
    
    pred_class = pred_class.softmax(dim=-1)
    
    max_prob, max_idx = pred_class.max(-1)
    keep = (max_idx != 0) & (max_prob > threshold)
    
    batch_size = pred_class.shape[0]
    filtered_classes = []
    filtered_bboxes = []
    
    for i in range(batch_size):
        filtered_classes.append(pred_class[i][keep[i]])
        filtered_bboxes.append(pred_bbox[i][keep[i]])
    
    return filtered_classes, filtered_bboxes
