import torch
import torch.nn.functional as F

from torch import Tensor
from modules.model.matcher import HungarianMatcher, jonker_volgenant
from modules.loss.detr_loss import SetCriterion
from modules.utils.box_ops import box_giou


def manual_detr_loss(
    pred_class: Tensor, 
    pred_bbox: Tensor, 
    gt_class: Tensor, 
    gt_bbox: Tensor, 
    cost_class=1, 
    cost_bbox=5, 
    cost_giou=2,
    eos_coef=0.1,
):
    bs = pred_class.shape[0]
    
    # 进行匈牙利匹配
    matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
    row_inds, col_inds = matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # 计算分类损失
    class_loss = 0.0
    bbox_loss = 0.0
    giou_loss = 0.0
    
    for i in range(bs):
        # 获取匹配结果
        matched_row = row_inds[i]
        matched_col = col_inds[i]
        
        # 分类损失计算
        target_class = torch.zeros(pred_class.size(1), dtype=torch.long)
        target_class[matched_row] = gt_class[i][matched_col]
        class_weight = torch.ones_like(target_class)
        class_weight[0] = eos_coef
        matched_class_loss = F.cross_entropy(
            pred_class[i],
            target_class,
            weight=class_weight,
            reduction="mean"
        )
        class_loss += matched_class_loss
        
        # 边界框损失计算 (L1损失)
        matched_pred_bbox = pred_bbox[i][matched_row]
        matched_gt_bbox = gt_bbox[i][matched_col]
        bbox_loss += torch.abs(matched_pred_bbox - matched_gt_bbox).mean()
        
        # GIoU损失计算
        giou = box_giou(matched_pred_bbox, matched_gt_bbox)
        giou_loss += (1 - giou).mean()
    
    # 平均批次损失
    total_loss = (cost_class * class_loss + cost_bbox * bbox_loss + cost_giou * giou_loss) / bs
    return total_loss


def test_detr_loss_initialization():
    """测试SetCriterion初始化"""
    matcher = HungarianMatcher()
    num_classes = 10
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher)
    
    assert criterion.num_classes == num_classes
    assert criterion.matcher == matcher
    assert criterion.eos_coef == 0.1
    assert len(criterion.empty_weight) == num_classes + 1

def test_detr_loss_forward():
    """测试DETR损失前向计算"""
    # 创建matcher和criterion
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=10, matcher=matcher)
    
    # 创建模拟预测数据
    batch_size = 2
    num_queries = 5
    num_classes = 10
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_bbox = torch.rand(batch_size, num_queries, 4)
    
    # 创建模拟真实标签
    gt_class = torch.randint(1, num_classes + 1, (batch_size, 2))  # 每张图片2个对象
    gt_bbox = torch.rand(batch_size, 2, 4)
    
    # 计算损失
    loss = criterion(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # 验证损失是一个标量张量
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_detr_loss_with_empty_targets():
    """测试当没有目标时的损失计算"""
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=10, matcher=matcher)
    
    # 创建模拟预测数据
    batch_size = 2
    num_queries = 5
    num_classes = 10
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_bbox = torch.rand(batch_size, num_queries, 4)
    
    # 创建空的真实标签
    gt_class = torch.zeros((batch_size, 1), dtype=torch.long)  # 只有背景类别
    gt_bbox = torch.zeros((batch_size, 1, 4))
    
    # 计算损失
    loss = criterion(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # 验证损失计算正常进行
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_detr_loss_with_manual():
    """测试手动计算损失"""
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=10, matcher=matcher)

    batch_size = 2
    num_queries = 5
    num_classes = 10
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_bbox = torch.rand(batch_size, num_queries, 4)
    
    gt_class = torch.randint(1, num_classes + 1, (batch_size, 2))
    gt_bbox = torch.rand(batch_size, 2, 4)

    loss = criterion(pred_class, pred_bbox, gt_class, gt_bbox)
    manual_loss = manual_detr_loss(pred_class, pred_bbox, gt_class, gt_bbox)
    
    assert torch.allclose(loss, manual_loss, atol=1e-4)