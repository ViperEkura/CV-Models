import torch
import pytest
from modules.loss.detr_loss import SetCriterion
from modules.model.matcher import HungarianMatcher

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
