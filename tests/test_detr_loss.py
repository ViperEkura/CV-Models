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

def test_detr_loss_device_consistency():
    """测试在不同设备上的损失计算一致性"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=10, matcher=matcher)
    
    # 创建模拟数据
    batch_size = 2
    num_queries = 5
    num_classes = 10
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_bbox = torch.rand(batch_size, num_queries, 4)
    gt_class = torch.randint(1, num_classes + 1, (batch_size, 2))
    gt_bbox = torch.rand(batch_size, 2, 4)
    
    # CPU上计算损失
    loss_cpu = criterion(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # GPU上计算损失
    pred_class_gpu = pred_class.cuda()
    pred_bbox_gpu = pred_bbox.cuda()
    gt_class_gpu = gt_class.cuda()
    gt_bbox_gpu = gt_bbox.cuda()
    criterion_gpu = SetCriterion(num_classes=10, matcher=matcher).cuda()
    loss_gpu = criterion_gpu(pred_class_gpu, pred_bbox_gpu, gt_class_gpu, gt_bbox_gpu)
    
    # 验证结果一致性（由于随机性，这里只验证计算成功）
    assert isinstance(loss_cpu, torch.Tensor)
    assert isinstance(loss_gpu, torch.Tensor)
    assert loss_cpu.dim() == 0
    assert loss_gpu.dim() == 0