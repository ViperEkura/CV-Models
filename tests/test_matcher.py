import torch
import pytest
import numpy as np
from modules.model.matcher import HungarianMatcher, jonker_volgenant, _box_giou


def test_jonker_volgenant():
    """测试Jonker-Volgenant算法实现"""
    # 创建一个简单的成本矩阵
    cost_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)
    
    total_cost, row_indices, col_indices = jonker_volgenant(cost_matrix)
    
    # 验证返回值类型和基本属性
    assert isinstance(total_cost, (int, float))
    assert isinstance(row_indices, np.ndarray)
    assert isinstance(col_indices, np.ndarray)
    assert len(row_indices) == cost_matrix.shape[0]
    assert len(col_indices) == cost_matrix.shape[1]


def test_hungarian_matcher_initialization():
    """测试HungarianMatcher初始化"""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=2.0, cost_giou=3.0)
    
    assert matcher.cost_class == 1.0
    assert matcher.cost_bbox == 2.0
    assert matcher.cost_giou == 3.0


def test_hungarian_matcher_matching():
    """测试匈牙利匹配算法"""
    matcher = HungarianMatcher()
    
    # 创建模拟预测数据
    batch_size = 2
    num_queries = 5
    num_classes = 10
    
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_bbox = torch.rand(batch_size, num_queries, 4)
    
    # 创建模拟真实标签
    gt_class = torch.randint(1, num_classes + 1, (batch_size, 3))  # 每张图片3个对象
    gt_bbox = torch.rand(batch_size, 3, 4)
    
    # 执行匹配
    row_inds, col_inds = matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # 验证返回值
    assert isinstance(row_inds, torch.Tensor)
    assert isinstance(col_inds, torch.Tensor)
    assert row_inds.shape == (batch_size, num_queries)
    assert col_inds.shape == (batch_size, 3)


def test_box_giou():
    """测试GIoU计算函数"""
    # 创建两个相同的边界框
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    
    giou = _box_giou(boxes1, boxes2)
    
    # 相同的框应该有GIoU值为1
    assert torch.allclose(giou, torch.tensor([[1.0]]), atol=1e-6)
    
    # 创建两个不相交的边界框
    boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    boxes2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    
    giou = _box_giou(boxes1, boxes2)
    
    # 不相交的框应该有负的GIoU值
    assert (giou < 1).all()
    assert (giou >= -1).all()


def test_matcher_cost_weight_dict():
    """测试匹配器权重字典"""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=2.0, cost_giou=3.0)
    weight_dict = matcher.get_cost_weight_dict()
    
    assert isinstance(weight_dict, dict)
    assert weight_dict["class"] == 1.0
    assert weight_dict["bbox"] == 2.0
    assert weight_dict["giou"] == 3.0