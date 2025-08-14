import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from modules.model.matcher import HungarianMatcher, jonker_volgenant
from modules.utils.box_ops import box_giou



def manual_match(pred_class, pred_bbox, gt_class, gt_bbox, alpha=1, beta=5, gamma=2):
    B = pred_class.size(0)
    row_inds, col_inds = [], []

    for i in range(B):
        # 提取单样本数据
        pc = pred_class[i]  # [Q, C]
        pb = pred_bbox[i]   # [Q, 4]
        gc = gt_class[i]    # [G]
        gb = gt_bbox[i]     # [G, 4]

        # 1. 分类代价：负的softmax概率（使用真实类别）
        class_cost = -F.softmax(pc, dim=-1)[:, gc]  # [Q, G]

        # 2. L1代价：预测框与真实框的L1距离
        bbox_cost = torch.cdist(pb, gb, p=1)  # [Q, G]

        # 3. GIoU代价：1 - GIoU
        giou_cost = 1 - box_giou(pb, gb)  # [Q, G]

        # 总代价矩阵
        cost_matrix = alpha * class_cost + beta * bbox_cost + gamma * giou_cost  # [Q, G]

        # 匈牙利算法求解最优匹配
        r, c = jonker_volgenant(cost_matrix.cpu().detach().numpy())
        row_inds.append(torch.tensor(r))
        col_inds.append(torch.tensor(c))

    # 堆叠为 [B, min(Q, G)] 格式
    return torch.stack(row_inds), torch.stack(col_inds)


def test_box_giou():
    # 随机生成10个边界框
    boxes1 = torch.rand(10, 4)
    boxes2 = torch.rand(10, 4)

    boxes1[:, :2] = torch.rand(10, 2) * 0.5
    boxes1[:, 2:] = boxes1[:, :2] + torch.rand(10, 2) * 0.5

    boxes2[:, :2] = torch.rand(10, 2) * 0.5
    boxes2[:, 2:] = boxes2[:, :2] + torch.rand(10, 2) * 0.5

    giou = box_giou(boxes1, boxes2)
    f_giou = giou
    std_giou = torchvision.ops.generalized_box_iou(boxes1, boxes2)
    
    assert torch.allclose(f_giou, std_giou), "Giou计算结果不一致"
    assert giou.shape == (10, 10), "IoU矩阵维度应为(10, 10)"
    assert (giou >= -1).all() and (giou <= 1).all(), "IoU值应在[0, 1]范围内"


def test_matcher_against_manual():
    """测试匈牙利匹配器与手动实现的一致性"""
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建匹配器
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    
    # 创建随机测试数据
    batch_size = 2
    num_queries = 3
    num_classes = 2  # 修改为2，确保pred_class输出维度为3（2+1）
    num_gt = 2
    
    # 随机生成预测数据
    pred_class = torch.randn(batch_size, num_queries, num_classes + 1)  # [2,3,3]
    pred_bbox = torch.rand(batch_size, num_queries, 4)              # [2,3,4]
    
    # 随机生成真实标签（类别从1开始，0可能是背景）
    gt_class = torch.randint(1, num_classes + 1, (batch_size, num_gt))  # [2,2]
    gt_bbox = torch.rand(batch_size, num_gt, 4)                       # [2,2,4]
    
    auto_row, auto_col = matcher.match(pred_class, pred_bbox, gt_class, gt_bbox)
    manual_row, manual_col = manual_match(pred_class, pred_bbox, gt_class, gt_bbox, 
                                         alpha=1, beta=5, gamma=2)
    
    assert torch.all(auto_row == manual_row), f"预测框索引不一致: {auto_row} vs {manual_col}"
    assert torch.all(auto_col == manual_col), f"真实框索引不一致: {auto_col} vs {manual_col}"
    print("自动匹配与手动计算结果一致")


def test_jonker_volgenant():
    """测试Jonker-Volgenant算法实现"""
    # 创建一个简单的成本矩阵
    cost_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)
    
    row_indices, col_indices = jonker_volgenant(cost_matrix)
    


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

def test_matcher_cost_weight_dict():
    """测试匹配器权重字典"""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=2.0, cost_giou=3.0)
    weight_dict = matcher.get_cost_weight_dict()
    
    assert isinstance(weight_dict, dict)
    assert weight_dict["class"] == 1.0
    assert weight_dict["bbox"] == 2.0
    assert weight_dict["giou"] == 3.0