import torch
import torch.nn.functional as F
from modules.model.matcher import HungarianMatcher, jonker_volgenant
from modules.loss.detr_loss import SetCriterion
from modules.utils.box_ops import box_giou


def match(pred_class, pred_bbox, gt_class, gt_bbox, cost_class=1, cost_bbox=5, cost_giou=2):
    """执行匈牙利匹配，返回匹配索引"""
    bs, num_queries = pred_class.shape[:2]
    indices = []
    
    for i in range(bs):
        # 1. 计算分类成本 [num_queries, num_gts]
        pred_prob = pred_class[i].softmax(-1)  # [Q, C+1]
        loss_class = -pred_prob[:, gt_class[i]]  # [Q, G]
        
        # 2. 计算L1距离成本 [num_queries, num_gts]
        loss_bbox = torch.cdist(pred_bbox[i], gt_bbox[i], p=1)  # [Q, G]
        
        # 3. 计算GIoU成本 [num_queries, num_gts]
        loss_giou = -box_giou(pred_bbox[i], gt_bbox[i])  # [Q, G]
        
        # 4. 加权总成本
        C = cost_class * loss_class + \
            cost_bbox * loss_bbox + \
            cost_giou * loss_giou
        
        # 5. 使用匈牙利算法
        indices.append(jonker_volgenant(C))
    
    # 转换为张量格式
    return [(torch.as_tensor(i, dtype=torch.int64), 
            torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def manual_detr_loss(pred_class, pred_bbox, gt_class, gt_bbox):
    """修正后的DETR损失实现"""
    bs, num_queries = pred_class.shape[:2]
    num_gts = gt_bbox.shape[1]
    
    # 1. 匈牙利匹配
    indices = match(pred_class, pred_bbox, gt_class, gt_bbox)
    
    # 2. 初始化目标张量
    target_classes = torch.full((bs, num_queries), 0, dtype=torch.long, device=pred_class.device)
    
    # 3. 填充匹配的真实类别
    for batch_idx, (query_idx, gt_idx) in enumerate(indices):
        target_classes[batch_idx, query_idx] = gt_class[batch_idx, gt_idx]
    
    # 4. 分类损失 (所有预测框)
    pred_class = pred_class.flatten(0, 1)  # [B*Q, C]
    target_classes = target_classes.flatten()  # [B*Q]
    loss_ce = F.cross_entropy(pred_class, target_classes, reduction='none')
    
    # 5. 边界框损失 (仅匹配的预测框)
    loss_bbox = torch.tensor(0., device=pred_bbox.device)
    loss_giou = torch.tensor(0., device=pred_bbox.device)
    num_matches = 0
    
    for batch_idx, (query_idx, gt_idx) in enumerate(indices):
        if len(query_idx) == 0:
            continue
            
        # 提取匹配的预测框和真实框
        matched_preds = pred_bbox[batch_idx, query_idx]
        matched_gts = gt_bbox[batch_idx, gt_idx]
        
        # L1损失
        loss_bbox += F.l1_loss(matched_preds, matched_gts, reduction='sum')
        
        # GIoU损失
        giou = box_giou(matched_preds, matched_gts)
        loss_giou += (1 - giou.diag()).sum()
        
        num_matches += len(query_idx)
    
    # 6. 归一化边界框损失
    if num_matches > 0:
        loss_bbox /= num_matches
        loss_giou /= num_matches
    
    # 7. 加权求和 (DETR标准权重)
    total_loss = loss_ce.mean() + 5.0 * loss_bbox + 2.0 * loss_giou
    
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

    