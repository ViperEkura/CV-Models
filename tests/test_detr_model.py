import torch
import pytest
from modules.model.detr import DETR


def create_detr_model():
    """创建DETR模型实例"""
    num_classes = 91  # COCO数据集类别数
    return DETR(num_classes=num_classes)


def create_sample_input():
    """创建样本输入数据"""
    batch_size, channels, height, width = 2, 3, 224, 224
    return torch.randn(batch_size, channels, height, width)


def create_single_image_input():
    """创建单张图像输入数据"""
    channels, height, width = 3, 244, 244
    return torch.randn(channels, height, width)


def test_model_initialization():
    """测试模型初始化"""
    detr_model = create_detr_model()
    assert isinstance(detr_model, DETR)
    assert hasattr(detr_model, 'backbone')
    assert hasattr(detr_model, 'transformer')
    assert detr_model.query_pos.shape[0] == detr_model.num_queries
    assert detr_model.query_pos.dim() == 2  # 应该是2维参数 [num_queries, hidden_dim]


def test_forward_pass_batch_input():
    """测试批量输入的前向传播"""
    detr_model = create_detr_model()
    sample_input = create_sample_input()
    
    detr_model.eval()  # 设置为评估模式
    with torch.no_grad():
        pred_class, pred_bbox = detr_model(sample_input)
    
    batch_size = sample_input.shape[0]
    
    # 检查输出形状
    assert pred_class.shape == (batch_size, detr_model.num_queries, detr_model.linear_class.out_features)
    assert pred_bbox.shape == (batch_size, detr_model.num_queries, 4)
    
    # 检查边界框值在[0,1]范围内（因为使用了sigmoid）
    assert torch.all(pred_bbox >= 0)
    assert torch.all(pred_bbox <= 1)


def test_forward_pass_single_image():
    """测试单张图像输入的前向传播"""
    detr_model = create_detr_model()
    single_image_input = create_single_image_input()
    
    detr_model.eval()  # 设置为评估模式
    with torch.no_grad():
        pred_class, pred_bbox = detr_model(single_image_input)
    
    # 检查输出形状 - 单张图像应该自动扩展batch维度
    assert pred_class.shape == (1, detr_model.num_queries, detr_model.linear_class.out_features)
    assert pred_bbox.shape == (1, detr_model.num_queries, 4)
    
    # 检查边界框值在[0,1]范围内
    assert torch.all(pred_bbox >= 0)
    assert torch.all(pred_bbox <= 1)



@pytest.mark.parametrize("num_classes", [10, 50, 91])
def test_different_num_classes(num_classes):
    """测试不同类别数的模型初始化"""
    model = DETR(num_classes=num_classes)
    assert model.linear_class.out_features == num_classes + 1  # +1 for background