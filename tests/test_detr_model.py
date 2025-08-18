import torch
import pytest
from modules.model.detr import DETR

@pytest.fixture
def detr_model():
    # 创建一个小型的DETR模型用于测试
    model = DETR(
        num_classes=91,  # COCO数据集类别数
        num_queries=10,  # 减少查询数量以加快测试
        hidden_dim=256,
        num_encoder_layers=2,  # 减少层数以加快测试
        num_decoder_layers=2
    )
    return model

@pytest.fixture
def sample_input():
    # 创建一个示例输入张量 (batch_size, channels, height, width)
    return torch.randn(2, 3, 224, 224)

def test_model_initialization():
    """测试模型初始化"""
    model = DETR(
        num_classes=91,
        num_queries=10,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    assert model is not None
    assert isinstance(model, DETR)

def test_forward_pass(detr_model, sample_input):
    """测试前向传播"""
    detr_model.eval()
    with torch.no_grad():
        pred_class, pred_bbox = detr_model(sample_input)
    
    assert isinstance(pred_class, torch.Tensor)
    assert isinstance(pred_bbox, torch.Tensor)
    assert torch.all( pred_bbox > 0)
    assert torch.all(pred_class.isnan() | pred_class.isnan())
    
    batch_size = sample_input.shape[0]

    num_queries = detr_model.query_pos.shape[0] if hasattr(detr_model, 'query_pos') else 100
    num_classes = detr_model.class_embed.out_features
    
    # 检查输出形状
    assert pred_class.shape == (batch_size, num_queries, num_classes)
    assert pred_bbox.shape == (batch_size, num_queries, 4)

