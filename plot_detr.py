import torch
import matplotlib.pyplot as plt
from modules.utils.plot_detection import plot_detection
from modules.model.detr import DETR

# 加载训练好的DETR模型并进行预测绘制
def load_and_predict(model_path, img_path, confidence_threshold=0.5, device=torch.device('cpu')):
    """
    加载训练好的DETR模型并绘制预测结果
    
    Args:
        model_path: 模型文件路径
        img_path: 图像文件路径
        confidence_threshold: 置信度阈值
        device: 计算设备
    """
    # 加载模型
    model_dict = torch.load(model_path, map_location=device)
    model = DETR(num_classes=100)
    model.load_state_dict(model_dict)
    model.eval()
    
    # 加载图像
    from PIL import Image
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    
    # 绘制检测结果
    result_img = plot_detection(model, img, confidence_threshold, device)
    
    return result_img

if __name__ == "__main__":
    # 使用示例
    model_path = "detr.pth"
    img_path = r"data\voc\VOC2012\JPEGImages\2010_004900.jpg"  # 请替换为实际的测试图像路径
    
    result = load_and_predict(model_path, img_path, confidence_threshold=0.2)