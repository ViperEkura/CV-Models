import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import ImageDraw

def plot_detection(model, img, device):
    model.eval()
    with torch.no_grad():
        img_tensor = img.unsqueeze(0).to(device)
        outputs = model(img_tensor)
        
        pred_boxes = outputs['pred_boxes'][0].cpu()
        pred_logits = outputs['pred_logits'][0].cpu()
        
        img_display = F.to_pil_image(img)
        draw = ImageDraw.Draw(img_display)
        
        for box, logit in zip(pred_boxes, pred_logits):
            img_w, img_h = img_display.size
            xmin = box[0] * img_w
            ymin = box[1] * img_h
            xmax = box[2] * img_w
            ymax = box[3] * img_h
            
            prob = torch.softmax(logit, dim=0)
            score, class_idx = torch.max(prob, dim=0)
            
            if score > 0.5:
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                draw.text((xmin, ymin), f"{class_idx}: {score:.2f}", fill="blue")
        
        plt.imshow(img_display)
        plt.axis('off')
        plt.show()