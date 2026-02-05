import torch
import torch.nn as nn
from .backbone import FrozenDINOv3
from .detr_head import SimpleDETRHead


class DINOv3DETR(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = FrozenDINOv3()
        self.detr_head = SimpleDETRHead(num_classes=num_classes)

        # 添加一个用于输出格式兼容的方法
        self.num_classes = num_classes

    def forward(self, x):
        # DINOv3提取特征（冻结）
        with torch.no_grad():
            features = self.backbone(x)

        # DETR头预测
        pred_logits, pred_boxes = self.detr_head(features)

        # DETR头预测
        return pred_logits, pred_boxes